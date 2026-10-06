"""A miner measures the capacity it declares.

For each endpoint: warm the engine with the model's standard request mix until two runs agree, find the lowest
concurrency that reaches nearly the highest throughput with latency within the policy's ``max_latency_s``, measure
several runs there, and declare ``capacity_margin`` times the median. Requests take the path test traffic takes,
through the miner's own service (sessions the miner signs itself), so the service's own limits are measured too. All
endpoints of a miner are measured in the same rounds, so endpoints sharing hardware each measure their real share, as
a network-wide test sees them. Results are stored with a fingerprint of what they depend on and reused until that
changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import secrets
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from bittensor_wallet import Keypair

from verathos.miner.engine import Engine, EngineError
from verathos.protocol.canonical import digest
from verathos.throughput.policy import HDR_REQUEST_ID, Session, session_headers
from verathos.throughput.workload import Workload, make_request

log = logging.getLogger(__name__)

RUN_S = 8.0
RUN_START_SHARE = 0.125  # every run first sends this share of its duration uncounted, to reach steady state
STABLE_TOLERANCE = 0.05
MAX_WARMUP_RUNS = 12
FINAL_RUNS = 5
CONCURRENCY_LADDER = tuple(2 ** i for i in range(13))  # 1 .. 4096
READY_TIMEOUT_S = 900.0
KNEE_SHARE = 0.97
SESSION_S = 240.0  # within the session lifetime the miner's replay protection covers


class MeasurementFailed(RuntimeError):
    pass


class MeasurementInterrupted(RuntimeError):
    """The measurement stopped early (a throughput test reached the miner); nothing was measured."""


@dataclass(frozen=True)
class Measurement:
    endpoint_id: str
    fingerprint: str
    measured_rps: float  # median of the final runs
    concurrency: int  # the knee: lowest concurrency within KNEE_SHARE of the best throughput
    runs: tuple[float, ...]
    measured_at: float

    def declared(self, margin: float) -> float:
        return round(self.measured_rps * margin, 2)


@dataclass(frozen=True)
class Target:
    endpoint_id: str
    model_id: str
    engine: Engine
    workload: Workload
    max_concurrency: int
    fingerprint: str


def engine_identity(info: Any) -> Any:
    """The static part of an engine's ``GET /v1/models`` document: the scalar fields of each listed model. Nested
    sections (counters, caches) change while the engine serves."""
    if not isinstance(info, dict):
        return None
    return {key: [{k: v for k, v in item.items() if not isinstance(v, dict | list)}
                  for item in info[key] if isinstance(item, dict)]
            for key in ("models", "data") if isinstance(info.get(key), list)}


def fingerprint(endpoint_config: dict[str, Any], engine_info: Any, workload: Workload) -> str:
    """What a measurement depends on: the endpoint's configuration, the engine's identity and the mix."""
    return digest({"endpoint": endpoint_config, "engine": engine_identity(engine_info),
                   "workload": workload.model_dump(mode="json")})


class ServicePath:
    """An endpoint reached through the miner's own service as test traffic, under a session the miner signs."""

    def __init__(self, client: httpx.AsyncClient, url: str, keypair: Keypair, netuid: int, endpoint_id: str) -> None:
        self.client, self.url, self.keypair, self.netuid, self.endpoint_id = client, url, keypair, netuid, endpoint_id
        self._headers: tuple[float, dict[str, str]] | None = None

    def _session(self) -> dict[str, str]:
        if self._headers is None or time.time() > self._headers[0] - 60:
            expires = time.time() + SESSION_S
            session = Session(netuid=self.netuid, auditor=self.keypair.ss58_address,
                              miner_hotkey=self.keypair.ss58_address, endpoint_id=self.endpoint_id,
                              test_id="measure-" + secrets.token_hex(8), not_after_ms=int(expires * 1000))
            self._headers = (expires, session_headers(session, self.keypair))
        return self._headers[1]

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        headers = {**self._session(), HDR_REQUEST_ID: secrets.token_hex(16)}
        try:
            r = await self.client.post(self.url, json=request, headers=headers)
        except httpx.HTTPError as e:
            raise EngineError(f"service unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"service returned {r.status_code}: {r.text[:200]}")
        return r.json()


async def run_load(target: Target, concurrency: int, run_s: float = RUN_S,
                   stop: Callable[[], bool] = lambda: False) -> float:
    """Requests/s completed during one closed-loop run at ``concurrency`` (after a short uncounted start). Raises
    MeasurementInterrupted once ``stop`` returns True."""
    rng = random.Random()
    begin = time.monotonic() + RUN_START_SHARE * run_s
    end = begin + run_s
    done = 0

    async def worker() -> None:
        nonlocal done
        while time.monotonic() < end:
            if stop():
                raise MeasurementInterrupted()
            try:
                await target.engine.answer(make_request(target.model_id, target.workload, rng))
            except EngineError:
                await asyncio.sleep(0.05)
                continue
            if begin <= time.monotonic() <= end:
                done += 1

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return done / run_s


async def wait_ready(target: Target, timeout_s: float = READY_TIMEOUT_S) -> None:
    """Wait until the endpoint answers a request of the mix (its engine may still be loading the model). Raises
    MeasurementFailed."""
    deadline, logged = time.monotonic() + timeout_s, 0.0
    while True:
        try:
            await target.engine.answer(make_request(target.model_id, target.workload, random.Random()))
            return
        except EngineError as exc:
            if time.monotonic() > deadline:
                raise MeasurementFailed(f"not answering: {exc}") from None
            if time.monotonic() - logged > 60:
                log.info("endpoint %s: waiting for the engine (%s)", target.endpoint_id, exc)
                logged = time.monotonic()
            await asyncio.sleep(5)


def knee(steps: list[tuple[int, float]]) -> tuple[int, float]:
    """The lowest concurrency within ``KNEE_SHARE`` of the highest throughput: nearly all the capacity at the
    lowest latency."""
    top = max(rps for _, rps in steps)
    return min((c, rps) for c, rps in steps if rps >= KNEE_SHARE * top)


def _ladder(target: Target) -> list[int]:
    return sorted({c for c in CONCURRENCY_LADDER if c < target.max_concurrency} | {target.max_concurrency})


async def measure_all(targets: Sequence[Target], max_latency_s: float, run_s: float = RUN_S,
                      ready_timeout_s: float = READY_TIMEOUT_S,
                      stop: Callable[[], bool] = lambda: False) -> tuple[dict[str, Measurement], dict[str, str]]:
    """Measure every endpoint in the same rounds, so endpoints sharing an engine or a GPU always compete as they do in
    a test: warm up together until every endpoint's last two runs agree, step up the concurrency ladder together
    (an endpoint past its best, or whose latency exceeds ``max_latency_s``, stays at its knee), then run the final
    rounds together at each one's knee. In a closed loop the latency is concurrency / throughput.

    -> (measurements, failures): an endpoint fails if it never answers, takes longer than ``max_latency_s`` for a
    single request, or completes nothing under load; the others are measured without it. Raises
    MeasurementInterrupted once ``stop`` returns True."""
    failures: dict[str, str] = {}
    for t, outcome in zip(targets, await asyncio.gather(*(wait_ready(t, ready_timeout_s) for t in targets),
                                                        return_exceptions=True)):
        if isinstance(outcome, MeasurementFailed):
            failures[t.endpoint_id] = str(outcome)
        elif isinstance(outcome, BaseException):
            raise outcome
    targets = [t for t in targets if t.endpoint_id not in failures]
    if not targets:
        return {}, failures

    async def round_(concurrency: list[int]) -> list[float]:
        return list(await asyncio.gather(*(run_load(t, c, run_s, stop) for t, c in zip(targets, concurrency))))

    ladders = [_ladder(t) for t in targets]
    warm = [min(16, ladder[-1]) for ladder in ladders]
    previous = await round_(warm)
    for _ in range(MAX_WARMUP_RUNS):
        current = await round_(warm)
        stable = all(p > 0 and abs(c - p) <= STABLE_TOLERANCE * p for p, c in zip(previous, current))
        previous = current
        if stable:
            break
    steps: list[list[tuple[int, float]]] = [[] for _ in targets]  # (concurrency, requests/s) within the bound
    climbing = [True] * len(targets)
    for step in range(max(len(ladder) for ladder in ladders)):
        climbing = [up and step < len(ladder) for up, ladder in zip(climbing, ladders)]
        if not any(climbing):
            break
        held = [knee(s)[0] if s else ladder[0] for s, ladder in zip(steps, ladders)]
        rates = await round_([ladder[step] if up else h for up, ladder, h in zip(climbing, ladders, held)])
        for i, rps in enumerate(rates):
            if not climbing[i]:
                continue
            c = ladders[i][step]
            if rps <= 0 or c / rps > max_latency_s:
                climbing[i] = False  # answers too slowly at this concurrency
                continue
            steps[i].append((c, rps))
            if rps < 0.9 * max(r for _, r in steps[i]):
                climbing[i] = False  # past the engine's best operating point
    best = [knee(s) if s else (ladder[0], 0.0) for s, ladder in zip(steps, ladders)]
    finals = [await round_([b[0] for b in best]) for _ in range(FINAL_RUNS)]
    measurements = {}
    for i, t in enumerate(targets):
        runs = tuple(round(f[i], 2) for f in finals)
        if not steps[i]:
            failures[t.endpoint_id] = f"one request takes longer than {max_latency_s} s"
        elif statistics.median(runs) <= 0:
            failures[t.endpoint_id] = "completed no requests under load"
        else:
            measurements[t.endpoint_id] = Measurement(t.endpoint_id, t.fingerprint, statistics.median(runs),
                                                      best[i][0], runs, time.time())
    return measurements, failures


class MeasurementStore:
    """Measurements by endpoint, kept across restarts."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def load(self) -> dict[str, Measurement]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text())
        return {k: Measurement(**{**v, "runs": tuple(v["runs"])}) for k, v in raw.items()}

    def save(self, measurements: dict[str, Measurement]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(m) for k, m in measurements.items()}, indent=1, sort_keys=True))
        tmp.replace(self.path)


def stored(targets: Sequence[Target], store: MeasurementStore) -> dict[str, Measurement] | None:
    """The stored measurements of all ``targets`` if every fingerprint still matches, else None."""
    found = store.load()
    if all(t.endpoint_id in found and found[t.endpoint_id].fingerprint == t.fingerprint for t in targets):
        return {t.endpoint_id: found[t.endpoint_id] for t in targets}
    return None


async def measure_and_store(targets: Sequence[Target], store: MeasurementStore, *, max_latency_s: float,
                            run_s: float = RUN_S, stop: Callable[[], bool] = lambda: False,
                            ) -> tuple[dict[str, Measurement], dict[str, str]]:
    """Measure all ``targets`` together and store the results (failed endpoints are not stored). Raises
    MeasurementInterrupted once ``stop`` returns True."""
    runs = MAX_WARMUP_RUNS / 3 + max(len(_ladder(t)) for t in targets) + FINAL_RUNS
    log.info("measuring the capacity of %d endpoints (about %d minutes)", len(targets),
             max(1, round(runs * run_s * (1 + RUN_START_SHARE) / 60)))
    measured, failures = await measure_all(targets, max_latency_s, run_s, stop=stop)
    store.save(measured)
    return measured, failures
