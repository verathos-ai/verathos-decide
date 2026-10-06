"""A miner measures the capacity it declares.

For each endpoint: warm the engine with the model's standard request mix until two runs agree, find the
concurrency with the highest throughput, measure several runs there, and declare ``capacity_margin`` times the
median. All endpoints of a miner are measured in the same rounds, so endpoints sharing hardware each measure their
real share, as a network-wide test sees them. Results are stored with a fingerprint of what they depend on and reused
until that changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from verathos.miner.engine import Engine, EngineError
from verathos.protocol.canonical import digest
from verathos.throughput.workload import Workload, make_request

log = logging.getLogger(__name__)

RUN_S = 8.0
RUN_START_SHARE = 0.125  # every run first sends this share of its duration uncounted, to reach steady state
STABLE_TOLERANCE = 0.05
MAX_WARMUP_RUNS = 12
FINAL_RUNS = 5
CONCURRENCY_LADDER = (4, 8, 16, 32, 64, 128, 256)
READY_TIMEOUT_S = 900.0


class MeasurementFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class Measurement:
    endpoint_id: str
    fingerprint: str
    measured_rps: float  # median of the final runs
    concurrency: int  # concurrency with the highest throughput
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


def fingerprint(endpoint_config: dict[str, Any], engine_info: Any, workload: Workload) -> str:
    """What a measurement depends on: the endpoint's configuration, the engine's self-description and the mix."""
    return digest({"endpoint": endpoint_config, "engine": engine_info, "workload": workload.model_dump(mode="json")})


async def run_load(target: Target, concurrency: int, run_s: float = RUN_S) -> float:
    """Requests/s completed during one closed-loop run at ``concurrency`` (after a short uncounted start)."""
    rng = random.Random()
    begin = time.monotonic() + RUN_START_SHARE * run_s
    end = begin + run_s
    done = 0

    async def worker() -> None:
        nonlocal done
        while time.monotonic() < end:
            try:
                await target.engine.answer(make_request(target.model_id, target.workload, rng))
            except EngineError:
                continue
            if begin <= time.monotonic() <= end:
                done += 1

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return done / run_s


async def wait_ready(target: Target, timeout_s: float = READY_TIMEOUT_S) -> None:
    """Wait until the engine answers a request of the mix (it may still be loading its model)."""
    deadline, logged = time.monotonic() + timeout_s, 0.0
    while True:
        try:
            await target.engine.answer(make_request(target.model_id, target.workload, random.Random()))
            return
        except EngineError as exc:
            if time.monotonic() > deadline:
                raise MeasurementFailed(f"endpoint {target.endpoint_id}: engine not answering: {exc}") from None
            if time.monotonic() - logged > 60:
                log.info("endpoint %s: waiting for the engine (%s)", target.endpoint_id, exc)
                logged = time.monotonic()
            await asyncio.sleep(5)


def _ladder(target: Target) -> list[int]:
    return [c for c in CONCURRENCY_LADDER if c <= target.max_concurrency] or [target.max_concurrency]


async def measure_all(targets: Sequence[Target], run_s: float = RUN_S) -> list[Measurement]:
    """Measure every endpoint in the same rounds, so endpoints sharing an engine or a GPU always compete as they do in
    a test: warm up together until every endpoint's last two runs agree, step up the concurrency ladder together
    (an endpoint past its best stays at its best), then run the final rounds together at each one's best.

    Raises MeasurementFailed if an engine never answers or answers nothing under load."""
    await asyncio.gather(*(wait_ready(t) for t in targets))

    async def round_(concurrency: list[int]) -> list[float]:
        return list(await asyncio.gather(*(run_load(t, c, run_s) for t, c in zip(targets, concurrency))))

    ladders = [_ladder(t) for t in targets]
    warm = [min(16, ladder[-1]) for ladder in ladders]
    previous = await round_(warm)
    for _ in range(MAX_WARMUP_RUNS):
        current = await round_(warm)
        stable = all(p > 0 and abs(c - p) <= STABLE_TOLERANCE * p for p, c in zip(previous, current))
        previous = current
        if stable:
            break
    best = [(ladder[0], 0.0) for ladder in ladders]
    climbing = [True] * len(targets)
    for step in range(max(len(ladder) for ladder in ladders)):
        climbing = [up and step < len(ladder) for up, ladder in zip(climbing, ladders)]
        if not any(climbing):
            break
        rates = await round_([ladder[step] if up else b[0] for up, ladder, b in zip(climbing, ladders, best)])
        for i, rps in enumerate(rates):
            if not climbing[i]:
                continue
            if rps > best[i][1]:
                best[i] = (ladders[i][step], rps)
            elif rps < 0.9 * best[i][1]:
                climbing[i] = False  # past the engine's best operating point
    finals = [await round_([b[0] for b in best]) for _ in range(FINAL_RUNS)]
    out = []
    for i, t in enumerate(targets):
        runs = tuple(round(f[i], 2) for f in finals)
        if statistics.median(runs) <= 0:
            raise MeasurementFailed(f"endpoint {t.endpoint_id}: the engine completed no requests under load")
        out.append(Measurement(t.endpoint_id, t.fingerprint, statistics.median(runs), best[i][0], runs, time.time()))
    return out


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


async def current_measurements(targets: Sequence[Target], store: MeasurementStore, *, remeasure: bool = False,
                               run_s: float = RUN_S) -> dict[str, Measurement]:
    """Stored measurements whose fingerprint still matches; all endpoints are measured again (together) if any is
    new or changed, or when ``remeasure`` is set."""
    stored = store.load()
    if not remeasure and all(stored.get(t.endpoint_id) and stored[t.endpoint_id].fingerprint == t.fingerprint
                             for t in targets):
        return {t.endpoint_id: stored[t.endpoint_id] for t in targets}
    runs = MAX_WARMUP_RUNS / 3 + len(CONCURRENCY_LADDER) + FINAL_RUNS
    log.info("measuring the capacity of %d endpoints (about %d minutes)", len(targets),
             max(1, round(runs * run_s * (1 + RUN_START_SHARE) / 60)))
    fresh = {m.endpoint_id: m for m in await measure_all(targets, run_s)}
    store.save(fresh)
    return fresh
