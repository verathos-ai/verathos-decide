"""References one at a time (``reference_mode: sequential``), for a full validator whose memory does not hold every
reference at once.

The validator checks a model by starting its reference engine (the ``command`` the installer writes into
``references``), computing the model's canary answers and stopping it before the next model's starts, so the memory
needs only the largest reference; an engine without a command is taken as running. The references run within the
same share of the epoch as resident ones (``audit.REFERENCE_SHARE`` of the canaries' time), starts and stops included,
and the canaries follow as they do after resident references. The validator learns each model's start time, time per
canary and stop time (``state_dir/references.json``: the largest of the last ``TIMINGS_KEPT`` of each) and plans each
epoch with them. A request to a stopped reference (a proven wrong answer to recheck) starts it, if its start fits in
the time left.

Coverage: the models are taken in an order drawn each epoch from the validator's canary secret, so a miner cannot tell
which models are checked. A model whose reference has not run for ``k`` epochs in a row (``k``: the models it can
check that epoch) is overdue: overdue models come first, the longest overdue first; the others follow in a random
order in which each model weighs one plus the epochs it has waited. A model is taken when its start,
``canaries_per_endpoint`` answers and stop fit in the reference time the models taken before it leave; the first
model, when overdue, is taken whatever its time and computes what it can. Each taken model computes its full canary
pool while the time the models after it need remains. The longest overdue model runs every epoch and at most
``k - 1`` models are ahead of one that is overdue, so every model runs at least once every ``2k`` epochs, and more
often when several fit in an epoch. The models not checked take the source validator's results
(``validator.fallback``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import httpx

from verathos.catalog import Catalog, CatalogModel
from verathos.miner.engine import Engine, EngineError
from verathos.validator.audit import reference_answers
from verathos.validator.canaries import seeded
from verathos.validator.config import ReferenceEngine
from verathos.validator.references import wrong_revision

log = logging.getLogger(__name__)
TIMINGS_KEPT = 5  # measurements kept per model and kind; the largest plans the epoch
KINDS = ("start_s", "answer_s", "stop_s")
START_POLL_S = 1.0
STOP_WAIT_S = 30.0  # after SIGTERM, before SIGKILL
LOG_TAIL_LINES = 20
# Runs a reference's command (in place, by exec) so that it gets SIGTERM when the validator's thread that started it
# exits (Linux), however the validator ends: no reference outlives it holding the memory and the port.
WITH_VALIDATOR = (
    "import ctypes, os, signal, sys\n"
    "try:\n"
    "    ctypes.CDLL(None).prctl(1, signal.SIGTERM)\n"
    "except Exception:\n"
    "    pass\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(1)\n"
    "os.execvp(sys.argv[2], sys.argv[2:])\n"
)


def fingerprint(spec: ReferenceEngine) -> str:
    """What a reference's measured times hold for: its URL, command, directory and environment."""
    setup = json.dumps([spec.url, spec.command, spec.directory, spec.environment], sort_keys=True)
    return hashlib.sha256(setup.encode()).hexdigest()[:16]


class Timings:
    """Each model's measured start time, time per canary and stop time, kept at ``path`` across restarts and
    forgotten when the model's reference changes (``setup``)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.models: dict[str, dict[str, Any]] = {}
        if path is None:
            return
        try:
            loaded = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("%s unreadable, learning the references' times again: %s", path, exc)
            return
        if isinstance(loaded, dict):
            self.models = {m: v for m, v in loaded.items() if isinstance(v, dict)}

    def setup(self, model: str, setup: str) -> None:
        """Forget the model's times if they were measured on another setup."""
        if self.models.get(model, {}).get("setup") != setup:
            self.models[model] = {"setup": setup}

    def record(self, model: str, kind: str, seconds: float) -> None:
        entry = self.models.setdefault(model, {})
        entry[kind] = [*entry.get(kind, []), round(seconds, 3)][-TIMINGS_KEPT:]
        self.save()

    def get(self, model: str, kind: str) -> float | None:
        """The largest of the model's last measurements of ``kind``, or None."""
        kept = self.models.get(model, {}).get(kind, [])
        values = [float(v) for v in kept if isinstance(v, (int, float)) and v >= 0] if isinstance(kept, list) else []
        return max(values) if values else None

    def cost(self, model: str, answers: int) -> float | None:
        """Seconds the model's reference takes to start, compute ``answers`` canaries and stop, as measured (a part
        not measured yet counts as none); None while nothing is."""
        start, answer, stop = (self.get(model, kind) for kind in KINDS)
        if start is None and answer is None:
            return None
        return (start or 0.0) + answers * (answer or 0.0) + (stop or 0.0)

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.models, indent=1, sort_keys=True))
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("keeping the references' times in %s failed: %s", self.path, exc)


def coverage_order(models: Collection[str], waited: Mapping[str, int], secret: bytes, epoch: int) -> list[str]:
    """The order the models are taken in this epoch (see the module): the overdue ones (waited at least as many
    epochs as there are models) first, the longest overdue first; then the others in a random order weighted by one
    plus the epochs each waited (each model's key is u^(1/weight), u uniform, highest first), drawn from ``secret``
    and ``epoch``."""
    k = len(models)
    rng = seeded(secret, "coverage", epoch)

    def epochs(model: str) -> int:
        return max(int(waited.get(model, 0)), 0)

    key = {m: rng.random() ** (1.0 / (1 + epochs(m))) for m in sorted(models)}
    overdue = sorted((m for m in models if epochs(m) >= k), key=lambda m: (-epochs(m), -key[m]))
    others = sorted((m for m in models if epochs(m) < k), key=lambda m: -key[m])
    return overdue + others


def overdue(models: Collection[str], waited: Mapping[str, int]) -> set[str]:
    return {m for m in models if waited.get(m, 0) >= len(models)}


def waited_after(waited: Mapping[str, int], served: Collection[str], tried: Collection[str],
                 known: Collection[str]) -> dict[str, int]:
    """The epochs each model has waited, after an epoch in which the ``served`` models (with endpoints and a
    reference) could run and the ``tried`` ones did; models no longer ``known`` to the catalog are forgotten."""
    out = {m: n for m, n in waited.items() if m in known}
    for model in served:
        out[model] = 0 if model in tried else out.get(model, 0) + 1
    return out


def fitting(order: list[str], cost: Callable[[str], float | None], left: float | None,
            forced: str | None = None) -> list[str]:
    """The models of ``order`` taken in it: each whose ``cost`` (None: not measured yet, taken) fits in what the
    ones taken before leave of ``left`` seconds (None: no limit), and ``forced`` whatever its cost."""
    if left is None:
        return list(order)
    out = []
    for model in order:
        need = cost(model) or 0.0
        if model == forced or (left > 0 and need <= left):
            out.append(model)
            left -= need
    return out


class StartFailed(Exception):
    """A reference did not start in time, or does not serve the pinned weights."""

    def __init__(self, why: str, *, timed_out: bool = False) -> None:
        super().__init__(why)
        self.timed_out = timed_out


class Slot:
    """The reference that runs: at most one at a time."""

    def __init__(self) -> None:
        self.running: OnDemand | None = None

    async def take(self, reference: OnDemand) -> None:
        if self.running is not None and self.running is not reference:
            await self.running.stop()
        self.running = reference

    async def release(self) -> None:
        if self.running is not None:
            await self.running.stop()


class _Timed:
    """An engine whose answers are timed."""

    def __init__(self, engine: Engine, times: list[float]) -> None:
        self.engine, self.times = engine, times

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        out = await self.engine.answer(request)
        self.times.append(time.monotonic() - started)
        return out

    async def info(self) -> Any:
        return await self.engine.info()


async def _close(engine: Any) -> None:
    try:
        await engine.close()
    except Exception:  # a client of an earlier epoch's event loop
        pass


class OnDemand:
    """A model's reference, started when it is needed and stopped after, one at a time (``Slot``). It answers like an
    engine: a request while it is stopped starts it if its measured start and one answer fit before ``deadline``
    (``time.monotonic``; None: no limit), and fails otherwise."""

    def __init__(self, model: CatalogModel, spec: ReferenceEngine, *, slot: Slot, timings: Timings,
                 log_dir: Path | None = None) -> None:
        self.model, self.spec, self.slot, self.timings, self.log_dir = model, spec, slot, timings, log_dir
        self.deadline: float | None = None
        self.engine: Engine | None = None
        self.process: subprocess.Popen | None = None
        self._log: IO[bytes] | None = None
        self._times: list[float] = []
        timings.setup(model.model_id, fingerprint(spec))

    @property
    def model_id(self) -> str:
        return self.model.model_id

    async def start(self, deadline: float | None = None) -> Engine:
        """Start the reference unless it runs, stopping the one that does, by ``deadline``, and check that it serves
        the pinned weights. Raises StartFailed."""
        if self.engine is not None:
            return self.engine
        await self.slot.take(self)
        started = time.monotonic()
        try:
            engine = await self._launch(deadline)
        except StartFailed as exc:
            if exc.timed_out:  # it takes at least this long
                self.timings.record(self.model_id, "start_s", time.monotonic() - started)
            await self.stop()
            raise
        except BaseException:  # cancelled (a recheck out of time): nothing may keep running
            self._kill()
            if self.slot.running is self:
                self.slot.running = None
            raise
        if self.spec.command:
            took = time.monotonic() - started
            self.timings.record(self.model_id, "start_s", took)
            log.info("the %s reference started in %.0f s", self.model_id, took)
        self.engine = _Timed(engine, self._times)
        if why := await wrong_revision(engine, self.model):
            await self.stop()
            raise StartFailed(why)
        return self.engine

    async def stop(self) -> None:
        """Stop the reference (if it runs) and record its times."""
        engine, self.engine = self.engine, None
        if self.slot.running is self:
            self.slot.running = None
        if self._times:
            self.timings.record(self.model_id, "answer_s", sum(self._times) / len(self._times))
            log.info("the %s reference answered %d requests, %.1f s each", self.model_id, len(self._times),
                     sum(self._times) / len(self._times))
            self._times = []
        await self._halt()
        if isinstance(engine, _Timed):
            await _close(engine.engine)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.engine is None:
            left = None if self.deadline is None else self.deadline - time.monotonic()
            need = self.timings.cost(self.model_id, 1) or 0.0
            if left is not None and (left <= 0 or need > left):
                raise EngineError("no time left in the epoch to start the reference")
            try:
                await self.start(self.deadline)
            except StartFailed as exc:
                raise EngineError(f"the reference did not start: {exc}") from None
        return await self.engine.answer(request)

    async def info(self) -> Any:
        return None if self.engine is None else await self.engine.info()

    async def close(self) -> None:
        await self.stop()

    # ---- the process ----

    async def _launch(self, deadline: float | None) -> Engine:
        """The engine once it answers: started from its command, or taken as running without one. Raises
        StartFailed."""
        spec = self.spec
        if not spec.command:
            return spec.engine()
        url = spec.url.rstrip("/") + "/v1/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            if await _answers(client, url) is not None:
                raise StartFailed(f"{spec.url} answers before the reference starts: another process holds its port")
            self._spawn()
            started = time.monotonic()
            try:
                while True:
                    code = self.process.poll()
                    if code is not None:
                        raise StartFailed(f"it exited with code {code}{self._tail()}")
                    if await _answers(client, url) == 200:
                        return spec.engine()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise StartFailed(f"it did not answer within {time.monotonic() - started:.0f} s",
                                          timed_out=True)
                    await asyncio.sleep(START_POLL_S)
            except BaseException:
                self._kill()
                raise

    def _spawn(self) -> None:
        """Start the command, its output in ``log_dir/<model>.log`` (the latest start's)."""
        spec = self.spec
        output: Any = subprocess.DEVNULL
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._log = output = open(self.log_dir / f"{self.model_id}.log", "wb")
        # Started from this (the event loop's) thread, so the reference ends with the validator.
        self.process = subprocess.Popen([sys.executable, "-c", WITH_VALIDATOR, str(os.getpid()), *spec.command],
                                        cwd=spec.directory or None, env={**os.environ, **spec.environment},
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)

    def _tail(self) -> str:
        if self.log_dir is None:
            return ""
        try:
            lines = (self.log_dir / f"{self.model_id}.log").read_text(errors="replace").splitlines()
        except OSError:
            return ""
        last = [line for line in lines if line.strip()][-LOG_TAIL_LINES:]
        return (f"; the last lines of {self.log_dir / (self.model_id + '.log')}:\n  " + "\n  ".join(last)) if last \
            else ""

    async def _halt(self) -> None:
        """Stop the process: SIGTERM, and SIGKILL after ``STOP_WAIT_S``."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            started = time.monotonic()
            process.terminate()
            while process.poll() is None and time.monotonic() - started < STOP_WAIT_S:
                await asyncio.sleep(0.2)
            if process.poll() is None:
                process.kill()
                process.wait()
            self.timings.record(self.model_id, "stop_s", time.monotonic() - started)
        self._close_log()

    def _kill(self) -> None:
        """Kill the process at once (a failed or cancelled start)."""
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


async def _answers(client: httpx.AsyncClient, url: str) -> int | None:
    """The HTTP status ``url`` answers with, or None if nothing does."""
    try:
        return (await client.get(url)).status_code
    except httpx.HTTPError:
        return None


class Sequence:
    """One epoch's references, run one at a time in the coverage order (see the module). After ``answers``: the models
    whose reference it started (``tried``), those with answers (``checked``) and why the others have none
    (``why``)."""

    def __init__(self, references: Mapping[str, OnDemand], order: list[str], *, overdue: Collection[str],
                 per_endpoint: int, timings: Timings) -> None:
        self.references = references
        self.order = [m for m in order if m in references]
        self.forced = self.order[0] if self.order and self.order[0] in overdue else None
        self.per_endpoint, self.timings = per_endpoint, timings
        self.tried: list[str] = []
        self.checked: list[str] = []
        self.why: dict[str, str] = {}

    def cost(self, model: str) -> float | None:
        """Seconds to start the model's reference, compute a canary per endpoint slot and stop it."""
        return self.timings.cost(model, self.per_endpoint)

    def expected(self, models: Collection[str], left: float | None) -> list[str]:
        """Of ``models``, those planned to run in ``left`` seconds (None: no limit), in order."""
        return fitting([m for m in self.order if m in models], self.cost, left, self.forced)

    async def answers(self, pools: Mapping[str, list[dict[str, Any]]],
                      deadline: float | None) -> dict[str, list[dict[str, dict[str, float]] | None]]:
        """The reference answers of each model's canary ``pools`` that runs by ``deadline`` (``time.monotonic``),
        one model after the other."""
        remaining = [m for m in self.order if m in pools]
        expected = set(self.expected(remaining, None if deadline is None else deadline - time.monotonic()))
        out: dict[str, list[dict[str, dict[str, float]] | None]] = {}
        while remaining:
            model = remaining.pop(0)
            left = None if deadline is None else deadline - time.monotonic()
            taken = fitting([model, *remaining], self.cost, left, self.forced)
            if model not in taken:
                self.why[model] = "no time left" if model in expected else "not drawn"
                continue
            # what the models still to come need stays theirs; the stop is this model's
            reserve = sum(self.cost(m) or 0.0 for m in taken[1:]) + (self.timings.get(model, "stop_s") or 0.0)
            until = None if deadline is None else deadline - reserve
            reference = self.references[model]
            self.tried.append(model)
            try:
                engine = await reference.start(until)
            except StartFailed as exc:
                self.why[model] = "its reference did not start"
                log.warning("the %s reference did not start: %s", model, exc)
                continue
            try:
                got = await reference_answers(pools[model], engine, until, missing_level=logging.INFO)
            finally:
                await reference.stop()
            if any(a is not None for a in got):
                out[model] = got
                self.checked.append(model)
            else:
                self.why[model] = "its reference answered no canary in time"
        return out


@dataclass
class Runner:
    """What references one at a time keep across epochs: the one that runs, the times learned, where their output
    goes."""

    slot: Slot
    timings: Timings
    log_dir: Path | None

    @classmethod
    def at(cls, state_dir: Path) -> Runner:
        return cls(Slot(), Timings(state_dir / "references.json"), state_dir / "references")

    def references(self, specs: Mapping[str, ReferenceEngine], cat: Catalog,
                   models: Collection[str]) -> dict[str, OnDemand]:
        """This epoch's references of ``models`` (catalog models with a spec)."""
        return {m: OnDemand(cat.get(m), specs[m], slot=self.slot, timings=self.timings, log_dir=self.log_dir)
                for m in models}
