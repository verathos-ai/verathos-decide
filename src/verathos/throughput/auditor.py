"""The throughput auditor: runs every scheduled test, checks sampled answers, and publishes signed verdicts.

Per test:

1. At the selection block the auditor fixes the targets: every listed open endpoint with a declared capacity, each at
   its declared rate, but at most ``rate_growth`` times what it delivered in its last test (at most the model's
   ``first_test_rps`` before its first, and never less than ``min_test_share`` of that). A hotkey's endpoints
   together get at most ``rate_growth`` times what the hotkey delivered last, measured in first tests' worth, and at
   least one first test's worth. Neither an endpoint nor a hotkey can make the auditor send much more than it has
   shown it serves.
2. When the start block appears, the load generator drives all targets at once: a warmup drawn from
   [``warmup_s``, 2 x ``warmup_s``], then the measured window of ``test_s``. Endpoints sharing hardware, or a pool
   behind several endpoints, must deliver the sum of their rates at the same time.
3. Per endpoint: delivered = replies inside the window to requests sent no earlier than ``max_latency_s`` before it,
   per second, scaled by the share of sampled responses whose receipt is valid. Every signed sample must be well
   formed; ``answer_checks_per_test`` of them are recomputed (on the bf16 screen if there is one, whose flags the
   fp32 reference must confirm). Credit = delivered (up to the tested rate); below ``pass_fraction`` of the tested
   rate it is scaled by delivered / tested once more. A sampled answer the miner signed and the reference proves
   wrong makes the result invalid; the evidence is kept for the verdicts. The recomputed answers' differences are
   tallied per endpoint; one whose mean over its last ``drift_window_tests`` tests exceeds the model's ``mean_abs``
   gets no credit.
4. Once the selection block is final with the hash it was selected by, the test is confirmed; otherwise void. A test
   is also void when the load generator fails or starts late; one endpoint's result is void when the generator fell
   behind its schedule or ran out of local resources for it, or when the auditor could not evaluate it.

An endpoint is credited from the latest confirmed test only: one that was not among its targets has no credit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import secrets
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bittensor_wallet import Keypair

from verathos import catalog as catalog_module
from verathos.miner.engine import Engine
from verathos.protocol.status import ThroughputOutcome
from verathos.throughput import loadgen
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.feed import (
    MALFORMED,
    MAX_EVIDENCE_BYTES,
    Credit,
    Evidence,
    ReferenceUnavailable,
    SignedVerdicts,
    Verdicts,
    fault_of,
    parse_response,
    readable,
    recompute,
    signed_answers,
)
from verathos.throughput.follow import POLL_S, ScheduleFollower
from verathos.throughput.policy import AuditPolicy, ScheduledTest, Session, session_headers
from verathos.throughput.workload import WORDS
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.config import ReferenceEngine

log = logging.getLogger(__name__)
START_DELAY_MS = 500  # from observing the start block to the first request
MAX_LATE_START_MS = 1000
LATE_SEND_SHARE = 0.01  # more late sends than this voids an endpoint's result
REFERENCE_CONCURRENCY = 4
REFERENCE_TIMEOUT_S = 30.0
RECOMPUTE_BUDGET_S = 900.0  # after this, a test's remaining recomputes are skipped
BREAKER_FAILURES = 3  # an engine failing this often in a row is not asked again in the test
KEEP_EPOCHS = 30
SCHEMA_VERSION = 2
Runner = Callable[..., Awaitable[loadgen.LoadResult]]


class BoundedEngine:
    """An engine that takes at most ``concurrency`` requests at once (the reference GPU is shared by all checks)."""

    def __init__(self, engine: Engine, concurrency: int = REFERENCE_CONCURRENCY) -> None:
        self.engine, self._slots = engine, asyncio.Semaphore(concurrency)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        async with self._slots:
            return await self.engine.answer(request)


@dataclass(frozen=True)
class Checks:
    """Answer checks per model: the fp32 reference, and optionally a faster screen whose flags the reference must
    confirm before an answer counts as wrong."""

    references: dict[str, Engine]
    screens: dict[str, Engine]

    def for_test(self, budget_s: float = RECOMPUTE_BUDGET_S) -> Recomputes:
        return Recomputes(self, time.monotonic() + budget_s)


class Recomputes:
    """The answer checks of one test: recomputes stop at the test's deadline, and an engine that fails
    ``BREAKER_FAILURES`` times in a row is not asked again in it."""

    def __init__(self, checks: Checks, deadline: float) -> None:
        self.checks, self.deadline = checks, deadline
        self._failures: dict[int, int] = {}

    async def _compare(self, engine: Engine, model_id: str, request: dict[str, Any], got: dict[str, Any],
                       tolerance: Tolerance) -> RequestComparison | None:
        if time.monotonic() > self.deadline or self._failures.get(id(engine), 0) >= BREAKER_FAILURES:
            return None
        try:
            comparison = await recompute(request, got, engine, tolerance)
        except ReferenceUnavailable as exc:
            self._failures[id(engine)] = self._failures.get(id(engine), 0) + 1
            log.warning("a %s recompute failed, the answer is unchecked: %s", model_id, exc)
            return None
        self._failures[id(engine)] = 0
        return comparison

    async def check(self, model_id: str, request: dict[str, Any], answers: Any, tolerance: Tolerance, *,
                    recompute_answers: bool) -> tuple[str | None, RequestComparison | None]:
        """-> (why the answers are wrong or None, the comparison made or None). Without ``recompute_answers`` only
        their form is checked."""
        got = readable(answers)
        if got is None:
            return MALFORMED, None
        reference = self.checks.references.get(model_id)
        screen = self.checks.screens.get(model_id)
        first = screen or reference
        if not recompute_answers or first is None:
            return None, None
        comparison = await self._compare(first, model_id, request, got, tolerance)
        if comparison is None:
            return None, None
        fault = fault_of(comparison, tolerance)
        if fault is None or screen is None:
            return fault, comparison
        if reference is None:
            log.warning("the %s screen flagged an answer but there is no fp32 reference to confirm it", model_id)
            return None, comparison
        confirmed = await self._compare(reference, model_id, request, got, tolerance)
        if confirmed is None:
            return None, comparison
        return fault_of(confirmed, tolerance), confirmed


@dataclass(frozen=True)
class Slot:
    """An endpoint the auditor tests."""

    hotkey: str
    endpoint_id: str
    url: str
    model_id: str
    declared_rps: float

    @property
    def key(self) -> str:
        return f"{self.hotkey}/{self.endpoint_id}"


@dataclass(frozen=True)
class Target:
    slot: Slot
    tested_rps: float


@dataclass(frozen=True)
class Result:
    target: Target
    outcome: str  # passed | partial | invalid | void
    reason: str = ""
    delivered_rps: float = 0.0
    credited_rps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    detail: str = "{}"
    evidence: Evidence | None = None
    drift_sum: float = 0.0  # summed absolute differences of the recomputed answers
    drift_count: int = 0


_SCHEMA = """
CREATE TABLE tests (
    test_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, selection_block INTEGER NOT NULL, selection_hash TEXT NOT NULL,
    start_block INTEGER NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '');
CREATE TABLE results (
    test_id TEXT NOT NULL, hotkey TEXT NOT NULL, endpoint_id TEXT NOT NULL, model_id TEXT NOT NULL,
    declared_rps REAL NOT NULL, tested_rps REAL NOT NULL, delivered_rps REAL NOT NULL, credited_rps REAL NOT NULL,
    outcome TEXT NOT NULL, reason TEXT NOT NULL, latency_p50_ms REAL NOT NULL, latency_p95_ms REAL NOT NULL,
    detail TEXT NOT NULL, evidence TEXT, drift_sum REAL NOT NULL, drift_count INTEGER NOT NULL,
    PRIMARY KEY (test_id, hotkey, endpoint_id));
"""
# test states: running -> unconfirmed -> confirmed | void


class Store:
    """Tests and results, kept for ``KEEP_EPOCHS`` epochs. A database of another schema version is started over."""

    def __init__(self, path: Path) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        if self._db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            self._db.executescript("DROP TABLE IF EXISTS results; DROP TABLE IF EXISTS tests;" + _SCHEMA
                                   + f"PRAGMA user_version = {SCHEMA_VERSION};")
            log.info("auditor database %s started for schema %d", path, SCHEMA_VERSION)
        self._query("UPDATE tests SET state = 'void', reason = 'the auditor restarted during the test' "
                    "WHERE state = 'running'")

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    def known(self, test_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM tests WHERE test_id = ?", (test_id,)))

    def add_test(self, test: ScheduledTest) -> None:
        self._query("INSERT INTO tests (test_id, epoch, selection_block, selection_hash, start_block, state) "
                    "VALUES (?, ?, ?, ?, ?, 'running')",
                    (test.test_id, test.epoch, test.selection_block, test.selection_hash, test.start_block))

    def set_state(self, test_id: str, state: str, reason: str = "") -> None:
        self._query("UPDATE tests SET state = ?, reason = ? WHERE test_id = ?", (state, reason, test_id))

    def add_results(self, test_id: str, results: list[Result]) -> None:
        rows = [(test_id, r.target.slot.hotkey, r.target.slot.endpoint_id, r.target.slot.model_id,
                 r.target.slot.declared_rps, r.target.tested_rps, r.delivered_rps, r.credited_rps, r.outcome,
                 r.reason, r.latency_p50_ms, r.latency_p95_ms, r.detail,
                 r.evidence.model_dump_json() if r.evidence else None, r.drift_sum, r.drift_count) for r in results]
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany("INSERT OR REPLACE INTO results VALUES "
                                 "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self._db.execute("UPDATE tests SET state = 'unconfirmed' WHERE test_id = ?", (test_id,))
            self._db.execute("COMMIT")

    def unconfirmed(self) -> list[tuple[str, int, str]]:
        """-> [(test_id, selection_block, selection_hash)]"""
        return self._query("SELECT test_id, selection_block, selection_hash FROM tests WHERE state = 'unconfirmed'")

    def last_delivered(self) -> dict[str, tuple[str, float]]:
        """Endpoint key -> (model, delivered rate) in its latest counted result."""
        rows = self._query(
            "SELECT r.hotkey, r.endpoint_id, r.model_id, r.delivered_rps FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state != 'void' AND r.outcome != 'void' ORDER BY t.epoch, t.start_block")
        return {f"{hk}/{ep}": (model, rps) for hk, ep, model, rps in rows}

    def pending(self, first_epoch: int, last_epoch: int) -> bool:
        return bool(self._query("SELECT 1 FROM tests WHERE epoch BETWEEN ? AND ? AND state IN "
                                "('running', 'unconfirmed')", (first_epoch, last_epoch)))

    def credits(self, first_epoch: int, last_epoch: int, policy: AuditPolicy,
                tolerance_of: Callable[[str], Tolerance | None]) -> list[Credit]:
        """The endpoints of the latest confirmed test in the epoch range, with their credit; an endpoint whose result
        there is void keeps its latest counted result from the range."""
        latest = self._query("SELECT test_id FROM tests WHERE state = 'confirmed' AND epoch BETWEEN ? AND ? "
                             "ORDER BY epoch DESC, start_block DESC LIMIT 1", (first_epoch, last_epoch))
        if not latest:
            return []
        targeted = self._query("SELECT hotkey, endpoint_id FROM results WHERE test_id = ?", (latest[0][0],))
        counted = {(hk, ep): row for hk, ep, *row in self._query(
            "SELECT r.hotkey, r.endpoint_id, r.model_id, r.credited_rps, r.latency_p50_ms, t.epoch FROM results r "
            "JOIN tests t USING (test_id) WHERE t.state = 'confirmed' AND r.outcome != 'void' "
            "AND t.epoch BETWEEN ? AND ? ORDER BY t.epoch, t.start_block", (first_epoch, last_epoch))}
        passed = {(hk, ep): n for hk, ep, n in self._query(
            "SELECT r.hotkey, r.endpoint_id, COUNT(*) FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state = 'confirmed' AND r.credited_rps > 0 AND t.epoch <= ? GROUP BY r.hotkey, r.endpoint_id",
            (last_epoch,))}
        credits = []
        for hk, ep in sorted(targeted):
            if (hk, ep) not in counted:
                continue
            model_id, credited, latency, epoch = counted[hk, ep]
            drift = self._drifted(hk, ep, model_id, last_epoch, policy.drift_window_tests, tolerance_of(model_id))
            credits.append(Credit(hotkey=hk, endpoint_id=ep, model_id=model_id, credited_rps=0.0 if drift else credited,
                                  latency_ms=latency, test_epoch=epoch, drift=drift,
                                  qualified=passed.get((hk, ep), 0) >= policy.qualification_tests))
        return credits

    def _drifted(self, hotkey: str, endpoint_id: str, model_id: str, last_epoch: int, tests: int,
                 tolerance: Tolerance | None) -> bool:
        """Whether the endpoint's recomputed answers over its last ``tests`` confirmed tests drift beyond the
        model's mean tolerance (once they cover enough probabilities)."""
        if tolerance is None:
            return False
        rows = self._query(
            "SELECT r.drift_sum, r.drift_count FROM results r JOIN tests t USING (test_id) WHERE r.hotkey = ? "
            "AND r.endpoint_id = ? AND r.model_id = ? AND t.state = 'confirmed' AND t.epoch <= ? "
            "ORDER BY t.epoch DESC, t.start_block DESC LIMIT ?", (hotkey, endpoint_id, model_id, last_epoch, tests))
        total, count = sum(r[0] for r in rows), sum(r[1] for r in rows)
        return count >= tolerance.min_probabilities and total / count > tolerance.mean_abs

    def evidence(self, first_epoch: int, last_epoch: int) -> list[Evidence]:
        """The first wrong answer proven against each hotkey in the confirmed tests of the epoch range."""
        rows = self._query(
            "SELECT r.hotkey, r.evidence FROM results r JOIN tests t USING (test_id) WHERE t.state = 'confirmed' "
            "AND r.evidence IS NOT NULL AND t.epoch BETWEEN ? AND ? ORDER BY t.epoch, r.hotkey, r.endpoint_id",
            (first_epoch, last_epoch))
        first: dict[str, str] = {}
        for hotkey, evidence in rows:
            first.setdefault(hotkey, evidence)
        return [Evidence.model_validate_json(e) for e in first.values()]

    def outcomes(self, hotkey: str, since_epoch: int) -> list[ThroughputOutcome]:
        rows = self._query(
            "SELECT r.endpoint_id, t.epoch, t.start_block, t.state, r.outcome, "
            "CASE WHEN t.state = 'void' THEN t.reason ELSE r.reason END, r.declared_rps, r.tested_rps, "
            "r.delivered_rps, r.credited_rps, r.latency_p50_ms, r.latency_p95_ms "
            "FROM results r JOIN tests t USING (test_id) WHERE r.hotkey = ? AND t.epoch >= ? "
            "ORDER BY t.epoch, r.endpoint_id", (hotkey, since_epoch))
        keys = ("endpoint_id", "epoch", "start_block", "state", "outcome", "reason", "declared_rps", "tested_rps",
                "delivered_rps", "credited_rps", "latency_p50_ms", "latency_p95_ms")
        return [ThroughputOutcome(**dict(zip(keys, row))) for row in rows]

    def prune(self, before_epoch: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM results WHERE test_id IN (SELECT test_id FROM tests WHERE epoch < ?)",
                             (before_epoch,))
            self._db.execute("DELETE FROM tests WHERE epoch < ?", (before_epoch,))


def planned_rate(slot: Slot, first_test_rps: float, last_delivered: float | None, policy: AuditPolicy) -> float:
    """The rate an endpoint is tested at, before its hotkey's budget."""
    if last_delivered is None:
        ceiling = first_test_rps
    else:
        ceiling = max(policy.min_test_share * first_test_rps, policy.rate_growth * last_delivered)
    return round(min(slot.declared_rps, ceiling), 2)


def credit(tested_rps: float, delivered_rps: float, policy: AuditPolicy) -> float:
    """What was delivered, up to the tested rate; below ``pass_fraction`` of the tested rate, scaled by delivered /
    tested once more."""
    delivered = min(delivered_rps, tested_rps)
    if delivered >= policy.pass_fraction * tested_rps:
        return round(delivered, 2)
    return round(delivered * delivered / tested_rps, 2)


class Auditor:
    def __init__(self, *, netuid: int, mechid: int, keypair: Keypair, chain: SubtensorChainView,
                 policy: AuditPolicy, catalog: Callable[[], catalog_module.Catalog],
                 references: dict[str, ReferenceEngine], db_path: Path, runner: Runner = loadgen.run,
                 screens: dict[str, ReferenceEngine] | None = None) -> None:
        """``references`` recompute answers in fp32; a model with an entry in ``screens`` is screened there first
        (bf16), and the fp32 reference only confirms what the screen flags."""
        self.netuid, self.mechid, self.keypair, self.chain = netuid, mechid, keypair, chain
        self.hotkey = keypair.ss58_address
        self.policy, self.catalog, self.runner = policy, catalog, runner
        self.reference_specs, self.screen_specs = references, screens or {}
        self.store = Store(db_path)
        self.follower = ScheduleFollower(chain, policy)
        self._slots: list[Slot] = []
        self._verdicts: dict[int, SignedVerdicts] = {}
        self._pruned_epoch = -1

    # ---- inputs ----

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt a new policy for the tests and verdicts to come; verdicts already served stay as they are."""
        self.policy = policy
        self.follower.set_policy(policy)

    def update_slots(self, slots: list[Slot]) -> None:
        self._slots = slots

    def holding(self) -> bool:
        return self.follower.holding()

    # ---- running ----

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self.run()), name="throughput-auditor", daemon=True).start()

    async def run(self) -> None:
        checks = Checks(
            {m: BoundedEngine(r.engine(REFERENCE_TIMEOUT_S)) for m, r in self.reference_specs.items()},
            {m: BoundedEngine(r.engine(REFERENCE_TIMEOUT_S)) for m, r in self.screen_specs.items()})
        tasks: set[asyncio.Task] = set()
        while True:
            try:
                for test in await asyncio.to_thread(self.follower.poll):
                    task = asyncio.create_task(self.run_test(test, checks))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                await asyncio.to_thread(self.settle)
            except Exception:
                log.exception("auditor step failed")
            await asyncio.sleep(POLL_S)

    def targets(self, policy: AuditPolicy) -> list[Target]:
        cat, last = self.catalog(), self.store.last_delivered()
        by_hotkey: dict[str, list[tuple[Slot, catalog_module.CatalogModel]]] = {}
        for slot in self._slots:
            model = cat.get(slot.model_id)
            if model is not None and model.status != "retired":
                by_hotkey.setdefault(slot.hotkey, []).append((slot, model))
        out = []
        for hotkey, slots in by_hotkey.items():
            rates = [planned_rate(slot, model.first_test_rps, last.get(slot.key, (None, None))[1], policy)
                     for slot, model in slots]
            # The hotkey's budget in first tests' worth, from what all its endpoints (renamed ones too) delivered.
            delivered = sum(rps / cat.get(m).first_test_rps for key, (m, rps) in last.items()
                            if key.startswith(hotkey + "/") and cat.get(m) is not None)
            budget = max(1.0, policy.rate_growth * delivered)
            units = sum(rate / model.first_test_rps for rate, (_, model) in zip(rates, slots))
            scale = min(1.0, budget / units) if units else 1.0
            out += [Target(slot, round(rate * scale, 2)) for rate, (slot, _) in zip(rates, slots)
                    if round(rate * scale, 2) > 0]
        return out

    def plan(self, test: ScheduledTest, targets: list[Target], start_ms: int, policy: AuditPolicy) -> dict[str, Any]:
        cat, not_after_ms = self.catalog(), start_ms + int(policy.session_s * 1000)
        planned = []
        for t in targets:
            workload = cat.get(t.slot.model_id).workload
            session = Session(netuid=self.netuid, auditor=self.hotkey, miner_hotkey=t.slot.hotkey,
                              endpoint_id=t.slot.endpoint_id, test_id=test.test_id, not_after_ms=not_after_ms)
            planned.append({
                "id": t.slot.key, "url": t.slot.url + "/v1/systemone", "rate_rps": t.tested_rps,
                "max_outstanding": math.ceil(t.tested_rps * policy.request_timeout_s) + 16,
                "headers": session_headers(session, self.keypair), "model": t.slot.model_id,
                "state_words": list(workload.state_words), "questions": workload.questions,
                "samples": policy.samples_per_test,
            })
        # The miner cannot know when the window opens, so it cannot time held-back answers to it.
        warmup = round(policy.warmup_s * (1 + secrets.SystemRandom().random()), 3)
        return {"start_at_ms": start_ms, "warmup_s": warmup, "test_s": policy.test_s, "grace_s": policy.max_latency_s,
                "timeout_s": policy.request_timeout_s, "max_response_bytes": MAX_EVIDENCE_BYTES,
                "words": list(WORDS), "seed": secrets.token_hex(16), "targets": planned}

    async def run_test(self, test: ScheduledTest, checks: Checks) -> None:
        """Run one test; any failure of the auditor itself voids it."""
        if self.store.known(test.test_id) or self.follower.head > test.start_block:
            return  # already run, or selected before this auditor was following
        try:
            await self._run_test(test, checks)
        except Exception as exc:
            log.exception("throughput test of epoch %d failed", test.epoch)
            if self.store.known(test.test_id):
                self.store.set_state(test.test_id, "void", f"the auditor failed: {type(exc).__name__}")

    async def _run_test(self, test: ScheduledTest, checks: Checks) -> None:
        policy = self.policy
        targets = self.targets(policy)
        if not targets:
            log.info("throughput test of epoch %d: no endpoints to test", test.epoch)
            return
        self.store.add_test(test)
        log.info("throughput test of epoch %d selected at block %d: %d endpoints, %.0f requests/s, start block %d",
                 test.epoch, test.selection_block, len(targets), sum(t.tested_rps for t in targets),
                 test.start_block)
        while self.follower.head < test.start_block:
            await asyncio.sleep(0.25)
        if self.follower.head > test.start_block + 1:
            self.store.set_state(test.test_id, "void", "the auditor observed the start block too late")
            return
        start_ms = int(time.time() * 1000) + START_DELAY_MS
        try:
            result = await self.runner(self.plan(test, targets, start_ms, policy), timeout_s=policy.session_s + 30.0)
        except loadgen.LoadgenError as exc:
            log.error("throughput test of epoch %d void: load generator failed: %s", test.epoch, exc)
            self.store.set_state(test.test_id, "void", "the auditor's load generator failed")
            return
        if result.late_start_ms > MAX_LATE_START_MS:
            self.store.set_state(test.test_id, "void", "the auditor's load generator started late")
            return
        recomputes = checks.for_test()
        results = await asyncio.gather(*(self.evaluate(test, t, r, recomputes, policy)
                                         for t, r in zip(targets, result.targets)))
        self.store.add_results(test.test_id, results)
        counts = {o: sum(r.outcome == o for r in results) for o in ("passed", "partial", "invalid", "void")}
        log.info("throughput test of epoch %d done: %s; %.0f of %.0f requests/s credited", test.epoch,
                 ", ".join(f"{n} {o}" for o, n in counts.items() if n), sum(r.credited_rps for r in results),
                 sum(t.tested_rps for t in targets))

    async def evaluate(self, test: ScheduledTest, target: Target, r: loadgen.TargetResult, recomputes: Recomputes,
                       policy: AuditPolicy) -> Result:
        """One endpoint's result; whatever its responses contain, a failure here voids only this endpoint."""
        try:
            return await self._evaluate(test, target, r, recomputes, policy)
        except Exception:
            log.exception("evaluating %s in the test of epoch %d failed", target.slot.key, test.epoch)
            return Result(target, "void", "the auditor could not evaluate this endpoint")

    async def _evaluate(self, test: ScheduledTest, target: Target, r: loadgen.TargetResult, recomputes: Recomputes,
                        policy: AuditPolicy) -> Result:
        detail = json.dumps({"sent": r.sent, "completed": r.completed, "skipped": r.skipped, "errors": r.errors,
                             "late_sends": r.late_sends, "samples": len(r.samples)}, sort_keys=True)
        latency = {"latency_p50_ms": r.latency_ms.p50 or 0.0, "latency_p95_ms": r.latency_ms.p95 or 0.0,
                   "detail": detail}
        if r.late_sends > LATE_SEND_SHARE * max(r.sent, 1):
            return Result(target, "void", "the auditor's load generator fell behind", **latency)
        if r.errors.get("local"):
            return Result(target, "void", "the auditor's load generator ran out of local resources", **latency)
        slot, model = target.slot, self.catalog().get(target.slot.model_id)
        signed = []  # (sample, answers) whose receipt is valid
        for sample in r.samples:
            response = parse_response(sample.response)
            answers = None if response is None else signed_answers(
                response, sample.request, request_id=sample.request_id, hotkey=slot.hotkey,
                endpoint_id=slot.endpoint_id, model_id=slot.model_id, auditor=self.hotkey, netuid=self.netuid,
                mechid=self.mechid)
            if answers is not None:
                signed.append((sample, answers))
        valid, evidence, drift_sum, drift_count = len(signed), None, 0.0, 0
        recomputed = {id(s) for s, _ in random.sample(signed, min(policy.answer_checks_per_test, len(signed)))}
        for sample, answers in signed if model is not None else []:
            fault, comparison = await recomputes.check(slot.model_id, sample.request, answers, model.tolerance,
                                                       recompute_answers=id(sample) in recomputed)
            if comparison is not None:
                drift_sum, drift_count = drift_sum + comparison.sum_abs, drift_count + comparison.count
            if fault:
                evidence = Evidence(hotkey=slot.hotkey, endpoint_id=slot.endpoint_id, model_id=slot.model_id,
                                    test_id=test.test_id, reason=fault, request_id=sample.request_id,
                                    request=sample.request, response=sample.response)
                break
        drift = {"drift_sum": drift_sum, "drift_count": drift_count}
        delivered = round(r.completed / policy.test_s * (valid / len(r.samples) if r.samples else 1.0), 2)
        if evidence is not None:
            return Result(target, "invalid", evidence.reason, delivered, 0.0, evidence=evidence, **latency, **drift)
        credited = credit(target.tested_rps, delivered, policy)
        if delivered >= policy.pass_fraction * target.tested_rps:
            return Result(target, "passed", "", delivered, credited, **latency, **drift)
        reason = f"delivered {delivered:.1f} of {target.tested_rps:.1f} requests/s"
        if valid < len(r.samples):
            reason += f", {len(r.samples) - valid} of {len(r.samples)} sampled responses without a valid receipt"
        return Result(target, "partial", reason, delivered, credited, **latency, **drift)

    def settle(self) -> None:
        """Confirm tests whose selection block is final with the hash they were selected by; void the others."""
        epoch = self.follower.head // self.policy.epoch_blocks
        if epoch != self._pruned_epoch:
            self.store.prune(epoch - KEEP_EPOCHS)
            self._pruned_epoch = epoch
        unconfirmed = self.store.unconfirmed()
        if not unconfirmed:
            return
        finalized = self.chain.finalized_block()
        for test_id, block, block_hash in unconfirmed:
            if block > finalized:
                continue
            final_hash = self.chain.block_hash(block)
            if final_hash is None:
                continue
            if final_hash.hex() == block_hash:
                self.store.set_state(test_id, "confirmed")
            else:
                self.store.set_state(test_id, "void", "the selection block was not finalized")

    # ---- outputs ----

    def verdicts(self, epoch: int) -> SignedVerdicts | None:
        """The signed verdicts for scoring epoch ``epoch`` (the current one or the two before); None otherwise, or
        while a test they cover is not settled. Once served, an epoch's verdicts do not change."""
        cached = self._verdicts.get(epoch)
        if cached is not None:
            return cached
        policy, cat = self.policy, self.catalog()
        current, first = self.follower.head // policy.epoch_blocks, epoch - policy.credit_ttl_epochs
        if not current - 2 <= epoch <= current or self.store.pending(first, epoch - 1):
            return None

        def tolerance_of(model_id: str) -> Tolerance | None:
            model = cat.get(model_id)
            return model.tolerance if model is not None else None

        verdicts = Verdicts(netuid=self.netuid, auditor=self.hotkey, epoch=epoch,
                            credits=self.store.credits(first, epoch - 1, policy, tolerance_of),
                            invalid=self.store.evidence(first, epoch - 1))
        signed = SignedVerdicts.create(verdicts, self.keypair)
        self._verdicts = {e: v for e, v in self._verdicts.items() if e >= current - 2} | {epoch: signed}
        return signed

    def outcomes(self, hotkey: str, epoch: int) -> list[ThroughputOutcome]:
        """The hotkey's test results since ``credit_ttl_epochs + 1`` epochs before ``epoch``."""
        return self.store.outcomes(hotkey, epoch - self.policy.credit_ttl_epochs - 1)
