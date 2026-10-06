"""The throughput auditor: runs every scheduled test, checks sampled answers, and publishes signed verdicts.

Per test:

1. At the selection block the auditor fixes the targets: every listed open endpoint with a declared capacity, each at
   its declared rate, but at most ``rate_growth`` times what it delivered in its last test (at most the model's
   ``first_test_rps`` before its first). An endpoint cannot make the auditor send more than it has shown it serves.
2. When the start block appears, the load generator drives all targets at once for ``warmup_s + test_s``. Endpoints
   sharing hardware, or a pool behind several endpoints, must deliver the sum of their rates at the same time.
3. Per endpoint: delivered = requests completed in the measured window / ``test_s``, scaled by the share of sampled
   responses whose receipt is valid. Credit = the tested rate if delivered reaches ``pass_fraction`` of it, else the
   delivered rate. A sampled answer the miner signed and the reference proves wrong makes the result invalid; the
   evidence is kept for the verdicts.
4. Once the selection block is final with the same hash, the test is confirmed; otherwise void. A test is also void
   when the load generator fails or falls behind its schedule (for one endpoint: that endpoint's result).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
from verathos.miner.engine import Engine, HttpEngine
from verathos.protocol.status import ThroughputOutcome
from verathos.throughput import loadgen
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.feed import Credit, Evidence, SignedVerdicts, Verdicts, answer_fault, signed_answers
from verathos.throughput.follow import POLL_S, ScheduleFollower
from verathos.throughput.policy import AuditPolicy, ScheduledTest, Session, session_headers
from verathos.throughput.workload import WORDS

log = logging.getLogger(__name__)
START_DELAY_MS = 500  # from observing the start block to the first request
MAX_LATE_START_MS = 1000
LATE_SEND_SHARE = 0.01  # more late sends than this voids an endpoint's result
REFERENCE_CONCURRENCY = 4
KEEP_EPOCHS = 30
Runner = Callable[..., Awaitable[loadgen.LoadResult]]


class BoundedEngine:
    """An engine that takes at most ``concurrency`` requests at once (the reference GPU is shared by all checks)."""

    def __init__(self, engine: Engine, concurrency: int = REFERENCE_CONCURRENCY) -> None:
        self.engine, self._slots = engine, asyncio.Semaphore(concurrency)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        async with self._slots:
            return await self.engine.answer(request)


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


_SCHEMA = """
CREATE TABLE IF NOT EXISTS tests (
    test_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, selection_block INTEGER NOT NULL, selection_hash TEXT NOT NULL,
    start_block INTEGER NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS results (
    test_id TEXT NOT NULL, hotkey TEXT NOT NULL, endpoint_id TEXT NOT NULL, model_id TEXT NOT NULL,
    declared_rps REAL NOT NULL, tested_rps REAL NOT NULL, delivered_rps REAL NOT NULL, credited_rps REAL NOT NULL,
    outcome TEXT NOT NULL, reason TEXT NOT NULL, latency_p50_ms REAL NOT NULL, latency_p95_ms REAL NOT NULL,
    detail TEXT NOT NULL, evidence TEXT, PRIMARY KEY (test_id, hotkey, endpoint_id));
"""
# test states: running -> unconfirmed -> confirmed | void


class Store:
    """Tests and results, kept for ``KEEP_EPOCHS`` epochs."""

    def __init__(self, path: Path) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
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
                 r.evidence.model_dump_json() if r.evidence else None) for r in results]
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany("INSERT OR REPLACE INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 rows)
            self._db.execute("UPDATE tests SET state = 'unconfirmed' WHERE test_id = ?", (test_id,))
            self._db.execute("COMMIT")

    def unconfirmed(self) -> list[tuple[str, int, str]]:
        """-> [(test_id, selection_block, selection_hash)]"""
        return self._query("SELECT test_id, selection_block, selection_hash FROM tests WHERE state = 'unconfirmed'")

    def last_delivered(self) -> dict[str, float]:
        """Endpoint key -> delivered rate in its latest counted result."""
        rows = self._query(
            "SELECT r.hotkey, r.endpoint_id, r.delivered_rps FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state != 'void' AND r.outcome != 'void' ORDER BY t.epoch")
        return {f"{hk}/{ep}": rps for hk, ep, rps in rows}

    def pending(self, first_epoch: int, last_epoch: int) -> bool:
        return bool(self._query("SELECT 1 FROM tests WHERE epoch BETWEEN ? AND ? AND state IN "
                                "('running', 'unconfirmed')", (first_epoch, last_epoch)))

    def credits(self, first_epoch: int, last_epoch: int, qualification_tests: int) -> list[Credit]:
        """Per endpoint, the credit of its latest counted result in confirmed tests of the epoch range."""
        rows = self._query(
            "SELECT r.hotkey, r.endpoint_id, r.credited_rps, t.epoch FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state = 'confirmed' AND r.outcome != 'void' AND t.epoch BETWEEN ? AND ? ORDER BY t.epoch",
            (first_epoch, last_epoch))
        latest = {(hk, ep): (rps, epoch) for hk, ep, rps, epoch in rows}
        passed = {(hk, ep): n for hk, ep, n in self._query(
            "SELECT r.hotkey, r.endpoint_id, COUNT(*) FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state = 'confirmed' AND r.credited_rps > 0 AND t.epoch <= ? GROUP BY r.hotkey, r.endpoint_id",
            (last_epoch,))}
        return [Credit(hotkey=hk, endpoint_id=ep, credited_rps=rps, test_epoch=epoch,
                       qualified=passed.get((hk, ep), 0) >= qualification_tests)
                for (hk, ep), (rps, epoch) in sorted(latest.items())]

    def evidence(self, first_epoch: int, last_epoch: int) -> list[Evidence]:
        rows = self._query(
            "SELECT r.evidence FROM results r JOIN tests t USING (test_id) WHERE t.state = 'confirmed' "
            "AND r.evidence IS NOT NULL AND t.epoch BETWEEN ? AND ? ORDER BY t.epoch, r.hotkey, r.endpoint_id",
            (first_epoch, last_epoch))
        return [Evidence.model_validate_json(e) for (e,) in rows]

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
    """The rate an endpoint is tested at."""
    ceiling = first_test_rps if last_delivered is None else max(first_test_rps, policy.rate_growth * last_delivered)
    return round(min(slot.declared_rps, ceiling), 2)


def credit(tested_rps: float, delivered_rps: float, policy: AuditPolicy) -> float:
    return tested_rps if delivered_rps >= policy.pass_fraction * tested_rps else round(delivered_rps, 2)


class Auditor:
    def __init__(self, *, netuid: int, mechid: int, keypair: Keypair, chain: SubtensorChainView,
                 policy: AuditPolicy, catalog: Callable[[], catalog_module.Catalog], references: dict[str, str],
                 db_path: Path, runner: Runner = loadgen.run) -> None:
        self.netuid, self.mechid, self.keypair, self.chain = netuid, mechid, keypair, chain
        self.hotkey = keypair.ss58_address
        self.policy, self.catalog, self.reference_urls, self.runner = policy, catalog, references, runner
        self.store = Store(db_path)
        self.follower = ScheduleFollower(chain, policy)
        self._slots: list[Slot] = []
        self._verdicts: dict[int, SignedVerdicts] = {}
        self._pruned_epoch = -1

    # ---- inputs ----

    def set_policy(self, policy: AuditPolicy) -> None:
        self.policy = policy
        self.follower.set_policy(policy)
        self._verdicts = {}

    def update_slots(self, slots: list[Slot]) -> None:
        self._slots = slots

    def holding(self) -> bool:
        return self.follower.holding()

    # ---- running ----

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self.run()), name="throughput-auditor", daemon=True).start()

    async def run(self) -> None:
        references: dict[str, Engine] = {m: BoundedEngine(HttpEngine(url)) for m, url in self.reference_urls.items()}
        tasks: set[asyncio.Task] = set()
        while True:
            try:
                for test in await asyncio.to_thread(self.follower.poll):
                    task = asyncio.create_task(self.run_test(test, references))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                await asyncio.to_thread(self.settle)
            except Exception:
                log.exception("auditor step failed")
            await asyncio.sleep(POLL_S)

    def targets(self, policy: AuditPolicy) -> list[Target]:
        cat, last = self.catalog(), self.store.last_delivered()
        out = []
        for slot in self._slots:
            model = cat.get(slot.model_id)
            if model is None or model.status == "retired":
                continue
            rate = planned_rate(slot, model.first_test_rps, last.get(slot.key), policy)
            if rate > 0:
                out.append(Target(slot, rate))
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
        return {"start_at_ms": start_ms, "warmup_s": policy.warmup_s, "test_s": policy.test_s,
                "timeout_s": policy.request_timeout_s, "words": list(WORDS), "seed": secrets.token_hex(16),
                "targets": planned}

    async def run_test(self, test: ScheduledTest, references: dict[str, Engine]) -> None:
        policy = self.policy
        if self.store.known(test.test_id) or self.follower.head > test.start_block:
            return  # already run, or selected before this auditor was following
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
        results = await asyncio.gather(*(self.evaluate(test, t, r, references, policy)
                                         for t, r in zip(targets, result.targets)))
        self.store.add_results(test.test_id, results)
        counts = {o: sum(r.outcome == o for r in results) for o in ("passed", "partial", "invalid", "void")}
        log.info("throughput test of epoch %d done: %s; %.0f of %.0f requests/s credited", test.epoch,
                 ", ".join(f"{n} {o}" for o, n in counts.items() if n), sum(r.credited_rps for r in results),
                 sum(t.tested_rps for t in targets))

    async def evaluate(self, test: ScheduledTest, target: Target, r: loadgen.TargetResult,
                       references: dict[str, Engine], policy: AuditPolicy) -> Result:
        detail = json.dumps({"sent": r.sent, "completed": r.completed, "skipped": r.skipped, "errors": r.errors,
                             "late_sends": r.late_sends, "samples": len(r.samples)}, sort_keys=True)
        latency = {"latency_p50_ms": r.latency_ms.p50 or 0.0, "latency_p95_ms": r.latency_ms.p95 or 0.0,
                   "detail": detail}
        if r.late_sends > LATE_SEND_SHARE * max(r.sent, 1):
            return Result(target, "void", "the auditor's load generator fell behind", **latency)
        slot, model = target.slot, self.catalog().get(target.slot.model_id)
        valid, evidence = 0, None
        for sample in r.samples:
            answers = signed_answers(sample.response, sample.request, request_id=sample.request_id,
                                     hotkey=slot.hotkey, endpoint_id=slot.endpoint_id, model_id=slot.model_id,
                                     auditor=self.hotkey, netuid=self.netuid, mechid=self.mechid)
            if answers is None:
                continue
            valid += 1
            if evidence is None and model is not None:
                fault = await answer_fault(sample.request, answers, references.get(slot.model_id), model.tolerance)
                if fault:
                    evidence = Evidence(hotkey=slot.hotkey, endpoint_id=slot.endpoint_id, model_id=slot.model_id,
                                        test_id=test.test_id, reason=fault, request_id=sample.request_id,
                                        request=sample.request, response=sample.response)
        delivered = r.completed / policy.test_s * (valid / len(r.samples) if r.samples else 1.0)
        if evidence is not None:
            return Result(target, "invalid", evidence.reason, round(delivered, 2), 0.0, evidence=evidence, **latency)
        credited = credit(target.tested_rps, delivered, policy)
        if credited >= target.tested_rps:
            return Result(target, "passed", "", round(delivered, 2), credited, **latency)
        reason = f"delivered {delivered:.1f} of {target.tested_rps:.1f} requests/s"
        if valid < len(r.samples):
            reason += f", {len(r.samples) - valid} of {len(r.samples)} sampled receipts invalid"
        return Result(target, "partial", reason, round(delivered, 2), credited, **latency)

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
        """The signed verdicts for scoring epoch ``epoch``; None for an epoch that has not started or whose tests are
        no longer kept, or while a test they cover is not settled."""
        if epoch in self._verdicts:
            return self._verdicts[epoch]
        policy = self.policy
        current, first = self.follower.head // policy.epoch_blocks, epoch - policy.credit_ttl_epochs
        if not current - KEEP_EPOCHS < epoch <= current or self.store.pending(first, epoch - 1):
            return None
        verdicts = Verdicts(netuid=self.netuid, auditor=self.hotkey, epoch=epoch,
                            credits=self.store.credits(first, epoch - 1, policy.qualification_tests),
                            invalid=self.store.evidence(first, epoch - 1))
        signed = SignedVerdicts.create(verdicts, self.keypair)
        self._verdicts = {e: v for e, v in self._verdicts.items() if e >= current - 2} | {epoch: signed}
        return signed

    def outcomes(self, hotkey: str, epoch: int) -> list[ThroughputOutcome]:
        """The hotkey's test results since ``credit_ttl_epochs + 1`` epochs before ``epoch``."""
        return self.store.outcomes(hotkey, epoch - self.policy.credit_ttl_epochs - 1)
