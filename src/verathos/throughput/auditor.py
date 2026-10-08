"""The throughput auditor: runs every scheduled test, checks sampled answers, and publishes signed verdicts.

Per test:

1. At the selection block the auditor fixes the targets: every listed open endpoint with a declared capacity, each at
   its declared rate, but at most ``rate_growth`` times what it delivered in its last test, or ``retest_growth``
   times while one of its last ``regrow_after`` tests fell short (at most the model's ``first_test_rps`` before its
   first of the model it serves now, and never less than ``min_test_share`` of that): a test probes beyond what an
   endpoint has shown it serves only while it keeps up, so users of an endpoint that cannot are not held up. A
   hotkey's endpoints together get at most ``rate_growth`` times what the hotkey delivered last, measured in first
   tests' worth, and at least one first test's worth. Neither an endpoint nor a hotkey can make the auditor send much
   more than it has shown it serves.
2. When the start block appears, the load generator drives all targets at once: a warmup drawn from
   [``warmup_s``, 2 x ``warmup_s``], then the measured window of ``test_s``. Endpoints sharing hardware, or a pool
   behind several endpoints, must deliver the sum of their rates at the same time.
3. Per endpoint: delivered = replies inside the window to requests sent no earlier than ``max_latency_s`` before it,
   per second, scaled by the share of sampled responses whose receipt is valid. Every signed sample must be well
   formed; ``answer_checks_per_test`` of them are recomputed (on the bf16 screen if there is one, whose flags the
   fp32 reference must confirm). Credit = the tested rate x the share of what was asked in the window (the user
   requests the gateways sent it, and the test's own requests up to what the users' answers leave of the tested
   rate) that was delivered (the test's replies and the users' answers), at most 1; below ``pass_fraction`` scaled by
   that share once more. The users' answers count only once the gateways' sample of them from the test is checked;
   without one the credit is at most what the test's own replies prove. A sampled answer the miner signed and the
   reference proves wrong makes the result invalid; the evidence is kept for the verdicts. The recomputed
   answers' differences are tallied per endpoint; one whose mean over its last ``drift_window_tests`` tests exceeds
   the model's ``mean_abs`` gets no credit.
4. Once the selection block is final with the hash it was selected by, the test is confirmed; otherwise void. A test
   is also void when the load generator fails or starts late; one endpoint's result is void when the generator fell
   behind its schedule or ran out of local resources for it, or when the auditor could not evaluate it.

An endpoint is credited from the latest confirmed test only: one that was not among its targets has no credit.

The subnet's gateways submit sampled exchanges of their user traffic (``protocol.gateway``). Per endpoint and epoch
the auditor checks up to ``gateway_checks_per_epoch`` of them the same way, between tests: a wrong answer is evidence
like one in a test, and their differences form a drift tally of their own, so correct test answers cannot dilute
drifted answers to users.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import math
import random
import secrets
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bittensor_wallet import Keypair

from verathos import catalog as catalog_module
from verathos.miner.engine import Engine
from verathos.protocol.api import MAX_RESPONSE_BYTES
from verathos.protocol.gateway import (
    GatewaySample,
    Route,
    Routes,
    ServedCount,
    SignedRoutes,
    latency_median_ms,
)
from verathos.protocol.manifest import MAX_ENDPOINTS
from verathos.protocol.receipts import parse_response, signed_answers
from verathos.protocol.status import STATUS_TESTS, ThroughputOutcome
from verathos.throughput import loadgen
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.feed import (
    GATEWAY_EVIDENCE,
    MALFORMED,
    Credit,
    Evidence,
    PaidServed,
    ReferenceUnavailable,
    SignedVerdicts,
    Verdicts,
    claimed_tokens,
    fault_of,
    readable,
    recompute,
    tokens_fault,
)
from verathos.throughput.follow import POLL_S, ScheduleFollower
from verathos.throughput.policy import BLOCK_TIME_S, AuditPolicy, ScheduledTest, Session, session_headers
from verathos.throughput.workload import WORDS, in_mix_form
from verathos.validator.canaries import canary_like
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.config import ReferenceEngine
from verathos.validator.references import transitioning, wrong_revision
from verathos.validator.scoring import bounded_shares, model_shares

log = logging.getLogger(__name__)
START_DELAY_MS = 500  # from observing the start block to the first request
MAX_LATE_START_MS = 1000
LATE_SEND_SHARE = 0.01  # more late sends than this voids an endpoint's result
REFERENCE_CONCURRENCY = 4
REFERENCE_TIMEOUT_S = 30.0
RECOMPUTE_BUDGET_S = 900.0  # after this, a test's remaining recomputes are skipped
BREAKER_FAILURES = 3  # an engine failing this often in a row is not asked again in the test
_UNCOUNTED: set[str] = set()  # models whose reference reported no input tokens (logged once each)
ROUTES_TTL_S = 30.0
KEPT_PER_ENDPOINT = 8  # gateway samples kept per endpoint and epoch, of which gateway_checks_per_epoch are checked
MAX_CHECK_ATTEMPTS = 3
SAMPLE_BUDGET_S = 40.0  # per round of sample checks, under the lead time of a test
SERVED_WAIT_S = 12.0  # after a test's window, for the gateways' last served counts
TEST_SAMPLES_WAIT_S = 2 * BLOCK_TIME_S  # after a test's sessions end, for the gateways' samples of users' answers in it
USER_RATE_S = 60.0  # a test leaves room for what the endpoint answered the gateways over this long ...
USER_RATE_LAG_S = 10.0  # ... ending this long before the test (the gateways report every few seconds)
MIN_TEST_SHARE = 0.1  # a test sends at least this share of the tested rate itself
USERS_BEFORE_S = (80.0, 20.0)  # an endpoint's usual user rate: [window start - 80 s, window start - 20 s)
SERVED_KEEP_S = 900
KEEP_EPOCHS = 30  # at least STATUS_TESTS: with a test every epoch, the tests a status lists are all kept
SCHEMA_VERSION = 11
_DEMAND_SCHEMA = """
CREATE TABLE demand (epoch INTEGER NOT NULL, model_id TEXT NOT NULL, input_tokens INTEGER NOT NULL,
    ema REAL NOT NULL, PRIMARY KEY (epoch, model_id));
"""
_USERS_SCHEMA = """
CREATE TABLE users (epoch INTEGER NOT NULL, hotkey TEXT NOT NULL, endpoint_id TEXT NOT NULL, model_id TEXT NOT NULL,
    offered INTEGER NOT NULL, refused INTEGER NOT NULL, answers INTEGER NOT NULL, latency TEXT NOT NULL,
    PRIMARY KEY (epoch, hotkey, endpoint_id, model_id));
"""
_RESULTS_INDEX = "CREATE INDEX IF NOT EXISTS results_by_endpoint ON results (hotkey, endpoint_id);"
_SAMPLES_BY_TEST = "CREATE INDEX IF NOT EXISTS samples_by_test ON samples (test_id, hotkey, endpoint_id);"
# meta: epoch_blocks (the rows' numbering), paid_demand (the gateways told paid from unpaid input tokens once), pins
# (per model what the catalog pins and since when, ``validator.references.transitioning``)
_META_SCHEMA = "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
_SHARES_SCHEMA = """
ALTER TABLE demand ADD COLUMN paid_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE demand ADD COLUMN paid_ema REAL NOT NULL DEFAULT 0;
CREATE TABLE shares (epoch INTEGER NOT NULL, model_id TEXT NOT NULL, share REAL NOT NULL,
    PRIMARY KEY (epoch, model_id));
"""
_PAID_SCHEMA = """
CREATE TABLE paid (epoch INTEGER NOT NULL, hotkey TEXT NOT NULL, endpoint_id TEXT NOT NULL, model_id TEXT NOT NULL,
    input_tokens INTEGER NOT NULL, PRIMARY KEY (epoch, hotkey, endpoint_id, model_id));
"""
# from version -> the statements that bring it one version up
_MIGRATIONS = {4: _DEMAND_SCHEMA, 5: _USERS_SCHEMA, 6: _RESULTS_INDEX,
               7: "ALTER TABLE samples ADD COLUMN test_id TEXT; " + _SAMPLES_BY_TEST, 8: _META_SCHEMA,
               9: _SHARES_SCHEMA, 10: _PAID_SCHEMA}
Runner = Callable[..., Awaitable[loadgen.LoadResult]]


@dataclass
class UserTally:
    """What one endpoint did for the gateways' users in one epoch: times it was considered for a request, refusals
    (see ``gateway.router``), answers with a known duration, those durations (``protocol.gateway`` buckets), and the
    input tokens of its answers users paid for."""

    offered: int = 0
    refused: int = 0
    answers: int = 0
    latency: dict[int, int] = field(default_factory=dict)
    paid_tokens: int = 0

    def add(self, other: UserTally) -> None:
        self.offered += other.offered
        self.refused += other.refused
        self.answers += other.answers
        self.paid_tokens += other.paid_tokens
        for bucket, n in other.latency.items():
            self.latency[bucket] = self.latency.get(bucket, 0) + n


class BoundedEngine:
    """An engine that takes at most ``concurrency`` requests at once (the reference GPU is shared by all checks)."""

    def __init__(self, engine: Engine, concurrency: int = REFERENCE_CONCURRENCY) -> None:
        self.engine, self._slots = engine, asyncio.Semaphore(concurrency)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        async with self._slots:
            return await self.engine.answer(request)

    async def info(self) -> Any:
        return await self.engine.info()


@dataclass(frozen=True)
class Checks:
    """Answer checks per model: the fp32 reference, and optionally a faster screen whose flags the reference must
    confirm before an answer counts as wrong."""

    references: dict[str, Engine]
    screens: dict[str, Engine]

    def for_test(self, budget_s: float = RECOMPUTE_BUDGET_S) -> Recomputes:
        return Recomputes(self, time.monotonic() + budget_s)

    async def usable(self, cat: catalog_module.Catalog, transitioning: Collection[str]) -> Checks:
        """The checks that may run now: none for a model in transition (``validator.references``), and none on an
        engine that names other weights than the catalog pins for its model."""

        async def keep(engines: dict[str, Engine], role: str) -> dict[str, Engine]:
            out = {}
            for model_id, engine in engines.items():
                model = cat.get(model_id)
                if model is None or model_id in transitioning:
                    continue
                if why := await wrong_revision(engine, model):
                    log.error("the %s %s engine is not used: %s", model_id, role, why)
                    continue
                out[model_id] = engine
            return out

        return Checks(await keep(self.references, "reference"), await keep(self.screens, "screen"))


class RankedSlots:
    """At most ``n`` holders at once; waiters are let in lowest rank first (ties: first come)."""

    def __init__(self, n: int) -> None:
        self._free, self._waiting, self._order = n, [], itertools.count()

    async def acquire(self, rank: float) -> None:
        if self._free > 0 and not self._waiting:
            self._free -= 1
            return
        waiter = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (rank, next(self._order), waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                self.release()
            raise

    def release(self) -> None:
        while self._waiting:
            _, _, waiter = heapq.heappop(self._waiting)
            if not waiter.done():
                waiter.set_result(None)
                return
        self._free += 1


class Recomputes:
    """The answer checks of one test: recomputes stop at the test's deadline, and an engine that fails
    ``BREAKER_FAILURES`` times in a row is not asked again in it. Under pressure the order decides which run: a
    recompute's rank (lower first) is its round times the number of endpoints plus the endpoint's place in a random
    order, so every endpoint's first check runs before any second, and which endpoints miss out is not predictable."""

    def __init__(self, checks: Checks, deadline: float) -> None:
        self.checks, self.deadline = checks, deadline
        self._failures: dict[int, int] = {}
        self._slots = RankedSlots(REFERENCE_CONCURRENCY)

    async def _compare(self, engine: Engine, model_id: str, request: dict[str, Any], got: dict[str, Any],
                       tolerance: Tolerance, rank: float = 0.0) -> RequestComparison | None:
        await self._slots.acquire(rank)
        try:  # the deadline and the breaker are judged when the recompute would start
            if time.monotonic() > self.deadline or self._failures.get(id(engine), 0) >= BREAKER_FAILURES:
                return None
            try:
                comparison = await recompute(request, got, engine, tolerance)
            except ReferenceUnavailable as exc:
                self._failures[id(engine)] = self._failures.get(id(engine), 0) + 1
                log.warning("a %s recompute failed, the answer is unchecked: %s", model_id, exc)
                return None
            self._failures[id(engine)] = 0
            if comparison.reference_input_tokens is None and model_id not in _UNCOUNTED:
                _UNCOUNTED.add(model_id)
                log.warning("the %s reference reports no usage.input_tokens: the input tokens miners sign for it, "
                            "which set its demand, are not checked", model_id)
            return comparison
        finally:
            self._slots.release()

    async def check(self, model_id: str, request: dict[str, Any], answers: Any, tolerance: Tolerance, *,
                    recompute_answers: bool, claimed: int | None = None,
                    rank: float = 0.0) -> tuple[str | None, RequestComparison | None]:
        """-> (why the answers are wrong or None, the comparison made or None). Without ``recompute_answers`` only
        their form is checked; with it, also the input tokens the receipt signs (``claimed``). ``rank``: see the
        class."""
        fault, comparison = await self._check(model_id, request, answers, tolerance, recompute_answers, rank)
        return fault or tokens_fault(claimed, comparison), comparison

    async def _check(self, model_id: str, request: dict[str, Any], answers: Any, tolerance: Tolerance,
                     recompute_answers: bool, rank: float) -> tuple[str | None, RequestComparison | None]:
        got = readable(request, answers)
        if got is None:
            return MALFORMED, None
        reference = self.checks.references.get(model_id)
        screen = self.checks.screens.get(model_id)
        first = screen or reference
        if not recompute_answers or first is None:
            return None, None
        comparison = await self._compare(first, model_id, request, got, tolerance, rank)
        if comparison is None:
            return None, None
        fault = fault_of(comparison, tolerance)
        if fault is None or screen is None:
            return fault, comparison
        if reference is None:
            log.warning("the %s screen flagged an answer but there is no fp32 reference to confirm it", model_id)
            return None, comparison
        confirmed = await self._compare(reference, model_id, request, got, tolerance, -1.0)  # a flag is confirmed first
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
CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, gateway TEXT NOT NULL, hotkey TEXT NOT NULL,
    endpoint_id TEXT NOT NULL, model_id TEXT NOT NULL, request_id TEXT NOT NULL, request TEXT NOT NULL,
    response TEXT NOT NULL, proof INTEGER NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
    evidence TEXT, drift_sum REAL NOT NULL DEFAULT 0, drift_count INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0, checked_epoch INTEGER, test_id TEXT);
CREATE INDEX samples_by_state ON samples (state, epoch);
CREATE INDEX samples_by_endpoint ON samples (hotkey, endpoint_id, epoch);
CREATE TABLE verdicts (epoch INTEGER PRIMARY KEY, document TEXT NOT NULL);
""" + _DEMAND_SCHEMA + _USERS_SCHEMA + _RESULTS_INDEX + _SAMPLES_BY_TEST + _META_SCHEMA + _SHARES_SCHEMA + _PAID_SCHEMA
# test states: running -> unconfirmed -> confirmed | void
_SCRUB = "request = '{}', response = ''"  # a user's request and answers are kept only until their check
# sample states: pending -> checked | skipped (its endpoint's checks for the epoch were done) | unchecked (no
# reference for the model, its recompute kept failing, or it was not checked within the next epoch). A sample counts
# (evidence, drift) in the epoch its check finished: verdicts for an epoch depend only on checks finished before it.
# A sample with a test_id is the gateways' pick from an endpoint's answers to users in that test: checked when the
# test is evaluated, or like a proof between tests if it came late or its recompute could not run then.


class Store:
    """Tests, results and sampled gateway exchanges, kept for ``KEEP_EPOCHS`` epochs, numbered by epochs of
    ``epoch_blocks`` (a new epoch length renumbers them: ``renumber``). A database of another schema version is
    started over. ``revision`` changes whenever a test or a result is recorded, confirmed, voided, renumbered or
    pruned: what ``outcomes`` lists can have changed."""

    def __init__(self, path: Path, *, epoch_blocks: int) -> None:
        self.revision = 0
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA secure_delete = ON")  # a scrubbed request leaves nothing in the file
        self._lock = threading.Lock()
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        while version in _MIGRATIONS:
            self._db.executescript(f"BEGIN; {_MIGRATIONS[version]} PRAGMA user_version = {version + 1}; COMMIT;")
            version += 1
        if version != SCHEMA_VERSION:
            self._db.executescript("BEGIN; DROP TABLE IF EXISTS results; DROP TABLE IF EXISTS tests; "
                                   "DROP TABLE IF EXISTS samples; DROP TABLE IF EXISTS verdicts; "
                                   "DROP TABLE IF EXISTS demand; DROP TABLE IF EXISTS users; "
                                   "DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS shares; "
                                   "DROP TABLE IF EXISTS paid;" + _SCHEMA
                                   + f"PRAGMA user_version = {SCHEMA_VERSION}; COMMIT;")
            log.info("auditor database %s started for schema %d", path, SCHEMA_VERSION)
        self._query("UPDATE tests SET state = 'void', reason = 'the auditor restarted during the test' "
                    "WHERE state = 'running'")
        numbered = self._query("SELECT value FROM meta WHERE key = 'epoch_blocks'")
        self.epoch_blocks = int(numbered[0][0]) if numbered else epoch_blocks
        self.renumber(epoch_blocks)

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    def renumber(self, epoch_blocks: int) -> None:
        """Number the rows by epochs of ``epoch_blocks``: a test by its selection block, anything else by its
        epoch's first block. Rows that fall into one new epoch merge: users' tallies add up, the latest demand and
        shares stand. Verdicts served for the old epochs are dropped: they cover epochs that no longer exist."""
        old, new = self.epoch_blocks, epoch_blocks
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('epoch_blocks', ?)", (str(new),))
            if old != new:
                self._db.execute("UPDATE tests SET epoch = selection_block / ?", (new,))
                self._db.execute("UPDATE samples SET epoch = epoch * ? / ?, checked_epoch = checked_epoch * ? / ?",
                                 (old, new, old, new))
                self._db.execute("DELETE FROM verdicts")
                for (value,) in self._db.execute("SELECT value FROM meta WHERE key = 'pins'").fetchall():
                    pins = {m: [pin, since * old // new] for m, (pin, since) in json.loads(value).items()}
                    self._db.execute("UPDATE meta SET value = ? WHERE key = 'pins'", (json.dumps(pins),))
                latest: dict[str, dict[tuple[int, str], tuple]] = {"demand": {}, "shares": {}}
                for table, rows in latest.items():
                    for epoch, model, *values in self._db.execute(f"SELECT * FROM {table} ORDER BY epoch").fetchall():
                        rows[epoch * old // new, model] = (epoch * old // new, model, *values)
                users: dict[tuple, UserTally] = {}
                for epoch, hk, ep, model, offered, refused, answers, latency in self._db.execute(
                        "SELECT * FROM users").fetchall():
                    users.setdefault((epoch * old // new, hk, ep, model), UserTally()).add(UserTally(
                        offered, refused, answers, {int(b): n for b, n in json.loads(latency).items()}))
                paid: dict[tuple, int] = {}
                for epoch, hk, ep, model, tokens in self._db.execute("SELECT * FROM paid").fetchall():
                    paid[epoch * old // new, hk, ep, model] = paid.get((epoch * old // new, hk, ep, model), 0) + tokens
                for table, rows in latest.items():
                    self._db.execute(f"DELETE FROM {table}")
                    for row in rows.values():
                        self._db.execute(f"INSERT INTO {table} VALUES ({', '.join('?' * len(row))})", row)
                self._db.execute("DELETE FROM users")
                self._db.executemany("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                     [(*key, t.offered, t.refused, t.answers, json.dumps(t.latency, sort_keys=True))
                                      for key, t in users.items()])
                self._db.execute("DELETE FROM paid")
                self._db.executemany("INSERT INTO paid VALUES (?, ?, ?, ?, ?)",
                                     [(*key, tokens) for key, tokens in paid.items()])
                log.info("auditor database renumbered from epochs of %d blocks to epochs of %d", old, new)
            self._db.execute("COMMIT")
            self.revision += 1
        self.epoch_blocks = new

    def known(self, test_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM tests WHERE test_id = ?", (test_id,)))

    def transitioning(self, cat: catalog_module.Catalog, epoch: int) -> set[str]:
        """The models in transition at ``epoch`` (``validator.references.transitioning``), with the pins kept here."""
        with self._lock:
            rows = self._db.execute("SELECT value FROM meta WHERE key = 'pins'").fetchall()
            pins = json.loads(rows[0][0]) if rows else {}
            out = transitioning(pins, cat, epoch)
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('pins', ?)", (json.dumps(pins),))
        return out

    def add_test(self, test: ScheduledTest) -> None:
        self._query("INSERT INTO tests (test_id, epoch, selection_block, selection_hash, start_block, state) "
                    "VALUES (?, ?, ?, ?, ?, 'running')",
                    (test.test_id, test.selection_block // self.epoch_blocks, test.selection_block,
                     test.selection_hash, test.start_block))
        self.revision += 1

    def set_state(self, test_id: str, state: str, reason: str = "") -> None:
        self._query("UPDATE tests SET state = ?, reason = ? WHERE test_id = ?", (state, reason, test_id))
        self.revision += 1

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
            self.revision += 1

    def unconfirmed(self) -> list[tuple[str, int, str]]:
        """-> [(test_id, selection_block, selection_hash)]"""
        return self._query("SELECT test_id, selection_block, selection_hash FROM tests WHERE state = 'unconfirmed'")

    def last_delivered(self, recent: int = 1) -> dict[str, tuple[str, float, bool]]:
        """Endpoint key -> (model, delivered rate in its latest counted result, whether one of its latest ``recent``
        counted results fell short of passing)."""
        rows = self._query(
            "SELECT r.hotkey, r.endpoint_id, r.model_id, r.delivered_rps, r.outcome FROM results r "
            "JOIN tests t USING (test_id) WHERE t.state != 'void' AND r.outcome != 'void' "
            "ORDER BY t.start_block")
        outcomes: dict[str, list[str]] = {}
        last: dict[str, tuple[str, float]] = {}
        for hk, ep, model, rps, outcome in rows:
            outcomes.setdefault(f"{hk}/{ep}", []).append(outcome)
            last[f"{hk}/{ep}"] = (model, rps)
        return {key: (model, rps, any(o != "passed" for o in outcomes[key][-recent:]))
                for key, (model, rps) in last.items()}

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
        tests, gateway = self._drift_tallies(last_epoch, policy.drift_window_tests)
        credits = []
        for hk, ep in sorted(targeted):
            if (hk, ep) not in counted:
                continue
            model_id, credited, latency, epoch = counted[hk, ep]
            tolerance = tolerance_of(model_id)
            drift = tolerance is not None and (_beyond(*tests.get((hk, ep, model_id), (0.0, 0)), tolerance)
                                               or _beyond(*gateway.get((hk, ep, model_id), (0.0, 0)), tolerance))
            credits.append(Credit(hotkey=hk, endpoint_id=ep, model_id=model_id, credited_rps=0.0 if drift else credited,
                                  latency_ms=latency, test_epoch=epoch, drift=drift,
                                  qualified=passed.get((hk, ep), 0) >= policy.qualification_tests))
        return credits

    def _drift_tallies(self, last_epoch: int, window: int) -> tuple[dict, dict]:
        """Per (hotkey, endpoint, model): the recomputed answers' summed differences and count over its last
        ``window`` confirmed tests, and over its checked gateway answers of the last ``window`` epochs. Two grouped
        queries, whatever the number of endpoints."""
        tests = {(hk, ep, m): (total, count) for hk, ep, m, total, count in self._query(
            "SELECT hotkey, endpoint_id, model_id, SUM(drift_sum), SUM(drift_count) FROM ("
            "SELECT r.hotkey, r.endpoint_id, r.model_id, r.drift_sum, r.drift_count, ROW_NUMBER() OVER ("
            "PARTITION BY r.hotkey, r.endpoint_id, r.model_id ORDER BY t.epoch DESC, t.start_block DESC) AS n "
            "FROM results r JOIN tests t USING (test_id) WHERE t.state = 'confirmed' AND t.epoch <= ?) "
            "WHERE n <= ? GROUP BY hotkey, endpoint_id, model_id", (last_epoch, window))}
        gateway = {(hk, ep, m): (total, count) for hk, ep, m, total, count in self._query(
            "SELECT hotkey, endpoint_id, model_id, SUM(drift_sum), SUM(drift_count) FROM samples "
            "WHERE state = 'checked' AND checked_epoch BETWEEN ? AND ? GROUP BY hotkey, endpoint_id, model_id",
            (last_epoch - window + 1, last_epoch))}
        return tests, gateway

    def evidence(self, first_epoch: int, last_epoch: int) -> list[Evidence]:
        """The first wrong answer proven against each hotkey in the epoch range: in a confirmed test, or in a checked
        gateway sample."""
        rows = self._query(
            "SELECT t.epoch, r.hotkey, r.evidence FROM results r JOIN tests t USING (test_id) "
            "WHERE t.state = 'confirmed' AND r.evidence IS NOT NULL AND t.epoch BETWEEN ? AND ? "
            "UNION ALL SELECT checked_epoch, hotkey, evidence FROM samples WHERE evidence IS NOT NULL "
            "AND checked_epoch BETWEEN ? AND ? ORDER BY 1, 2", (first_epoch, last_epoch, first_epoch, last_epoch))
        first: dict[str, str] = {}
        for _, hotkey, evidence in rows:
            first.setdefault(hotkey, evidence)
        return [Evidence.model_validate_json(e) for e in first.values()]

    # ---- gateway samples ----

    def add_sample(self, *, sample_id: str, epoch: int, gateway: str, hotkey: str, endpoint_id: str, model_id: str,
                   request_id: str, request: dict[str, Any], response: str, proof: bool,
                   test_id: str | None = None) -> bool:
        """Keep a sample, unless its endpoint already has ``KEPT_PER_ENDPOINT`` in the epoch (a proof is always kept;
        a test's sample (``test_id``) is kept once per endpoint and test, besides those) or it is known."""
        with self._lock:
            if test_id is not None:
                if self._db.execute("SELECT 1 FROM samples WHERE test_id = ? AND hotkey = ? AND endpoint_id = ?",
                                    (test_id, hotkey, endpoint_id)).fetchall():
                    return False
            else:
                (count,), = self._db.execute(
                    "SELECT COUNT(*) FROM samples WHERE hotkey = ? AND endpoint_id = ? AND epoch = ? AND proof = 0 "
                    "AND test_id IS NULL", (hotkey, endpoint_id, epoch)).fetchall()
                if count >= KEPT_PER_ENDPOINT and not proof:
                    return False
            cursor = self._db.execute(
                "INSERT OR IGNORE INTO samples (sample_id, epoch, gateway, hotkey, endpoint_id, model_id, request_id, "
                "request, response, proof, state, test_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (sample_id, epoch, gateway, hotkey, endpoint_id, model_id, request_id, json.dumps(request), response,
                 int(proof), test_id))
            return cursor.rowcount == 1

    def test_sample(self, test_id: str, hotkey: str, endpoint_id: str) -> tuple[str, str, str, str] | None:
        """The pending sample of the endpoint's answers to users in the test: (sample_id, request_id, request,
        response), or None."""
        rows = self._query("SELECT sample_id, request_id, request, response FROM samples WHERE test_id = ? "
                           "AND hotkey = ? AND endpoint_id = ? AND state = 'pending'", (test_id, hotkey, endpoint_id))
        return rows[0] if rows else None

    def samples_to_check(self, epoch: int, per_endpoint: int, rng: random.Random) -> list[tuple]:
        """The samples to check now: every pending proof and test's sample, and per endpoint and epoch (this one and
        the last) random pending samples until ``per_endpoint`` are checked; the rest of a done endpoint's are
        skipped, and pending ones of older epochs given up. In the order to check them: proofs and tests' samples,
        then every endpoint's first sample (the endpoints in random order), then every second one, so a round cut
        short checks as many endpoints as it can.
        -> [(sample_id, gateway, hotkey, endpoint_id, model_id, request_id, request, response)]"""
        self._query(f"UPDATE samples SET state = 'unchecked', reason = 'not checked within the next epoch', "
                    f"{_SCRUB} WHERE state = 'pending' AND epoch < ?", (epoch - 1,))
        rows = self._query("SELECT sample_id, epoch, gateway, hotkey, endpoint_id, model_id, request_id, request, "
                           "response, proof OR test_id IS NOT NULL FROM samples WHERE state = 'pending'")
        checked = {(hk, ep, e): n for hk, ep, e, n in self._query(
            "SELECT hotkey, endpoint_id, epoch, COUNT(*) FROM samples WHERE state = 'checked' AND proof = 0 "
            "AND test_id IS NULL AND epoch >= ? GROUP BY hotkey, endpoint_id, epoch", (epoch - 1,))}
        groups: dict[tuple[str, str, int], list[tuple]] = {}
        chosen = [r for r in rows if r[9]]
        for r in rows:
            if not r[9]:
                groups.setdefault((r[3], r[4], r[1]), []).append(r)
        picks: dict[tuple[str, str, int], list[tuple]] = {}
        for key, pending in groups.items():
            wanted = max(per_endpoint - checked.get(key, 0), 0)
            picked = picks[key] = rng.sample(pending, min(wanted, len(pending)))
            if wanted <= len(picked):  # the endpoint's checks for the epoch are done (once these finish)
                rest = [r[0] for r in pending if r not in picked]
                if rest:
                    marks = ",".join("?" * len(rest))
                    self._query(f"UPDATE samples SET state = 'skipped', {_SCRUB} WHERE sample_id IN ({marks})",
                                rest)
        keys = list(picks)
        rng.shuffle(keys)
        for k in range(max((len(p) for p in picks.values()), default=0)):
            chosen += [picks[key][k] for key in keys if k < len(picks[key])]
        return [(r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8]) for r in chosen]

    def failed_check(self, sample_id: str) -> None:
        """A recompute of the sample could not run; after ``MAX_CHECK_ATTEMPTS`` it is given up."""
        self._query("UPDATE samples SET attempts = attempts + 1, state = CASE WHEN attempts + 1 >= ? THEN 'unchecked' "
                    "ELSE state END, reason = CASE WHEN attempts + 1 >= ? THEN 'its recompute kept failing' "
                    "ELSE reason END WHERE sample_id = ?", (MAX_CHECK_ATTEMPTS, MAX_CHECK_ATTEMPTS, sample_id))
        self._query(f"UPDATE samples SET {_SCRUB} WHERE sample_id = ? AND state = 'unchecked'", (sample_id,))

    def finish_sample(self, sample_id: str, state: str, reason: str = "", evidence: Evidence | None = None,
                      drift_sum: float = 0.0, drift_count: int = 0, *, checked_epoch: int | None = None) -> None:
        self._query(f"UPDATE samples SET state = ?, reason = ?, evidence = ?, drift_sum = ?, drift_count = ?, "
                    f"checked_epoch = ?, {_SCRUB} WHERE sample_id = ?",
                    (state, reason, evidence.model_dump_json() if evidence else None, drift_sum, drift_count,
                     checked_epoch, sample_id))

    def offenders(self, since_epoch: int) -> set[str]:
        """Hotkeys with a wrong answer proven since ``since_epoch``, confirmed or not yet."""
        rows = self._query(
            "SELECT r.hotkey FROM results r JOIN tests t USING (test_id) WHERE r.evidence IS NOT NULL "
            "AND t.state != 'void' AND t.epoch >= ? UNION SELECT hotkey FROM samples WHERE evidence IS NOT NULL "
            "AND checked_epoch >= ?", (since_epoch, since_epoch))
        return {r[0] for r in rows}

    # ---- user demand ----

    def demand_closed(self, epoch: int) -> bool:
        return bool(self._query("SELECT 1 FROM demand WHERE epoch = ? LIMIT 1", (epoch,)))

    def close_demand(self, epoch: int, tokens: dict[str, int], epoch_s: float, alpha: float,
                     paid: dict[str, int] | None = None) -> dict[str, float]:
        """Record the epoch's input tokens per model, and of them those users paid for (``paid``; None: the gateways
        did not tell); -> each model's demand EMA (input tokens/s) after it. Once the gateways told paid from unpaid,
        an epoch without word counts as unpaid."""
        before, before_paid = self.demand_ema(epoch - 1), self.paid_demand(epoch - 1) or {}
        models = set(before) | set(tokens) | set(before_paid) | set(paid or {})
        ema = {m: alpha * tokens.get(m, 0) / epoch_s + (1 - alpha) * before.get(m, 0.0) for m in models}
        paid_ema = {m: alpha * (paid or {}).get(m, 0) / epoch_s + (1 - alpha) * before_paid.get(m, 0.0)
                    for m in models}
        rows = [(epoch, m, tokens.get(m, 0), round(ema[m], 6), (paid or {}).get(m, 0), round(paid_ema[m], 6))
                for m in models if ema[m] > 1e-6 or paid_ema[m] > 1e-6 or m in tokens]
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany("INSERT OR REPLACE INTO demand VALUES (?, ?, ?, ?, ?, ?)", rows)
            if paid is not None:
                self._db.execute("INSERT OR REPLACE INTO meta VALUES ('paid_demand', '1')")
            self._db.execute("COMMIT")
        return {m: v for _, m, _, v, _, _ in rows}

    def demand_ema(self, epoch: int) -> dict[str, float]:
        """Each model's demand EMA as of the latest closed epoch up to ``epoch``."""
        rows = self._query("SELECT model_id, ema FROM demand WHERE epoch = (SELECT MAX(epoch) FROM demand WHERE "
                           "epoch <= ?)", (epoch,))
        return dict(rows)

    def paid_demand(self, epoch: int) -> dict[str, float] | None:
        """Each model's EMA of the input tokens/s users paid for, as of the latest closed epoch up to ``epoch``; None
        until the gateways tell paid from unpaid."""
        if not self._query("SELECT 1 FROM meta WHERE key = 'paid_demand'"):
            return None
        rows = self._query("SELECT model_id, paid_ema FROM demand WHERE epoch = (SELECT MAX(epoch) FROM demand "
                           "WHERE epoch <= ?)", (epoch,))
        return {m: v for m, v in rows if v > 0}

    def close_shares(self, epoch: int, target: dict[str, float], step: float,
                     start: dict[str, float]) -> dict[str, float]:
        """The models' emission shares for ``epoch``: ``target`` reached from the latest shares before it (``start``
        if there are none) by at most ``step`` per epoch since (``validator.scoring.bounded_shares``), recorded."""
        rows = self._query("SELECT epoch, model_id, share FROM shares WHERE epoch = (SELECT MAX(epoch) FROM shares "
                           "WHERE epoch < ?)", (epoch,))
        previous = {m: share for _, m, share in rows} or start
        shares = bounded_shares(previous, target, step * (epoch - rows[0][0] if rows else 1))
        with self._lock:
            self._db.execute("BEGIN")
            self._db.execute("DELETE FROM shares WHERE epoch = ?", (epoch,))
            self._db.executemany("INSERT INTO shares VALUES (?, ?, ?)", [(epoch, m, v) for m, v in shares.items()])
            self._db.execute("COMMIT")
        return shares

    # ---- what users got ----

    def close_users(self, epoch: int, tallies: dict[tuple[str, str, str], UserTally]) -> None:
        rows = [(epoch, hk, ep, model, t.offered, t.refused, t.answers, json.dumps(t.latency, sort_keys=True))
                for (hk, ep, model), t in tallies.items()]
        paid = [(epoch, *key, t.paid_tokens) for key, t in tallies.items() if t.paid_tokens > 0]
        with self._lock:
            self._db.execute("BEGIN")
            self._db.executemany("INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            self._db.executemany("INSERT OR REPLACE INTO paid VALUES (?, ?, ?, ?, ?)", paid)
            self._db.execute("COMMIT")

    def users_closed(self, epoch: int) -> bool:
        return bool(self._query("SELECT 1 FROM users WHERE epoch = ? LIMIT 1", (epoch,)))

    def paid(self, first_epoch: int, last_epoch: int) -> dict[tuple[str, str, str], int]:
        """The input tokens users paid for that each endpoint answered, per (hotkey, endpoint, model), in epochs
        ``first_epoch .. last_epoch``."""
        rows = self._query("SELECT hotkey, endpoint_id, model_id, SUM(input_tokens) FROM paid WHERE epoch BETWEEN ? "
                           "AND ? GROUP BY hotkey, endpoint_id, model_id", (first_epoch, last_epoch))
        return {(hk, ep, model): int(tokens) for hk, ep, model, tokens in rows}

    def users(self, epoch: int) -> dict[tuple[str, str, str], UserTally]:
        """What each endpoint did for the gateways' users of each model in ``epoch``."""
        rows = self._query("SELECT hotkey, endpoint_id, model_id, offered, refused, answers, latency FROM users "
                           "WHERE epoch = ?", (epoch,))
        return {(hk, ep, model): UserTally(offered, refused, answers,
                                           {int(b): n for b, n in json.loads(latency).items()})
                for hk, ep, model, offered, refused, answers, latency in rows}

    # ---- served verdicts ----

    def served_verdicts(self, epoch: int) -> str | None:
        rows = self._query("SELECT document FROM verdicts WHERE epoch = ?", (epoch,))
        return rows[0][0] if rows else None

    def serve_verdicts(self, epoch: int, document: str) -> None:
        self._query("INSERT OR IGNORE INTO verdicts (epoch, document) VALUES (?, ?)", (epoch, document))

    def outcomes(self, hotkey: str, tests: int = STATUS_TESTS) -> list[ThroughputOutcome]:
        """The hotkey's results in the last ``tests`` tests (whatever their state), oldest first: at most
        ``MAX_ENDPOINTS`` per test, as a manifest lists."""
        rows = self._query(
            "SELECT r.endpoint_id, t.epoch, t.start_block, t.state, r.outcome, "
            "CASE WHEN t.state = 'void' THEN t.reason ELSE r.reason END, r.declared_rps, r.tested_rps, "
            "r.delivered_rps, r.credited_rps, r.latency_p50_ms, r.latency_p95_ms "
            "FROM results r JOIN tests t USING (test_id) WHERE r.hotkey = ? AND t.test_id IN "
            "(SELECT test_id FROM tests ORDER BY start_block DESC LIMIT ?) "
            "ORDER BY t.start_block DESC, r.endpoint_id DESC LIMIT ?", (hotkey, tests, tests * MAX_ENDPOINTS))
        keys = ("endpoint_id", "epoch", "start_block", "state", "outcome", "reason", "declared_rps", "tested_rps",
                "delivered_rps", "credited_rps", "latency_p50_ms", "latency_p95_ms")
        return [ThroughputOutcome(**dict(zip(keys, row))) for row in reversed(rows)]

    def prune(self, before_epoch: int) -> None:
        """Forget epochs before ``before_epoch``."""
        with self._lock:
            self._db.execute("DELETE FROM results WHERE test_id IN (SELECT test_id FROM tests WHERE epoch < ?)",
                             (before_epoch,))
            self._db.execute("DELETE FROM tests WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM samples WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM verdicts WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM demand WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM users WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM shares WHERE epoch < ?", (before_epoch,))
            self._db.execute("DELETE FROM paid WHERE epoch < ?", (before_epoch,))
            self.revision += 1


def _beyond(total: float, count: int, tolerance: Tolerance) -> bool:
    """Whether ``count`` compared probabilities summing to ``total`` exceed the tolerance's mean, once they are
    enough to judge."""
    return count >= tolerance.min_probabilities and total / count > tolerance.mean_abs


def _renumbered(by_epoch: dict[int, dict[str, int]], old: int, new: int) -> dict[int, dict[str, int]]:
    """Per-epoch counts of epochs of ``old`` blocks, added up per epoch of ``new`` blocks (by their first block)."""
    out: dict[int, dict[str, int]] = {}
    for epoch, counts in by_epoch.items():
        merged = out.setdefault(epoch * old // new, {})
        for key, n in counts.items():
            merged[key] = merged.get(key, 0) + n
    return out


def _gateway_evidence(hotkey: str, endpoint_id: str, model_id: str, request_id: str, fault: str,
                      response: dict[str, Any]) -> Evidence:
    """Evidence of a wrong answer to a user: the miner's receipt only, as user requests are not published."""
    return Evidence(hotkey=hotkey, endpoint_id=endpoint_id, model_id=model_id,
                    test_id=f"{GATEWAY_EVIDENCE}-{request_id}", reason=fault, request_id=request_id, request={},
                    response=json.dumps({"receipt": response["receipt"]}))


def _tolerance_in(cat: catalog_module.Catalog) -> Callable[[str], Tolerance | None]:
    """A model's tolerance in ``cat``; None for a model it does not list."""

    def tolerance_of(model_id: str) -> Tolerance | None:
        model = cat.get(model_id)
        return model.tolerance if model is not None else None

    return tolerance_of


def planned_rate(slot: Slot, first_test_rps: float, last_delivered: float | None, policy: AuditPolicy,
                 fell_short: bool = False) -> float:
    """The rate an endpoint is tested at, before its hotkey's budget; ``fell_short``: one of its last
    ``regrow_after`` counted tests did."""
    if last_delivered is None:
        ceiling = first_test_rps
    else:
        growth = policy.retest_growth if fell_short else policy.rate_growth
        ceiling = max(policy.min_test_share * first_test_rps, growth * last_delivered)
    return round(min(slot.declared_rps, ceiling), 2)


def credit(tested_rps: float, delivered_rps: float, policy: AuditPolicy, asked_rps: float | None = None) -> float:
    """The tested rate times the share of what was asked that was delivered (at most 1); below ``pass_fraction``
    scaled by that share once more. ``asked_rps``: what the endpoint was asked in the window, the test's own
    requests and the users' (default: the tested rate), so users sending less than the test left room for cost
    nothing, and users refused count as asked."""
    asked = tested_rps if asked_rps is None else asked_rps
    share = min(delivered_rps / asked, 1.0) if asked > 0 else 0.0
    if share >= policy.pass_fraction:
        return round(tested_rps * share, 2)
    return round(tested_rps * share * share, 2)


class Auditor:
    def __init__(self, *, netuid: int, keypair: Keypair, chain: SubtensorChainView,
                 policy: AuditPolicy, catalog: Callable[[], catalog_module.Catalog],
                 references: dict[str, ReferenceEngine], db_path: Path, runner: Runner = loadgen.run,
                 screens: dict[str, ReferenceEngine] | None = None) -> None:
        """``references`` recompute answers in fp32; a model with an entry in ``screens`` is screened there first
        (bf16), and the fp32 reference only confirms what the screen flags."""
        self.netuid, self.keypair, self.chain = netuid, keypair, chain
        self.hotkey = keypair.ss58_address
        self.policy, self.catalog, self.runner = policy, catalog, runner
        self.reference_specs, self.screen_specs = references, screens or {}
        self.store = Store(db_path, epoch_blocks=policy.epoch_blocks)
        self.follower = ScheduleFollower(chain, policy)
        self._slots: list[Slot] = []
        self._discovered = False  # whether _slots holds a discovery yet
        self._verdicts: dict[int, SignedVerdicts] = {}
        self._pruned_epoch = -1
        self._tests_running = 0
        self._routes: tuple[float, SignedRoutes] | None = None
        self._verdicts_lock = threading.Lock()
        # (hotkey, endpoint) -> second -> [user requests answered, user requests sent]
        self._served: dict[tuple[str, str], dict[int, list[int]]] = {}
        self._demand: dict[int, dict[str, int]] = {}  # epoch -> model -> users' input tokens (not closed yet)
        self._paid: dict[int, dict[str, int]] = {}  # the same, paid for: epochs the gateways told paid from unpaid
        self._users: dict[int, dict[tuple[str, str, str], UserTally]] = {}  # epoch -> (hotkey, endpoint, model) ->
        # tally, not closed yet
        self._reports: dict[str, float] = {}  # served reports taken lately -> when
        self._demand_lock = threading.Lock()
        self._rng = random.SystemRandom()

    # ---- inputs ----

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt a new policy for the tests and verdicts to come; verdicts already served stay as they are. A new
        epoch length renumbers what the auditor holds (``Store.renumber``), so no history is lost."""
        old, new = self.policy.epoch_blocks, policy.epoch_blocks
        with self._verdicts_lock, self._demand_lock:  # what numbers rows by epoch holds one of them
            if old != new:
                self.store.renumber(new)
                self._verdicts, self._pruned_epoch = {}, -1
                self._demand, self._paid = _renumbered(self._demand, old, new), _renumbered(self._paid, old, new)
                users: dict[int, dict[tuple[str, str, str], UserTally]] = {}
                for epoch, tallies in self._users.items():
                    merged_users = users.setdefault(epoch * old // new, {})
                    for key, tally in tallies.items():
                        merged_users.setdefault(key, UserTally()).add(tally)
                self._users = users
            self.policy = policy
        self.follower.set_policy(policy)

    def update_slots(self, slots: list[Slot]) -> None:
        self._slots, self._discovered = slots, True

    def holding(self) -> bool:
        return self.follower.holding()

    # ---- running ----

    def start(self) -> None:
        threading.Thread(target=lambda: asyncio.run(self.run()), name="throughput-auditor", daemon=True).start()

    async def run(self) -> None:
        checks = Checks(
            {m: BoundedEngine(r.engine(REFERENCE_TIMEOUT_S)) for m, r in self.reference_specs.items()},
            {m: BoundedEngine(r.engine(REFERENCE_TIMEOUT_S)) for m, r in self.screen_specs.items()})
        tasks: set[asyncio.Task] = {asyncio.create_task(self.check_samples_forever(checks))}
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

    async def check_samples_forever(self, checks: Checks) -> None:
        """Gateway samples are checked between tests, never holding up the schedule."""
        while True:
            try:
                if not self._tests_running:
                    await self.check_samples(checks)
            except Exception:
                log.exception("checking gateway samples failed")
            await asyncio.sleep(POLL_S)

    async def _usable(self, checks: Checks) -> Checks:
        """``checks`` less what may not run now (``Checks.usable``)."""
        cat, epoch = self.catalog(), self.follower.head // self.policy.epoch_blocks
        return await checks.usable(cat, await asyncio.to_thread(self.store.transitioning, cat, epoch))

    def targets(self, policy: AuditPolicy) -> list[Target]:
        cat, last = self.catalog(), self.store.last_delivered(policy.regrow_after)
        by_hotkey: dict[str, list[tuple[Slot, catalog_module.CatalogModel]]] = {}
        for slot in self._slots:
            model = cat.get(slot.model_id)
            if model is not None and model.status != "retired":
                by_hotkey.setdefault(slot.hotkey, []).append((slot, model))
        out = []
        for hotkey, slots in by_hotkey.items():
            rates = []
            for slot, model in slots:
                model_last, delivered_last, fell_short = last.get(slot.key, (None, None, False))
                if model_last != slot.model_id:  # what it delivered of another model says nothing of this one
                    delivered_last, fell_short = None, False
                rates.append(planned_rate(slot, model.first_test_rps, delivered_last, policy, fell_short))
            # The hotkey's budget in first tests' worth, from what all its endpoints (renamed ones too) delivered.
            delivered = sum(rps / cat.get(m).first_test_rps for key, (m, rps, _) in last.items()
                            if key.startswith(hotkey + "/") and cat.get(m) is not None)
            budget = max(1.0, policy.rate_growth * delivered)
            units = sum(rate / model.first_test_rps for rate, (_, model) in zip(rates, slots))
            scale = min(1.0, budget / units) if units else 1.0
            out += [Target(slot, round(rate * scale, 2)) for rate, (slot, _) in zip(rates, slots)
                    if round(rate * scale, 2) > 0]
        return out

    def test_rate(self, target: Target, now_s: float, policy: AuditPolicy) -> float:
        """What the test itself sends the endpoint: the tested rate less its recent rate of user requests, which it
        serves first and which count as delivered; at least ``MIN_TEST_SHARE`` of the tested rate, and enough for
        the test's samples."""
        end = now_s - USER_RATE_LAG_S
        key = (target.slot.hotkey, target.slot.endpoint_id, end - USER_RATE_S, end)
        users = max(self.served_in(*key, attempts=True), self.served_in(*key)) / USER_RATE_S
        floor = max(MIN_TEST_SHARE * target.tested_rps, policy.samples_per_test / policy.test_s)
        return round(min(target.tested_rps, max(floor, target.tested_rps - users)), 3)

    def plan(self, test: ScheduledTest, targets: list[Target], start_ms: int, policy: AuditPolicy) -> dict[str, Any]:
        cat, not_after_ms, now_s = self.catalog(), start_ms + int(policy.session_s * 1000), time.time()
        planned = []
        for t in targets:
            workload = cat.get(t.slot.model_id).workload
            session = Session(netuid=self.netuid, auditor=self.hotkey, miner_hotkey=t.slot.hotkey,
                              endpoint_id=t.slot.endpoint_id, test_id=test.test_id, not_after_ms=not_after_ms)
            planned.append({
                "id": t.slot.key, "url": t.slot.url + "/v1/systemone", "rate_rps": self.test_rate(t, now_s, policy),
                "max_outstanding": math.ceil(t.tested_rps * policy.request_timeout_s) + 16,
                "headers": session_headers(session, self.keypair), "model": t.slot.model_id,
                "state_words": list(workload.state_words), "questions": workload.questions,
                "samples": policy.samples_per_test,
            })
        # The miner cannot know when the window opens, so it cannot time held-back answers to it.
        warmup = round(policy.warmup_s * (1 + secrets.SystemRandom().random()), 3)
        return {"start_at_ms": start_ms, "warmup_s": warmup, "test_s": policy.test_s, "grace_s": policy.max_latency_s,
                "timeout_s": policy.request_timeout_s, "max_response_bytes": MAX_RESPONSE_BYTES,
                "words": list(WORDS), "seed": secrets.token_hex(16), "targets": planned}

    async def run_test(self, test: ScheduledTest, checks: Checks) -> None:
        """Run one test; any failure of the auditor itself voids it."""
        if self.store.known(test.test_id) or self.follower.head > test.start_block:
            return  # already run, or selected before this auditor was following
        self._tests_running += 1
        try:
            await self._run_test(test, checks)
        except Exception as exc:
            log.exception("throughput test of epoch %d failed", test.epoch)
            if self.store.known(test.test_id):
                self.store.set_state(test.test_id, "void", f"the auditor failed: {type(exc).__name__}")
        finally:
            self._tests_running -= 1

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
        plan = self.plan(test, targets, start_ms, policy)
        log.info("throughput test of epoch %d sends %.1f requests/s itself, users fill the other %.1f", test.epoch,
                 sum(t["rate_rps"] for t in plan["targets"]),
                 sum(t.tested_rps for t in targets) - sum(t["rate_rps"] for t in plan["targets"]))
        try:
            result = await self.runner(plan, timeout_s=policy.session_s + 30.0)
        except loadgen.LoadgenError as exc:
            log.error("throughput test of epoch %d void: load generator failed: %s", test.epoch, exc)
            self.store.set_state(test.test_id, "void", "the auditor's load generator failed")
            return
        if result.late_start_ms > MAX_LATE_START_MS:
            self.store.set_state(test.test_id, "void", "the auditor's load generator started late")
            return
        window_s = (result.started_at_ms / 1000 + plan["warmup_s"], result.started_at_ms / 1000 + plan["warmup_s"]
                    + policy.test_s)
        await asyncio.sleep(max(0.0, window_s[1] + SERVED_WAIT_S - time.time()))  # the gateways' last counts
        if any(self.served_in(t.slot.hotkey, t.slot.endpoint_id, *window_s) for t in targets):
            # users were answered in the window: wait for the gateways' samples of those answers (they follow the
            # start block on their own chain view)
            await asyncio.sleep(max(0.0, start_ms / 1000 + policy.session_s + TEST_SAMPLES_WAIT_S - time.time()))
        recomputes, scales = (await self._usable(checks)).for_test(), self.user_scales(window_s)
        order = list(range(len(targets)))
        secrets.SystemRandom().shuffle(order)  # which endpoints' checks wait longest under pressure: unpredictable
        results = await asyncio.gather(*(self.evaluate(test, t, r, recomputes, policy, window_s,
                                                       scales.get(t.slot.model_id, 1.0), (place, len(targets)))
                                         for t, r, place in zip(targets, result.targets, order)))
        self.store.add_results(test.test_id, results)
        counts = {o: sum(r.outcome == o for r in results) for o in ("passed", "partial", "invalid", "void")}
        log.info("throughput test of epoch %d done: %s; %.0f of %.0f requests/s credited", test.epoch,
                 ", ".join(f"{n} {o}" for o, n in counts.items() if n), sum(r.credited_rps for r in results),
                 sum(t.tested_rps for t in targets))

    def user_scales(self, window_s: tuple[float, float]) -> dict[str, float]:
        """Per model: the gateways' user answers in the window relative to before it; a dip of all users is no
        endpoint's doing."""
        before = (window_s[0] - USERS_BEFORE_S[0], window_s[0] - USERS_BEFORE_S[1])
        totals: dict[str, list[float]] = {}
        for s in self._slots:
            t = totals.setdefault(s.model_id, [0.0, 0.0])
            t[0] += self.served_in(s.hotkey, s.endpoint_id, *before) / (before[1] - before[0])
            t[1] += self.served_in(s.hotkey, s.endpoint_id, *window_s) / max(window_s[1] - window_s[0], 1e-9)
        return {m: (w / b if b > 0 else 1.0) for m, (b, w) in totals.items()}

    async def evaluate(self, test: ScheduledTest, target: Target, r: loadgen.TargetResult, recomputes: Recomputes,
                       policy: AuditPolicy, window_s: tuple[float, float] = (0.0, 0.0),
                       user_scale: float = 1.0, place: tuple[int, int] = (0, 1)) -> Result:
        """One endpoint's result; whatever its responses contain, a failure here voids only this endpoint.
        ``window_s``: the measured window (unix seconds), in which what the endpoint answered the gateways counts as
        delivered too."""
        try:
            return await self._evaluate(test, target, r, recomputes, policy, window_s, user_scale, place)
        except Exception:
            log.exception("evaluating %s in the test of epoch %d failed", target.slot.key, test.epoch)
            return Result(target, "void", "the auditor could not evaluate this endpoint")

    async def _evaluate(self, test: ScheduledTest, target: Target, r: loadgen.TargetResult, recomputes: Recomputes,
                        policy: AuditPolicy, window_s: tuple[float, float], user_scale: float = 1.0,
                        place: tuple[int, int] = (0, 1)) -> Result:
        endpoint = (target.slot.hotkey, target.slot.endpoint_id)
        gateway = self.served_in(*endpoint, *window_s)
        users = self.served_in(*endpoint, *window_s, attempts=True)
        # Users sending the endpoint less than usual in the window are excused within their random spread (scaled by
        # the dip of all the model's users); fewer than that, it drove them off (slow, refusing): asked all the same.
        before = (window_s[0] - USERS_BEFORE_S[0], window_s[0] - USERS_BEFORE_S[1])
        usual = max(self.served_in(*endpoint, *before, attempts=True), self.served_in(*endpoint, *before))
        expected = usual / (before[1] - before[0]) * user_scale * (window_s[1] - window_s[0])
        users_asked = max(users, gateway, expected - 2 * math.sqrt(max(expected, 0.0)))
        detail = json.dumps({"sent": r.sent, "completed": r.completed, "skipped": r.skipped, "errors": r.errors,
                             "late_sends": r.late_sends, "samples": len(r.samples), "gateway_sent": round(users, 1),
                             "gateway_answered": round(gateway, 1)}, sort_keys=True)
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
                endpoint_id=slot.endpoint_id, model_id=slot.model_id, callers={self.hotkey}, netuid=self.netuid)
            if answers is not None:
                signed.append((sample, answers))
        valid, evidence, drift_sum, drift_count = len(signed), None, 0.0, 0
        recomputed = {id(s): k for k, (s, _) in enumerate(
            random.sample(signed, min(policy.answer_checks_per_test, len(signed))))}  # -> its round
        for sample, answers in signed if model is not None else []:
            fault, comparison = await recomputes.check(slot.model_id, sample.request, answers, model.tolerance,
                                                       recompute_answers=id(sample) in recomputed,
                                                       claimed=claimed_tokens(parse_response(sample.response)),
                                                       rank=recomputed.get(id(sample), 0) * place[1] + place[0])
            if comparison is not None:
                drift_sum, drift_count = drift_sum + comparison.sum_abs, drift_count + comparison.count
            if fault:
                evidence = Evidence(hotkey=slot.hotkey, endpoint_id=slot.endpoint_id, model_id=slot.model_id,
                                    test_id=test.test_id, reason=fault, request_id=sample.request_id,
                                    request=sample.request, response=sample.response)
                break
        drift = {"drift_sum": drift_sum, "drift_count": drift_count}
        own = r.completed * (valid / len(r.samples) if r.samples else 1.0) / policy.test_s
        delivered = round(own + gateway / policy.test_s, 2)
        # Miners serve users first: users answered beyond what the test left of the tested rate displaced test requests
        # one for one, so the test's requests count as asked only up to what the users' answers leave of it. Users
        # refused count in full.
        own_asked = min(r.sent, max(target.tested_rps * policy.test_s - gateway, 0.0))
        asked = (users_asked + own_asked) / policy.test_s
        if evidence is not None:
            return Result(target, "invalid", evidence.reason, delivered, 0.0, evidence=evidence, **latency, **drift)
        users_fault = await self._check_users(test, slot, model, recomputes, place[1] + place[0]) if gateway else ""
        if users_fault:
            return Result(target, "invalid", f"a sampled answer to a user in the test is wrong: {users_fault}",
                          delivered, 0.0, **latency, **drift)
        credited, unchecked = credit(target.tested_rps, delivered, policy, asked), ""
        if users_fault is None:  # answers to users in the test that no checked sample covers prove no capacity
            credited = min(credited, round(own, 2))
            unchecked = "its answers to users in the test are unchecked: credited only its replies to the test"
        if delivered >= policy.pass_fraction * asked:
            return Result(target, "passed", unchecked, delivered, credited, **latency, **drift)
        reason = f"delivered {delivered:.1f} of {asked:.1f} requests/s asked"
        if gateway:
            reason += f" ({gateway / policy.test_s:.1f} of them to the gateway)"
        if valid < len(r.samples):
            reason += f", {len(r.samples) - valid} of {len(r.samples)} sampled responses without a valid receipt"
        return Result(target, "partial", f"{reason}; {unchecked}" if unchecked else reason, delivered, credited,
                      **latency, **drift)

    async def _check_users(self, test: ScheduledTest, slot: Slot, model: catalog_module.CatalogModel | None,
                           recomputes: Recomputes, rank: float) -> str | None:
        """Check the gateways' sample of the endpoint's answers to users in the test: -> why they are wrong, "" if
        they pass (or, without a reference for the model, are well formed, as test answers then), None if there is
        no sample or its recompute could not run (it is checked between tests then)."""
        row = await asyncio.to_thread(self.store.test_sample, test.test_id, slot.hotkey, slot.endpoint_id)
        if row is None or model is None:
            return None
        sample_id, request_id, request_json, response = row
        parsed = json.loads(response)
        fault, comparison = await recomputes.check(slot.model_id, json.loads(request_json), parsed["answers"],
                                                   model.tolerance, recompute_answers=True,
                                                   claimed=claimed_tokens(parsed), rank=rank)
        if fault is None and comparison is None:
            checks = recomputes.checks
            return None if slot.model_id in checks.references or slot.model_id in checks.screens else ""
        evidence = None if fault is None else _gateway_evidence(slot.hotkey, slot.endpoint_id, slot.model_id,
                                                                request_id, fault, parsed)
        await asyncio.to_thread(self._finish_checked, sample_id, fault, evidence, comparison)
        if evidence is not None:
            log.warning("answer of %s/%s to a user in the test of epoch %d is wrong: %s", slot.hotkey,
                        slot.endpoint_id, test.epoch, fault)
        return fault or ""

    # ---- gateway samples ----

    def accept_samples(self, gateway: str, samples: list[Any]) -> int:
        """Keep the samples of known endpoints whose receipt binds the request and the answers to the miner and names
        ``gateway`` as caller (each judged on its own; up to ``KEPT_PER_ENDPOINT`` per endpoint and epoch, proofs of
        malformed answers always); -> how many were kept."""
        if self.policy.gateway_checks_per_epoch == 0:
            return 0
        cat, known = self.catalog(), {(s.hotkey, s.endpoint_id) for s in self._slots}
        epoch, kept = self.follower.head // self.policy.epoch_blocks, 0
        for raw in samples:
            try:
                s = GatewaySample.model_validate(raw)
            except ValueError:
                continue
            response = parse_response(s.response)
            receipt = response.get("receipt") if response is not None else None
            try:
                nonce, sample_id = receipt["receipt"]["request_nonce"], receipt["signature"]
            except (KeyError, TypeError):
                continue
            if ((s.hotkey, s.endpoint_id) not in known or cat.get(s.model_id) is None or not isinstance(nonce, str)
                    or not isinstance(sample_id, str)):
                continue
            answers = signed_answers(response, s.request, request_id=nonce, hotkey=s.hotkey,
                                     endpoint_id=s.endpoint_id, model_id=s.model_id, callers={gateway},
                                     netuid=self.netuid)
            if answers is None:
                continue
            if in_mix_form(s.request.get("state")) or canary_like(s.request.get("state")):
                log.warning("%s/%s answered the gateway a request in the form of test traffic or a canary: a miner "
                            "relays what it is sent through the gateway", s.hotkey, s.endpoint_id)
            kept += self.store.add_sample(sample_id=sample_id, epoch=epoch, gateway=gateway, hotkey=s.hotkey,
                                          endpoint_id=s.endpoint_id, model_id=s.model_id, request_id=nonce,
                                          request=s.request, response=s.response,
                                          proof=readable(s.request, answers) is None,
                                          test_id=s.test_id if s.test_id and self.store.known(s.test_id) else None)
        return kept

    async def check_samples(self, checks: Checks) -> None:
        """Check the chosen gateway samples (``Store.samples_to_check``): a wrong answer becomes evidence, the
        differences go to the endpoint's gateway drift tally. A sample whose recompute could not run is tried again
        later, up to ``MAX_CHECK_ATTEMPTS``; a test starting stops the round."""
        epoch = self.follower.head // self.policy.epoch_blocks
        chosen = await asyncio.to_thread(self.store.samples_to_check, epoch, self.policy.gateway_checks_per_epoch,
                                         self._rng)
        if not chosen:
            return
        checks = await self._usable(checks)
        recomputes, cat = checks.for_test(SAMPLE_BUDGET_S), self.catalog()

        async def check(rank: int, row: tuple) -> None:
            sample_id, _, hotkey, endpoint_id, model_id, request_id, request_json, response = row
            if self._tests_running:
                return  # a test's checks come first; still pending
            model, request = cat.get(model_id), json.loads(request_json)
            if model is None or (model_id not in checks.references and model_id not in checks.screens):
                self.store.finish_sample(sample_id, "unchecked", "no reference for the model")
                return
            parsed = json.loads(response)
            fault, comparison = await recomputes.check(model_id, request, parsed["answers"], model.tolerance,
                                                       recompute_answers=True, claimed=claimed_tokens(parsed),
                                                       rank=rank)
            if fault is None and comparison is None:
                if not self._tests_running and time.monotonic() <= recomputes.deadline:
                    self.store.failed_check(sample_id)
                return
            evidence = None if fault is None else _gateway_evidence(hotkey, endpoint_id, model_id, request_id, fault,
                                                                    parsed)
            await asyncio.to_thread(self._finish_checked, sample_id, fault, evidence, comparison)
            if evidence is not None:
                log.warning("gateway answer of %s/%s is wrong: %s", hotkey, endpoint_id, fault)

        results = await asyncio.gather(*(check(i, row) for i, row in enumerate(chosen)), return_exceptions=True)
        for row, outcome in zip(chosen, results):
            if isinstance(outcome, Exception):
                log.error("checking gateway sample of %s/%s failed: %s", row[2], row[3], outcome)
                self.store.finish_sample(row[0], "unchecked",
                                         f"the auditor could not check it: {type(outcome).__name__}")

    def _finish_checked(self, sample_id: str, fault: str | None, evidence: Evidence | None,
                        comparison: RequestComparison | None) -> None:
        """Record a checked sample, labelled with an epoch whose verdicts are not served yet: under the verdicts'
        lock, which building them holds a while, so never on the auditor's event loop."""
        with self._verdicts_lock:
            self.store.finish_sample(sample_id, "checked", fault or "", evidence,
                                     comparison.sum_abs if comparison else 0.0, comparison.count if comparison else 0,
                                     checked_epoch=self.follower.head // self.policy.epoch_blocks)

    # ---- gateway served counts ----

    def accept_served(self, counts: list[ServedCount], report_id: str = "") -> int:
        """Add what the gateway's endpoints answered per second (known endpoints, recent seconds), their input tokens
        to the epoch's demand per model, and their offers, refusals and durations to the epoch's user tallies;
        -> counts taken. A report is taken once: a gateway posting it again (its answer was lost) adds nothing."""
        now, taken = time.time(), 0
        if report_id:
            with self._demand_lock:
                if report_id in self._reports:
                    return 0
                self._reports[report_id] = now
                for seen in [r for r, t in self._reports.items() if t < now - SERVED_KEEP_S]:
                    del self._reports[seen]
        model_of = {(s.hotkey, s.endpoint_id): s.model_id for s in self._slots}
        with self._demand_lock:
            epoch = self.follower.head // self.policy.epoch_blocks
            demand, users = self._demand.setdefault(epoch, {}), self._users.setdefault(epoch, {})
            for c in counts:
                key = (c.hotkey, c.endpoint_id)
                if key not in model_of or not now - SERVED_KEEP_S <= c.second <= now + 5:
                    continue
                if c.answered or c.attempts:
                    second = self._served.setdefault(key, {}).setdefault(c.second, [0, 0])
                    second[0] += c.answered
                    second[1] += c.attempts
                demand[model_of[key]] = demand.get(model_of[key], 0) + c.input_tokens
                if c.paid_input_tokens is not None:
                    paid = self._paid.setdefault(epoch, {})
                    paid[model_of[key]] = paid.get(model_of[key], 0) + c.paid_input_tokens
                tally = users.setdefault((*key, model_of[key]), UserTally())
                tally.offered += c.offered
                tally.refused += c.refused
                tally.paid_tokens += c.paid_input_tokens or 0
                for bucket, n in c.latency.items():
                    tally.answers += n
                    tally.latency[bucket] = tally.latency.get(bucket, 0) + n
                taken += 1
            cutoff = int(now) - SERVED_KEEP_S
            for key in list(self._served):
                per = self._served[key]
                for second in [s for s in per if s < cutoff]:
                    del per[second]
                if not per:
                    del self._served[key]
        return taken

    def served_in(self, hotkey: str, endpoint_id: str, start_s: float, end_s: float, *,
                  attempts: bool = False) -> float:
        """User requests the gateways sent the endpoint in [start_s, end_s) that it answered (``attempts``: all of
        them), counting partly covered seconds by their covered share."""
        per = self._served.get((hotkey, endpoint_id), {})
        total = 0.0
        for second in range(math.floor(start_s), math.ceil(end_s)):
            n = per.get(second)
            if n:
                total += n[1 if attempts else 0] * (min(end_s, second + 1) - max(start_s, second))
        return total

    def close_demand(self, current: int) -> None:
        """Record the demand of the epoch before ``current`` (once): the input tokens the gateways reported answered
        per model, and of them those users paid for, as tokens/s, folded into each model's EMAs
        (``scoring.demand_ema_alpha``). An epoch the auditor saw no reports for counts as no demand."""
        last = current - 1
        with self._demand_lock:
            for tally in (self._demand, self._paid):
                for epoch in [e for e in tally if e < last]:
                    del tally[epoch]  # too late to count
            if last < 0 or self.store.demand_closed(last):
                return
            ema = self.store.close_demand(last, self._demand.pop(last, {}), self.policy.epoch_blocks * BLOCK_TIME_S,
                                          self.catalog().scoring.demand_ema_alpha, self._paid.pop(last, None))
        if ema:
            log.info("user demand after epoch %d: %s", last,
                     ", ".join(f"{m} {v:.0f} input tokens/s" for m, v in sorted(ema.items())))

    def close_users(self, current: int) -> None:
        """Record what the endpoints did for users in the epoch before ``current`` (once)."""
        last = current - 1
        with self._demand_lock:
            for epoch in [e for e in self._users if e < last]:
                del self._users[epoch]  # too late to count
            if last < 0 or self.store.users_closed(last):
                return
            tallies = self._users.pop(last, {})
            if tallies:
                self.store.close_users(last, tallies)

    def close_epoch(self, current: int) -> None:
        self.close_demand(current)
        self.close_users(current)

    def settle(self) -> None:
        """Confirm tests whose selection block is final with the hash they were selected by; void the others. Close
        the users' demand of finished epochs."""
        with self._demand_lock:  # not while a new epoch length renumbers the rows
            epoch = self.follower.head // self.policy.epoch_blocks
            if epoch != self._pruned_epoch:
                self.store.prune(epoch - KEEP_EPOCHS)
                self._pruned_epoch = epoch
        if self.follower.head > 0:
            self.close_epoch(epoch)
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
        with self._verdicts_lock:
            return self._verdicts_for(epoch)

    def _verdicts_for(self, epoch: int) -> SignedVerdicts | None:
        cached = self._verdicts.get(epoch)
        if cached is not None:
            return cached
        if self.follower.head <= 0:
            return None  # the chain head is not known yet
        policy, cat = self.policy, self.catalog()
        current, first = self.follower.head // policy.epoch_blocks, epoch - policy.credit_ttl_epochs
        if not current - 2 <= epoch <= current:
            return None
        served = self.store.served_verdicts(epoch)
        if served is not None:  # served before a restart: the same document
            signed = SignedVerdicts.model_validate_json(served)
            self._verdicts = {e: v for e, v in self._verdicts.items() if e >= current - 2} | {epoch: signed}
            return signed
        if self.store.pending(first, epoch - 1):
            return None
        self.close_epoch(current)

        users = self.store.users(epoch - 1)

        def with_users(c: Credit) -> Credit:
            t = users.get((c.hotkey, c.endpoint_id, c.model_id))  # measured on the model credited
            if t is None:
                return c
            return c.model_copy(update={"user_offers": t.offered, "user_refusals": t.refused,
                                        "user_answers": t.answers,
                                        "user_latency_ms": round(latency_median_ms(t.latency) or 0.0, 1)})

        demand, paid = self.store.demand_ema(epoch - 1), self.store.paid_demand(epoch - 1)
        # only paid usage moves the shares, once the gateways tell it apart; by at most max_share_step per epoch
        shares = self.store.close_shares(epoch, model_shares(demand if paid is None else paid, cat),
                                         cat.scoring.max_share_step, start=model_shares({}, cat))
        told = {} if paid is None else {"paid_demand": paid, "paid_served": [
            PaidServed(hotkey=hk, endpoint_id=ep, model_id=model, input_tokens=tokens).model_dump()
            for (hk, ep, model), tokens in sorted(self.store.paid(epoch - 1, epoch - 1).items())]}
        verdicts = Verdicts(netuid=self.netuid, auditor=self.hotkey, epoch=epoch,
                            credits=[with_users(c) for c in self.store.credits(first, epoch - 1, policy,
                                                                               _tolerance_in(cat))],
                            invalid=self.store.evidence(first, epoch - 1), demand=demand, shares=shares, **told)
        signed = SignedVerdicts.create(verdicts, self.keypair)
        self.store.serve_verdicts(epoch, signed.model_dump_json())
        self._verdicts = {e: v for e, v in self._verdicts.items() if e >= current - 2} | {epoch: signed}
        return signed

    def routes(self, probation: Callable[[], Collection[str]]) -> SignedRoutes | None:
        """The endpoints the gateways may route to (``protocol.gateway``), signed; None until there are verdicts and
        the miners have been discovered once (a gateway keeps its last table meanwhile). ``probation`` gives the
        endpoint keys on probation at the owner's validator. Rebuilt at most every ``ROUTES_TTL_S``.

        The credits come from the verdicts; while they cover no confirmed test (the auditor missed tests, or they were
        void), from the latest confirmed test the auditor holds, whatever its age, leaving out hotkeys with a wrong
        answer proven since: an endpoint's score needs a recent test, routing users to it does not."""
        if not self._discovered:
            return None
        now = time.monotonic()
        if self._routes is not None and now - self._routes[0] < ROUTES_TTL_S:
            return self._routes[1]
        cat, current = self.catalog(), self.follower.head // self.policy.epoch_blocks
        signed = self.verdicts(current) or self.verdicts(current - 1)
        if signed is None:
            return None
        v = signed.verdicts
        excluded = {key.split("/", 1)[0] for key in probation()}  # probation covers the whole hotkey
        invalid = {e.hotkey for e in v.invalid} | self.store.offenders(current - 1)
        credited = v.credits
        if not credited:
            credited = self.store.credits(0, current, self.policy, _tolerance_in(cat))
            invalid |= self.store.offenders(min((c.test_epoch for c in credited), default=current))
        credits = {(c.hotkey, c.endpoint_id, c.model_id): c for c in credited
                   if c.qualified and c.credited_rps > 0 and not c.drift}
        routes = []
        for s in self._slots:
            credit_ = credits.get((s.hotkey, s.endpoint_id, s.model_id))
            model = cat.get(s.model_id)
            if (credit_ is None or model is None or model.status != "active" or s.hotkey in invalid
                    or s.hotkey in excluded):
                continue
            routes.append(Route(hotkey=s.hotkey, endpoint_id=s.endpoint_id, model_id=s.model_id, url=s.url,
                                credited_rps=min(credit_.credited_rps, s.declared_rps), latency_ms=credit_.latency_ms))
        table = Routes(netuid=self.netuid, auditor=self.hotkey, epoch=current, issued_ms=int(time.time() * 1000),
                       catalog_digest=cat.digest(), routes=routes)
        self._routes = (now, SignedRoutes.create(table, self.keypair))
        return self._routes[1]

    def outcomes(self, hotkey: str) -> list[ThroughputOutcome]:
        """The hotkey's results in the last ``STATUS_TESTS`` tests, for its status."""
        return self.store.outcomes(hotkey)

    @property
    def revision(self) -> int:
        """Changes whenever what ``outcomes`` lists may have (``Store.revision``)."""
        return self.store.revision
