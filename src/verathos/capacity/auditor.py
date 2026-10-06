"""Auditor side of the capacity audit (the subnet owner's validator).

The auditor follows current-head blocks with the same public derivations the miners use, so both sides agree on
every window and cohort without messages. For each drafted slot it then records the miner's signed artifacts:

- the final receipt is timed against the auditor's own observation of B_start (``deadline_s + transport_grace_s``)
  and must arrive before the auditor has seen B_proof, since the proof challenge is derived from B_proof;
- the proof payload must arrive within ``payload_deadline_s`` of B_proof and is verified against the slot's
  scheduled workload, the B_start-derived proof seed and the B_proof-derived challenge.

Both artifacts name the block hash the miner used; a miner whose node followed another fork misses the slot.

Outcomes become evidence once all three window blocks are final and unchanged; otherwise the window is void. In
``score_gate`` mode an endpoint whose evidence inside the repeat window crosses a policy threshold is gated (scored
zero). Validator-side faults never count against a miner: windows the auditor could not follow from their
selection block (start-up, a stalled block feed, B_start observed after the next block) and windows opened while a
newly committed policy propagates are recorded as void, so their slots still count as busy but give no evidence.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from verathos.capacity import core
from verathos.capacity.proof.combined import COMBINED_PROOF_FORMAT, verify_combined_proof

log = logging.getLogger(__name__)

RECEIPT_TYPE = "capacity_audit_final_receipt"
PROOF_TYPE = "capacity_audit_proof_payload"
UNKNOWN_SLOT_ERROR = "unknown audit slot"
FINALITY_CHECK_BLOCKS = 10
SCHEMA_VERSION = 1

# Slot outcomes. Open states change; the others are evidence (or neutral).
PENDING, TIMING_PASS, VERIFYING = "pending", "timing_pass", "verifying"
PASSED, TIMING_MISS, NO_SHOW, MISSING_PROOF, INVALID = "passed", "timing_miss", "no_show", "missing_proof", "invalid"
NEUTRAL = "neutral"
OPEN_OUTCOMES = (PENDING, TIMING_PASS, VERIFYING)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    audit_id TEXT PRIMARY KEY, epoch INTEGER NOT NULL, selection_block INTEGER NOT NULL,
    start_block INTEGER NOT NULL, proof_block INTEGER NOT NULL, selection_hash TEXT NOT NULL,
    start_hash TEXT, proof_hash TEXT, start_observed_at REAL, proof_observed_at REAL,
    chain_status TEXT NOT NULL DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS slots (
    audit_id TEXT NOT NULL, slot_id TEXT NOT NULL, hotkey TEXT NOT NULL, endpoint_id TEXT NOT NULL,
    gpu_class TEXT NOT NULL, lease_id TEXT NOT NULL, workload_spec TEXT NOT NULL, pass_count INTEGER NOT NULL,
    deadline_s REAL NOT NULL, final_at REAL, final_start_hash TEXT, final_transcript TEXT, final_commit TEXT,
    final_artifact TEXT, proof_at REAL, outcome TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (audit_id, slot_id));
CREATE INDEX IF NOT EXISTS slots_by_endpoint ON slots (hotkey, endpoint_id);
"""


class AuditorChain(Protocol):
    def head(self) -> tuple[int, bytes | None]: ...

    def block_hash(self, block_number: int) -> bytes | None: ...

    def epoch_blocks(self) -> int: ...

    def finalized_block(self) -> int: ...


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any]


class Auditor:
    """Schedules windows, ingests artifacts, verifies proofs and reports gated endpoints."""

    def __init__(self, *, netuid: int, policy: core.AuditPolicy, chain: AuditorChain, db_path: Path,
                 block_time_s: float = core.DEFAULT_BLOCK_TIME_S, verify_workers: int = 2,
                 clock: Callable[[], float] = time.time) -> None:
        policy.validate(block_time_s)
        self.netuid, self.policy, self.chain, self.block_time_s, self.clock = netuid, policy, chain, block_time_s, clock
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        if self._db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            log.info("capacity audit database %s has another schema; starting it afresh", db_path)
            self._db.executescript("DROP TABLE IF EXISTS slots; DROP TABLE IF EXISTS windows;")
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._db.executescript(_SCHEMA)
        self._lock = threading.RLock()
        self._slots: list[core.Slot] | None = None
        self._verifier = ThreadPoolExecutor(max_workers=verify_workers, thread_name_prefix="capacity-verify")
        self._last_block = 0
        self._transition_until = 0
        self._running = False

    # ---- inputs ----

    def set_policy(self, policy: core.AuditPolicy) -> None:
        """Adopt a newly committed policy. Windows opened until every miner can have adopted it are void."""
        if policy == self.policy:
            return
        policy.validate(self.block_time_s)
        propagation = math.ceil(2 * core.POLICY_WATCH_S / self.block_time_s)
        with self._lock:
            self._transition_until = self._last_block + propagation + max(self._hold_blocks(self.policy),
                                                                          self._hold_blocks(policy))
            self.policy = policy
        log.info("capacity policy changed; windows until block %d are void", self._transition_until)

    def _hold_blocks(self, policy: core.AuditPolicy) -> int:
        """Blocks from a window's selection until its slots may be drafted again."""
        return (policy.lead_blocks + max(1, policy.proof_challenge_delay_blocks)
                + core.payload_cooldown_blocks(policy, self.block_time_s))

    def update_slots(self, slots: list[core.Slot]) -> None:
        """The slot universe: every endpoint listed in an accepted miner manifest. Blocks are followed only once
        it is known."""
        with self._lock:
            self._slots = list(slots)

    def start(self, poll_s: float = 0.5) -> None:
        self._running = True
        threading.Thread(target=self._run, args=(poll_s,), name="capacity-auditor", daemon=True).start()

    def stop(self) -> None:
        self._running = False
        self._verifier.shutdown(wait=False, cancel_futures=True)

    def _run(self, poll_s: float) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                log.exception("capacity auditor block feed error")
            time.sleep(poll_s)

    def poll_once(self) -> None:
        """Process every block from the last processed one up to the current head, in order. On start-up the
        recent blocks are replayed so windows still holding slots are known (as void)."""
        if self._slots is None:
            return
        head, head_hash = self.chain.head()
        if head <= 0:
            return
        if self._last_block <= 0:
            self._last_block = head - self._hold_blocks(self.policy) - 1
        for n in range(self._last_block + 1, head + 1):
            h = head_hash if n == head and head_hash else self.chain.block_hash(n)
            if h is None:
                return
            self.on_block(n, h, head)
            self._last_block = n

    def current_epoch(self) -> int:
        return self._last_block // self.chain.epoch_blocks()

    # ---- blocks ----

    def on_block(self, block_number: int, block_hash: bytes, head: int) -> None:
        now = self.clock()
        epoch_blocks = self.chain.epoch_blocks()
        if block_number % FINALITY_CHECK_BLOCKS == 0:
            self._check_finality()
        with self._lock:
            self._observe_start(block_number, block_hash, head, now)
            self._observe_proof_block(block_number, block_hash, now)
            self._expire(now)
            window = core.open_window(self.netuid, block_number, block_hash, epoch_blocks, self.policy)
            if window is not None:
                late = head >= window.start_block
                self._schedule(window, block_hash, "auditor_late" if late else
                               "policy_transition" if block_number <= self._transition_until else "")

    def _schedule(self, window: core.Window, block_hash: bytes, void_reason: str) -> None:
        drafted = core.draft(window, self._slots or [], self.policy, self._busy_slots(window.selection_block))
        if not drafted:
            return
        self._db.execute("INSERT OR IGNORE INTO windows (audit_id, epoch, selection_block, start_block, proof_block, "
                         "selection_hash, chain_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (window.audit_id, window.epoch, window.selection_block, window.start_block,
                          window.proof_block, block_hash.hex(), "void" if void_reason else "pending"))
        for slot, row in drafted:
            spec = core.gpu_workload_spec(row)
            self._db.execute(
                "INSERT OR IGNORE INTO slots (audit_id, slot_id, hotkey, endpoint_id, gpu_class, lease_id, "
                "workload_spec, pass_count, deadline_s, outcome, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (window.audit_id, slot.slot_id, slot.hotkey, slot.endpoint_id, row.gpu_name,
                 core.lease_id(slot, window.epoch), json.dumps(spec, sort_keys=True), spec["pass_count"],
                 self.policy.deadline_s, NEUTRAL if void_reason else PENDING, void_reason))
        log.info("capacity window %s at block %d: %d slots drafted%s", window.audit_id[:12], window.selection_block,
                 len(drafted), f" (void: {void_reason})" if void_reason else "")

    def _busy_slots(self, block_number: int) -> set[str]:
        """Slots still inside an earlier window's hold (selection to payload cooldown after B_proof), void windows
        included: the miners drafted them all the same."""
        cooldown = core.payload_cooldown_blocks(self.policy, self.block_time_s)
        rows = self._db.execute("SELECT s.slot_id FROM slots s JOIN windows w USING (audit_id) "
                                "WHERE w.proof_block + ? >= ?", (cooldown, block_number))
        return {r["slot_id"] for r in rows}

    def _observe_start(self, block_number: int, block_hash: bytes, head: int, now: float) -> None:
        for w in self._db.execute("SELECT audit_id FROM windows WHERE start_block = ? AND start_hash IS NULL",
                                  (block_number,)).fetchall():
            self._db.execute("UPDATE windows SET start_hash = ?, start_observed_at = ? WHERE audit_id = ?",
                             (block_hash.hex(), now, w["audit_id"]))
            if head > block_number:
                self._set_outcome(w["audit_id"], None, NEUTRAL, "auditor_observed_start_late", only_open=True)
                continue
            for s in self._db.execute("SELECT * FROM slots WHERE audit_id = ? AND final_at IS NOT NULL",
                                      (w["audit_id"],)).fetchall():
                self._time_final(s)

    def _observe_proof_block(self, block_number: int, block_hash: bytes, now: float) -> None:
        self._db.execute("UPDATE windows SET proof_hash = ?, proof_observed_at = ? "
                         "WHERE proof_block = ? AND proof_hash IS NULL", (block_hash.hex(), now, block_number))

    def _expire(self, now: float) -> None:
        grace = self.policy.transport_grace_s
        self._db.execute(
            "UPDATE slots SET outcome = ?, reason = 'missing_final_receipt' WHERE outcome = ? AND audit_id IN "
            "(SELECT audit_id FROM windows WHERE start_observed_at IS NOT NULL) AND ? > deadline_s + ? + "
            "(SELECT start_observed_at FROM windows w WHERE w.audit_id = slots.audit_id)",
            (NO_SHOW, PENDING, now, grace))
        self._db.execute(
            "UPDATE slots SET outcome = ?, reason = 'missing_proof_payload' WHERE outcome = ? AND proof_at IS NULL "
            "AND ? > ? + (SELECT proof_observed_at FROM windows w WHERE w.audit_id = slots.audit_id)",
            (MISSING_PROOF, TIMING_PASS, now, self.policy.payload_deadline_s))

    def _check_finality(self) -> None:
        """Confirm or void windows whose blocks are final. Chain reads happen outside the lock, so ingest is never
        held up by a slow node."""
        finalized = self.chain.finalized_block()
        with self._lock:
            pending = self._db.execute("SELECT * FROM windows WHERE chain_status = 'pending' AND proof_block <= ?",
                                       (finalized,)).fetchall()
        decided = []
        for w in pending:
            stored = (w["selection_hash"], w["start_hash"], w["proof_hash"])
            if None in stored:
                decided.append((w["audit_id"], "void"))  # not every window block was observed: no evidence
                continue
            final = [self.chain.block_hash(b) for b in (w["selection_block"], w["start_block"], w["proof_block"])]
            if None not in final:
                decided.append((w["audit_id"], "confirmed" if [h.hex() for h in final] == list(stored) else "void"))
        with self._lock:
            for audit_id, status in decided:
                self._db.execute("UPDATE windows SET chain_status = ? WHERE audit_id = ? AND chain_status = 'pending'",
                                 (status, audit_id))
                if status == "void":
                    self._set_outcome(audit_id, None, NEUTRAL, "window_not_final")

    # ---- ingest ----

    def proof_precheck(self, audit_id: Any, slot_id: Any) -> Response | None:
        """The answer to a proof upload that can be given before its body is read, or None to read it."""
        with self._lock:
            row = self._row(audit_id, slot_id)
            if row is None:
                return Response(400, {"error": UNKNOWN_SLOT_ERROR})
            return self._proof_state(row)

    def receive(self, path_kind: str, artifact: Any, received_at: float | None = None) -> Response:
        """Ingest one signed artifact. ``path_kind`` is ``receipt`` or ``proof``."""
        received_at = self.clock() if received_at is None else received_at
        if not isinstance(artifact, dict):
            return Response(400, {"error": "artifact must be an object"})
        kind = artifact.get("artifact_type")
        if (path_kind == "receipt" and kind != RECEIPT_TYPE) or (path_kind == "proof" and kind != PROOF_TYPE):
            return Response(400, {"error": "unexpected artifact_type for this path"})
        if artifact.get("protocol_version") != core.PROTOCOL_VERSION:
            return Response(400, {"error": "unsupported protocol_version"})
        hotkey = artifact.get("hotkey")
        if not isinstance(hotkey, str) or not core.verify_artifact_signature(artifact, hotkey):
            return Response(400, {"error": "invalid miner_signature"})
        with self._lock:
            row = self._row(artifact.get("audit_id"), artifact.get("slot_id"))
            if row is None or row["hotkey"] != hotkey:
                return Response(400, {"error": UNKNOWN_SLOT_ERROR})
            mismatch = self._binding_mismatch(row, artifact)
            if mismatch:
                return Response(400, {"error": f"{mismatch} mismatch"})
            if kind == RECEIPT_TYPE:
                return self._final(row, artifact, received_at)
            return self._proof(row, artifact, received_at)

    def _row(self, audit_id: Any, slot_id: Any) -> sqlite3.Row | None:
        if not isinstance(audit_id, str) or not isinstance(slot_id, str):
            return None
        return self._db.execute(
            "SELECT s.*, w.selection_block, w.start_block, w.proof_block, w.start_hash, w.proof_hash, "
            "w.start_observed_at, w.proof_observed_at, w.chain_status FROM slots s JOIN windows w "
            "USING (audit_id) WHERE s.audit_id = ? AND s.slot_id = ?", (audit_id, slot_id)).fetchone()

    def _binding_mismatch(self, row: sqlite3.Row, artifact: dict[str, Any]) -> str:
        expected = {"netuid": self.netuid, "endpoint_id": row["endpoint_id"], "claimed_gpu_class": row["gpu_class"],
                    "gpu_index": 0, "lease_id": row["lease_id"], "B_select": row["selection_block"],
                    "B_start": row["start_block"], "B_proof": row["proof_block"], "pass_count": row["pass_count"]}
        return next((k for k, v in expected.items() if artifact.get(k) != v), "")

    def _final(self, row: sqlite3.Row, artifact: dict[str, Any], received_at: float) -> Response:
        transcript, commit = artifact.get("final_transcript_commit"), artifact.get("combined")
        start_hash = artifact.get("B_start_hash")
        if not isinstance(transcript, str) or not transcript or not isinstance(commit, dict) \
                or commit.get("format") != COMBINED_PROOF_FORMAT \
                or commit.get("combined_transcript_root") != transcript or not isinstance(start_hash, str):
            return Response(400, {"error": "final receipt does not carry a combined commitment and B_start hash"})
        encoded = core.canonical_json(artifact)
        if row["final_at"] is not None:
            if row["final_artifact"] == encoded:
                return Response(200, {"ok": True, "outcome": row["outcome"]})
            return Response(409, {"error": "final receipt already committed"})
        if row["outcome"] not in (PENDING, NO_SHOW):
            return Response(200, {"ok": True, "outcome": row["outcome"]})
        self._db.execute("UPDATE slots SET final_at = ?, final_start_hash = ?, final_transcript = ?, final_commit = ?, "
                         "final_artifact = ? WHERE audit_id = ? AND slot_id = ?",
                         (received_at, start_hash.removeprefix("0x"), transcript, json.dumps(commit, sort_keys=True),
                          encoded, row["audit_id"], row["slot_id"]))
        if row["start_observed_at"] is not None:
            self._time_final(self._slot(row["audit_id"], row["slot_id"]))
        elif row["proof_hash"] is not None:
            self._set_outcome(row["audit_id"], row["slot_id"], INVALID, "final_receipt_after_proof_challenge")
        return Response(200, {"ok": True, "outcome": self._slot(row["audit_id"], row["slot_id"])["outcome"]})

    def _time_final(self, slot: sqlite3.Row) -> None:
        """Time a final receipt against the observed B_start. An on-time receipt must also precede B_proof: the
        proof challenge is derived from it, so a transcript committed after it proves nothing."""
        if slot["outcome"] not in (PENDING, NO_SHOW):
            return
        window = self._db.execute("SELECT start_hash, start_observed_at, proof_observed_at FROM windows "
                                  "WHERE audit_id = ?", (slot["audit_id"],)).fetchone()
        if slot["final_start_hash"] != window["start_hash"]:
            outcome, reason = MISSING_PROOF, "start_hash_mismatch"
        elif slot["final_at"] > window["start_observed_at"] + slot["deadline_s"] + self.policy.transport_grace_s:
            outcome, reason = TIMING_MISS, "deadline_exceeded"
        elif window["proof_observed_at"] is not None and slot["final_at"] >= window["proof_observed_at"]:
            outcome, reason = INVALID, "final_receipt_after_proof_challenge"
        else:
            outcome, reason = TIMING_PASS, ""
        self._set_outcome(slot["audit_id"], slot["slot_id"], outcome, reason)

    def _proof_state(self, row: sqlite3.Row) -> Response | None:
        """None if the slot is waiting for its proof, else the answer to a proof upload."""
        if row["outcome"] != TIMING_PASS:
            if row["outcome"] in OPEN_OUTCOMES and row["final_at"] is None:
                return Response(409, {"error": "final receipt required before proof payload"})
            return Response(200, {"ok": True, "outcome": row["outcome"]})
        if row["proof_hash"] is None:
            return Response(409, {"error": "proof challenge block not observed yet"})
        return None

    def _proof(self, row: sqlite3.Row, artifact: dict[str, Any], received_at: float) -> Response:
        early = self._proof_state(row)
        if early is not None:
            return early
        if received_at > row["proof_observed_at"] + self.policy.payload_deadline_s:
            self._set_outcome(row["audit_id"], row["slot_id"], MISSING_PROOF, "proof_payload_after_deadline")
            return Response(200, {"ok": True, "outcome": MISSING_PROOF})
        if str(artifact.get("B_proof_hash") or "").removeprefix("0x") != row["proof_hash"]:
            self._set_outcome(row["audit_id"], row["slot_id"], MISSING_PROOF, "proof_hash_mismatch")
            return Response(200, {"ok": True, "outcome": MISSING_PROOF})
        payload = artifact.get("sampled_pass_proof")
        self._db.execute("UPDATE slots SET proof_at = ? WHERE audit_id = ? AND slot_id = ?",
                         (received_at, row["audit_id"], row["slot_id"]))
        self._set_outcome(row["audit_id"], row["slot_id"], VERIFYING, "")
        self._verifier.submit(self._verify, dict(row), payload)
        return Response(202, {"ok": True, "outcome": VERIFYING})
    def _verify(self, row: dict[str, Any], payload: Any) -> None:
        try:
            ok, reason = verify_combined_proof(
                proof=payload if isinstance(payload, dict) else {},
                final_commit=json.loads(row["final_commit"]),
                workload_spec=json.loads(row["workload_spec"]),
                lease_id=row["lease_id"],
                gpu_index=0,
                proof_seed_hex=core.derive_proof_seed(bytes.fromhex(row["start_hash"]), row["slot_id"], 0),
                b_proof_seed_hex=core.derive_proof_challenge_seed(row["final_transcript"],
                                                                  bytes.fromhex(row["proof_hash"]), row["lease_id"],
                                                                  row["slot_id"], 0),
            )
            outcome, reason = (PASSED, "") if ok else (INVALID, reason)
        except Exception as exc:
            log.exception("capacity proof verification error for %s", row["slot_id"][:12])
            outcome, reason = NEUTRAL, f"auditor_verify_error:{type(exc).__name__}"
        with self._lock:
            self._set_outcome(row["audit_id"], row["slot_id"], outcome, reason, only_if=VERIFYING)
        log.info("capacity proof %s %s/%s: %s %s", row["audit_id"][:12], row["hotkey"][:8], row["endpoint_id"],
                 outcome, reason)

    # ---- state helpers ----

    def _slot(self, audit_id: str, slot_id: str) -> sqlite3.Row:
        return self._db.execute("SELECT * FROM slots WHERE audit_id = ? AND slot_id = ?",
                                (audit_id, slot_id)).fetchone()

    def _set_outcome(self, audit_id: str, slot_id: str | None, outcome: str, reason: str, *,
                     only_open: bool = False, only_if: str | None = None) -> None:
        sql, args = "UPDATE slots SET outcome = ?, reason = ? WHERE audit_id = ?", [outcome, reason, audit_id]
        if slot_id is not None:
            sql, args = sql + " AND slot_id = ?", args + [slot_id]
        if only_open:
            sql, args = sql + f" AND outcome IN ({','.join('?' * len(OPEN_OUTCOMES))})", args + list(OPEN_OUTCOMES)
        if only_if is not None:
            sql, args = sql + " AND outcome = ?", args + [only_if]
        self._db.execute(sql, args)

    # ---- outputs ----

    def drained(self, hotkey: str, endpoint_id: str) -> bool:
        """Whether the endpoint is inside an audit hold (by time since its last drafting, like the miner's own
        drain); its canaries wait until the hold ends."""
        hold_blocks = core.active_hold_seconds(self.policy, self.block_time_s) / self.block_time_s
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(w.selection_block) AS b FROM slots s JOIN windows w USING (audit_id) "
                "WHERE s.hotkey = ? AND s.endpoint_id = ?", (hotkey, endpoint_id)).fetchone()
        return row["b"] is not None and self._last_block - row["b"] <= hold_blocks

    def evidence(self, epoch: int) -> dict[tuple[str, str], dict[str, int]]:
        """Final outcomes per endpoint from confirmed windows of the ``repeat_window_epochs`` epochs before
        ``epoch``. Only closed epochs count, so every validator that asks during ``epoch`` gets the same answer."""
        counts: dict[tuple[str, str], dict[str, int]] = {}
        with self._lock:
            rows = self._db.execute(
                "SELECT s.hotkey, s.endpoint_id, s.outcome FROM slots s JOIN windows w USING (audit_id) "
                "WHERE w.chain_status = 'confirmed' AND w.epoch BETWEEN ? AND ? AND s.outcome IN (?, ?, ?, ?)",
                (epoch - self.policy.repeat_window_epochs, epoch - 1, TIMING_MISS, NO_SHOW, MISSING_PROOF,
                 INVALID)).fetchall()
        for r in rows:
            c = counts.setdefault((r["hotkey"], r["endpoint_id"]), {})
            c[r["outcome"]] = c.get(r["outcome"], 0) + 1
        return counts

    def gated(self, epoch: int) -> dict[tuple[str, str], str]:
        """Endpoints whose evidence crosses a gate threshold, with the rule that fired. Empty in observe mode."""
        if self.policy.mode != "score_gate":
            return {}
        p, out = self.policy, {}
        for key, c in self.evidence(epoch).items():
            if c.get(INVALID, 0) >= p.invalid_proof_misses_for_zero_score:
                out[key] = "invalid_proof"
            elif c.get(MISSING_PROOF, 0) + c.get(NO_SHOW, 0) >= p.hard_proof_misses_for_zero_score:
                out[key] = "missing_evidence"
            elif c.get(TIMING_MISS, 0) >= p.timing_misses_for_zero_score:
                out[key] = "timing"
        return out

    def prune(self, epoch: int) -> None:
        """Drop windows that can no longer contribute evidence."""
        keep_from = epoch - self.policy.repeat_window_epochs - 1
        with self._lock:
            self._db.execute("DELETE FROM slots WHERE audit_id IN (SELECT audit_id FROM windows WHERE epoch < ?)",
                             (keep_from,))
            self._db.execute("DELETE FROM windows WHERE epoch < ?", (keep_from,))
