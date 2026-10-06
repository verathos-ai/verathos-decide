"""Miner side of the capacity audit.

The worker is non-interactive. It follows current-head blocks, derives every audit window and cohort locally from
public chain data and this miner's committed manifest, and when one of its endpoints is drafted it:

1. drains that endpoint (the miner service answers 503 for it until the audit ends),
2. prelaunches the bench on that endpoint's GPU so it is hot before B_start,
3. releases the bench at B_start with proof_seed = H(B_start hash, slot), publishing a pass-0 receipt as soon as the
   first pass commits and a final receipt when the timed work commits its transcript,
4. derives the challenge seed from the B_proof hash once it exists, lets the bench build the sampled proof, and
5. publishes the proof payload.

All artifacts are signed by the miner hotkey and sent only to the audit validator (the policy's ``auditor_hotkey``),
found through its axon. Several endpoints drafted in the same window bench concurrently on their own GPUs.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from bittensor_wallet import Keypair

from verathos.capacity import core
from verathos.capacity.proof.combined import COMBINED_PROOF_FORMAT, final_commitment

log = logging.getLogger(__name__)

RECEIPT_PATH = "/capacity/v1/receipt"
PROOF_PATH = "/capacity/v1/proof"
UNKNOWN_SLOT_ERROR = "unknown audit slot"


class ChainView(Protocol):
    """The chain data the worker needs."""

    def head(self) -> tuple[int, bytes | None]: ...

    def block_hash(self, block_number: int) -> bytes | None: ...

    def epoch_blocks(self) -> int: ...

    def auditor_url(self) -> str | None: ...


@dataclass(frozen=True)
class AuditEndpoint:
    """One endpoint as committed in the manifest, plus the local GPU it runs on."""

    endpoint_id: str
    url: str
    model_id: str
    gpu_name: str
    vram_gb: int
    cuda_device: int


@dataclass(frozen=True)
class SelectedAudit:
    endpoint: AuditEndpoint
    slot: core.Slot
    gpu_class: core.GpuClass
    workload_spec: dict[str, Any]
    audit_id: str
    selection_block: int
    audit_block: int
    proof_block: int
    lease: str

    @property
    def passes(self) -> int:
        return int(self.workload_spec["pass_count"])


class DrainState:
    """Endpoints currently drained for an audit; shared with the miner service."""

    def __init__(self) -> None:
        self._until: dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, endpoint_id: str, until_ts: float) -> None:
        with self._lock:
            self._until[endpoint_id] = max(self._until.get(endpoint_id, 0.0), until_ts)

    def clear(self, endpoint_id: str) -> None:
        with self._lock:
            self._until.pop(endpoint_id, None)

    def drained(self, endpoint_id: str) -> bool:
        with self._lock:
            until = self._until.get(endpoint_id)
            if until is None:
                return False
            if time.time() > until:
                self._until.pop(endpoint_id, None)
                return False
            return True


def bench_extension_available() -> tuple[bool, str]:
    """Whether the bench and its native extensions import in a fresh interpreter."""
    code = ("import verathos.capacity.bench.combined as c; c.load_extension(); "
            "from verathos.capacity.zk.block_prover import load_extension; load_extension()")
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stderr or proc.stdout or "")[-500:]


def default_bench_command(audit: SelectedAudit, *, out_dir: Path, challenge_file: Path, ready_file: Path,
                          start_file: Path, start_timeout_s: float, challenge_timeout_s: float) -> list[str]:
    return [sys.executable, "-m", "verathos.capacity.bench.combined", "--lease-id", audit.lease,
            "--workload", json.dumps(audit.workload_spec, sort_keys=True), "--out-dir", str(out_dir),
            "--start-file", str(start_file), "--ready-file", str(ready_file), "--start-timeout-s", str(start_timeout_s),
            "--challenge-file", str(challenge_file), "--challenge-timeout-s", str(challenge_timeout_s)]


def _root_hex(root_words: Any) -> str:
    if isinstance(root_words, str):
        raw = root_words.strip().removeprefix("0x")
        if len(raw) == 64:
            try:
                bytes.fromhex(raw)
                return raw
            except ValueError:
                pass
    return core.root_words_digest(root_words)


class _Publisher:
    """Ordered delivery of signed artifacts to the audit validator."""

    def __init__(self, chain: ChainView, client: httpx.Client) -> None:
        self._chain, self._client = chain, client
        self._queue: queue.Queue[tuple[str, dict, int, float]] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._rejected: set[str] = set()
        threading.Thread(target=self._run, name="capacity-publisher", daemon=True).start()

    def publish(self, path: str, artifact: dict, *, attempts: int, retry_delay_s: float) -> None:
        try:
            self._queue.put_nowait((path, artifact, max(1, attempts), max(0.0, retry_delay_s)))
        except queue.Full:
            log.warning("capacity publish queue full; dropping %s for %s", artifact.get("artifact_type"),
                        str(artifact.get("audit_id"))[:12])

    def stop(self) -> None:
        self._stop.set()

    def wait_idle(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path, artifact, attempts, delay = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._deliver(path, artifact, attempts, delay)
            except Exception:
                log.exception("capacity publish failed")
            finally:
                self._queue.task_done()

    def _deliver(self, path: str, artifact: dict, attempts: int, delay: float) -> None:
        audit_id = str(artifact.get("audit_id") or "")
        for attempt in range(1, attempts + 1):
            if audit_id in self._rejected:
                return
            base = self._chain.auditor_url()
            if base:
                try:
                    r = self._client.post(base.rstrip("/") + path, json=artifact, timeout=10.0)
                except httpx.HTTPError as exc:
                    log.warning("capacity publish error %s %s: %s", artifact.get("artifact_type"), audit_id[:12], exc)
                else:
                    if r.status_code < 300:
                        return
                    if r.status_code == 400 and UNKNOWN_SLOT_ERROR in r.text.lower():
                        self._rejected.add(audit_id)
                        log.info("auditor does not know audit %s; stopping delivery", audit_id[:12])
                        return
                    log.warning("capacity publish %s %s -> %d %s", artifact.get("artifact_type"), audit_id[:12],
                                r.status_code, r.text[:200])
                    if r.status_code not in (409, 425, 429, 500, 502, 503, 504):
                        return
            else:
                log.warning("no auditor endpoint known; retrying %s", artifact.get("artifact_type"))
            if attempt < attempts and self._stop.wait(max(0.1, delay)):
                return


class AuditWorker:
    """Capacity-audit worker for all endpoints of one miner hotkey."""

    def __init__(
        self,
        *,
        netuid: int,
        keypair: Keypair,
        endpoints: list[AuditEndpoint],
        policy: core.AuditPolicy,
        chain: ChainView,
        drain: DrainState,
        client: httpx.Client | None = None,
        bench_command: Callable[..., list[str]] = default_bench_command,
        poll_s: float = 0.5,
        block_time_s: float = core.DEFAULT_BLOCK_TIME_S,
        work_dir: Path | None = None,
    ) -> None:
        policy.validate(block_time_s)
        self.netuid, self.keypair, self.endpoints, self.policy = netuid, keypair, endpoints, policy
        self.chain, self.drain = chain, drain
        self.bench_command, self.poll_s, self.block_time_s = bench_command, poll_s, block_time_s
        self.work_dir = work_dir
        self.publisher = _Publisher(chain, client or httpx.Client())
        self._running = False
        self._last_block = 0
        self._hashes: dict[int, bytes] = {}
        self._hash_cond = threading.Condition()
        self._busy_until_block: dict[str, int] = {}
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ----
    def set_policy(self, policy: core.AuditPolicy) -> None:
        """Adopt a newly committed policy; windows already drafted keep their scheduled workload."""
        policy.validate(self.block_time_s)
        self.policy = policy


    def start(self) -> None:
        self._running = True
        threading.Thread(target=self._run, name="capacity-audit", daemon=True).start()
        log.info("capacity audit worker started for %d endpoints (auditor %s)", len(self.endpoints),
                 self.policy.auditor_hotkey[:12] or "unset")

    def stop(self) -> None:
        self._running = False
        self.publisher.stop()

    def join_audits(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        for t in list(self._threads):
            t.join(max(0.0, deadline - time.monotonic()))

    # ---- block feed ----

    def _run(self) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                log.exception("capacity audit block feed error")
            time.sleep(self.poll_s)

    def poll_once(self) -> None:
        """Process every block from the last seen one up to the current head, in order."""
        head, head_hash = self.chain.head()
        if head <= 0:
            return
        if self._last_block <= 0:
            self._last_block = head - 1
        for n in range(self._last_block + 1, head + 1):
            h = head_hash if n == head and head_hash else self.chain.block_hash(n)
            if h is None:
                return  # retry on the next poll
            with self._hash_cond:
                self._hashes[n] = h
                for old in [b for b in self._hashes if b < n - 255]:
                    del self._hashes[old]
                self._hash_cond.notify_all()
            self._last_block = n
            self.on_block(n, h)

    def _wait_hash(self, block_number: int, timeout_s: float) -> bytes | None:
        deadline = time.monotonic() + timeout_s
        with self._hash_cond:
            while self._running:
                h = self._hashes.get(block_number)
                if h is not None:
                    return h
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._hash_cond.wait(min(remaining, 0.5))
        return None

    # ---- selection ----

    def on_block(self, block_number: int, block_hash: bytes) -> None:
        window = core.open_window(self.netuid, block_number, block_hash, self.chain.epoch_blocks(), self.policy)
        if window is None:
            return
        endpoint_of = {self._slot(ep).slot_id: ep for ep in self.endpoints}
        busy = {slot_id for slot_id, ep in endpoint_of.items()
                if block_number <= self._busy_until_block.get(ep.endpoint_id, 0)}
        for slot, row in core.draft(window, [self._slot(ep) for ep in self.endpoints], self.policy, busy):
            ep = endpoint_of[slot.slot_id]
            audit_id, audit_block, proof_block = window.audit_id, window.start_block, window.proof_block
            audit = SelectedAudit(ep, slot, row, core.gpu_workload_spec(row), audit_id, block_number, audit_block,
                                  proof_block, core.lease_id(slot, window.epoch))
            self._busy_until_block[ep.endpoint_id] = proof_block + core.payload_cooldown_blocks(self.policy,
                                                                                                self.block_time_s)
            self.drain.set(ep.endpoint_id, time.time() + core.active_hold_seconds(self.policy, self.block_time_s))
            t = threading.Thread(target=self._run_audit, args=(audit,), name=f"audit-{audit_id[:8]}-{ep.endpoint_id}",
                                 daemon=True)
            self._threads.append(t)
            t.start()
            log.info("endpoint %s drafted: audit %s B_select=%d B_start=%d B_proof=%d", ep.endpoint_id, audit_id[:12],
                     block_number, audit_block, proof_block)

    def _slot(self, ep: AuditEndpoint) -> core.Slot:
        return core.Slot(self.netuid, self.keypair.ss58_address, ep.endpoint_id, ep.url, ep.model_id, ep.gpu_name,
                         ep.vram_gb)

    # ---- one audit ----

    def _artifact(self, audit: SelectedAudit, **fields: Any) -> dict:
        artifact = {
            "protocol_version": core.PROTOCOL_VERSION, "audit_id": audit.audit_id, "slot_id": audit.slot.slot_id,
            "netuid": self.netuid, "hotkey": self.keypair.ss58_address, "endpoint_id": audit.endpoint.endpoint_id,
            "claimed_gpu_class": audit.gpu_class.match_gpu_name, "gpu_index": 0, "B_select": audit.selection_block,
            "B_start": audit.audit_block, "B_proof": audit.proof_block, "pass_count": audit.passes, **fields,
        }
        artifact["miner_signature"] = core.sign_artifact(artifact, self.keypair)
        return artifact

    def _run_audit(self, audit: SelectedAudit) -> None:
        ep_id = audit.endpoint.endpoint_id
        proc = None
        try:
            proc = self._execute(audit)
        except Exception:
            log.exception("capacity audit %s failed", audit.audit_id[:12])
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
            self.drain.clear(ep_id)

    def _execute(self, audit: SelectedAudit) -> subprocess.Popen | None:
        p = self.policy
        lead_s = (audit.audit_block - audit.selection_block) * self.block_time_s
        challenge_wait_s = (audit.proof_block - audit.audit_block) * self.block_time_s + p.payload_deadline_s
        out_dir = Path(tempfile.mkdtemp(prefix="verathos_capacity_", dir=self.work_dir))
        files = {name: out_dir / f"{audit.lease}_{name}" for name in
                 ("challenge.txt", "start.json", "ready.json", "pass0.json", "final_timing.json", "final.json")}
        cmd = self.bench_command(audit, out_dir=out_dir, challenge_file=files["challenge.txt"],
                                 ready_file=files["ready.json"], start_file=files["start.json"],
                                 start_timeout_s=max(60.0, lead_s + 60.0),
                                 challenge_timeout_s=max(30.0, challenge_wait_s))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(audit.endpoint.cuda_device)}
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        start_hash = self._wait_hash(audit.audit_block, lead_s + 60.0)
        if start_hash is None:
            log.warning("audit %s: B_start %d not observed in time", audit.audit_id[:12], audit.audit_block)
            return proc
        if proc.poll() is not None:
            log.warning("audit %s: bench exited before B_start: %s", audit.audit_id[:12], proc.stderr.read()[-500:])
            return proc
        proof_seed = core.derive_proof_seed(start_hash, audit.slot.slot_id, 0)
        tmp = files["start.json"].with_suffix(".tmp")
        tmp.write_text(json.dumps({"seed_hex": proof_seed, "audit_id": audit.audit_id, "B_start": audit.audit_block,
                                   "t": time.time()}, sort_keys=True) + "\n")
        os.replace(tmp, files["start.json"])

        pass0_root = final_root = transcript = ""
        deadline = time.time() + max(120.0, audit.gpu_class.deadline_s + 90.0 + challenge_wait_s)
        while time.time() < deadline:
            if not pass0_root and files["pass0.json"].exists():
                root = json.loads(files["pass0.json"].read_text()).get("root")
                if root not in (None, "", []):
                    pass0_root = _root_hex(root)
                    self.publisher.publish(RECEIPT_PATH, self._artifact(
                        audit, artifact_type="capacity_audit_pass0_receipt", pass0_root=pass0_root,
                        pass0_transcript_commit=pass0_root), attempts=3, retry_delay_s=0.25)
            if files["final_timing.json"].exists():
                timing = json.loads(files["final_timing.json"].read_text())
                if not pass0_root and timing.get("pass0_root"):
                    pass0_root = _root_hex(timing["pass0_root"])
                    self.publisher.publish(RECEIPT_PATH, self._artifact(
                        audit, artifact_type="capacity_audit_pass0_receipt", pass0_root=pass0_root,
                        pass0_transcript_commit=pass0_root), attempts=3, retry_delay_s=0.25)
                final_root = _root_hex(timing.get("root") or [])
                transcript = str(timing.get("transcript_root") or "") or core.transcript_root([pass0_root, final_root])
                final = dict(artifact_type="capacity_audit_final_receipt", pass0_root=pass0_root, final_root=final_root,
                             final_transcript_commit=transcript, combined=final_commitment(timing))
                self.publisher.publish(RECEIPT_PATH, self._artifact(audit, **final), attempts=3, retry_delay_s=0.25)
                break
            if proc.poll() is not None:
                log.warning("audit %s: bench ended without final receipt: %s", audit.audit_id[:12],
                            proc.stderr.read()[-500:])
                return proc
            time.sleep(0.02)
        if not transcript:
            log.warning("audit %s: no final receipt within the deadline", audit.audit_id[:12])
            return proc

        proof_hash = self._wait_hash(audit.proof_block, challenge_wait_s)
        if proof_hash is None:
            log.warning("audit %s: B_proof %d not observed in time", audit.audit_id[:12], audit.proof_block)
            return proc
        challenge = core.derive_proof_challenge_seed(transcript, proof_hash, audit.lease, audit.slot.slot_id, 0)
        tmp = files["challenge.txt"].with_suffix(".tmp")
        tmp.write_text(challenge)
        os.replace(tmp, files["challenge.txt"])

        try:
            proc.communicate(timeout=p.payload_deadline_s + 30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        summary = json.loads(files["final.json"].read_text()) if files["final.json"].exists() else {}
        payload = summary.get("proof_payload")
        if not isinstance(payload, dict) or payload.get("format") != COMBINED_PROOF_FORMAT:
            log.warning("audit %s: bench produced no proof payload", audit.audit_id[:12])
            return proc
        sampled = (payload.get("capacity_proof") or {}).get("sampled") or {}
        try:
            sampled_index = int(sampled.get("pass_index"))
        except (TypeError, ValueError):
            sampled_index = 0
        self.publisher.publish(PROOF_PATH, self._artifact(
            audit, artifact_type="capacity_audit_proof_payload", sampled_pass_index=sampled_index,
            sampled_opening={"lease_id": audit.lease, "transcript_root": transcript, "pass0_root": pass0_root,
                             "final_root": final_root, "pass_index": sampled_index},
            sampled_pass_proof=payload), attempts=min(12, max(1, int(p.payload_deadline_s // 5))), retry_delay_s=5.0)
        log.info("audit %s: proof published", audit.audit_id[:12])
        return proc
