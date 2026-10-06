"""Capacity-audit miner worker against a simulated chain, a fake bench speaking the real bench's file protocol, and a
recording auditor endpoint."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import httpx
from bittensor_wallet import Keypair

from verathos import catalog
from verathos.capacity import core
from verathos.capacity.miner import AuditEndpoint, AuditWorker, DrainState
from verathos.capacity.proof.combined import COMBINED_PROOF_FORMAT

BLOCK_S = 0.05

FAKE_BENCH = r'''
import json, sys, time, pathlib
out, lease, challenge_file, start_file = sys.argv[1:5]
out = pathlib.Path(out)
while not pathlib.Path(start_file).exists():
    time.sleep(0.005)
seed = json.loads(pathlib.Path(start_file).read_text())["seed_hex"]
time.sleep(0.02)
(out / f"{lease}_final_timing.json").write_text(json.dumps({
    "transcript_root": "cc" * 32, "combined_transcript_root": "cc" * 32,
    "workload_version": "hot_capacity_combined", "pass_count": 337}))
while not pathlib.Path(challenge_file).exists():
    time.sleep(0.005)
challenge = pathlib.Path(challenge_file).read_text()
(out / f"{lease}_final.json").write_text(json.dumps({"proof_payload": {
    "format": "capacity_combined_proof_v1",
    "seed_seen": seed, "challenge_seen": challenge}}))
'''


def block_hash(n: int) -> bytes:
    return hashlib.sha256(f"block-{n}".encode()).digest()


class FakeChain:
    def __init__(self, start: int) -> None:
        self.n = start
        self.lock = threading.Lock()

    def tick(self) -> None:
        with self.lock:
            self.n += 1

    def head(self):
        with self.lock:
            return self.n, block_hash(self.n)

    def block_hash(self, n):
        with self.lock:
            return block_hash(n) if n <= self.n else None


    def auditor_url(self) -> str:
        return "http://auditor"


def test_drafted_endpoint_runs_audit_and_publishes_signed_artifacts(tmp_path: Path) -> None:
    script = tmp_path / "fake_bench.py"
    script.write_text(FAKE_BENCH)
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        artifact = json.loads(request.content)
        assert request.headers[core.HDR_SLOT_ID] == artifact["slot_id"]
        received.append({"path": request.url.path, **artifact})
        return httpx.Response(200, json={"ok": True})

    def bench_command(audit, *, out_dir, challenge_file, start_file, start_timeout_s, challenge_timeout_s):
        return [sys.executable, str(script), str(out_dir), audit.lease, str(challenge_file), str(start_file)]

    # Policy forcing every window to trigger and every slot to be drafted, on a fast clock.
    policy = core.AuditPolicy(auditor_hotkey="5Auditor", epoch_blocks=360, windows_per_epoch=360,
                              draft_probability=1.0, group_stress_fraction=0.0, lead_blocks=2,
                              proof_challenge_delay_blocks=3, deadline_s=0.05, transport_grace_s=0.05,
                              payload_deadline_s=2.0, gpu_classes=catalog.load().audit_policy().gpu_classes)
    kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    endpoints = [
        AuditEndpoint("gpu0", "http://203.0.113.5:8091/e/gpu0", "kev-4b", "NVIDIA GeForce RTX 5090", 32, 0),
        AuditEndpoint("gpu1", "http://203.0.113.5:8091/e/gpu1", "kev-4b", "Some Unknown GPU", 16, 1),
    ]
    class RecordingDrain(DrainState):
        drained_at: list[float] = []

        def set(self, endpoint_id: str, until_ts: float) -> None:
            self.drained_at.append(time.time())
            super().set(endpoint_id, until_ts)

    chain, drain = FakeChain(start=1000), RecordingDrain()
    worker = AuditWorker(netuid=405, keypair=kp, endpoints=endpoints, policy=policy, chain=chain, drain=drain,
                         client=httpx.Client(transport=httpx.MockTransport(handler)), bench_command=bench_command,
                         poll_s=0.005, block_time_s=BLOCK_S, work_dir=tmp_path)
    worker.start()
    deadline = time.time() + 30
    while time.time() < deadline and not any(r["artifact_type"] == "capacity_audit_proof_payload" for r in received):
        chain.tick()
        time.sleep(BLOCK_S)
    worker.stop()
    worker.join_audits(10)
    worker.publisher.wait_idle(5)

    first = [r for r in received if r["audit_id"] == received[0]["audit_id"]]
    assert [r["artifact_type"] for r in first] == ["capacity_audit_final_receipt", "capacity_audit_proof_payload"]
    assert {r["path"] for r in first} == {"/capacity/v1/receipt", "/capacity/v1/proof"}
    assert all(r["endpoint_id"] == "gpu0" for r in received), "uncalibrated endpoint must never be audited"
    for r in first:
        assert core.verify_artifact_signature({k: v for k, v in r.items() if k != "path"}, kp.ss58_address)

    final, proof = first
    slot = core.Slot(405, kp.ss58_address, "gpu0", endpoints[0].url, "kev-4b", "NVIDIA GeForce RTX 5090", 32)
    assert final["slot_id"] == slot.slot_id
    assert final["combined"]["format"] == COMBINED_PROOF_FORMAT
    assert final["combined"]["combined_transcript_root"] == "cc" * 32
    assert final["lease_id"] == proof["lease_id"] == core.lease_id(slot, final["B_select"] // 360)
    assert final["B_start_hash"] == block_hash(final["B_start"]).hex()
    assert proof["B_proof_hash"] == block_hash(final["B_proof"]).hex()
    payload = proof["sampled_pass_proof"]
    assert payload["seed_seen"] == core.derive_proof_seed(block_hash(final["B_start"]), slot.slot_id, 0)
    assert payload["challenge_seen"] == core.derive_proof_challenge_seed(
        "cc" * 32, block_hash(final["B_proof"]), proof["lease_id"], slot.slot_id, 0)
    assert drain.drained_at and not drain.drained("gpu0")  # drained during the timed work only


def test_drain_state_expires() -> None:
    drain = DrainState()
    drain.set("gpu0", time.time() + 0.05)
    assert drain.drained("gpu0")
    time.sleep(0.06)
    assert not drain.drained("gpu0")
