"""Capacity audit end to end on the CPU: miner workers run a bench that speaks the GPU bench's file protocol (backed by
the CPU reference bench) and publish through the ingest app to the auditor, which times receipts, verifies proofs,
confirms finality and gates endpoints. An honest miner passes; a late miner and a forging miner are gated."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from verathos.capacity import core
from verathos.capacity.auditor import INVALID, PASSED, TIMING_MISS, Auditor
from verathos.capacity.holds import HoldTracker
from verathos.capacity.ingest import create_ingest_app
from verathos.capacity.miner import AuditEndpoint, AuditWorker, DrainState
from verathos.capacity.verdicts import accepted_verdicts

REPO = Path(__file__).resolve().parents[2]
AUDITOR_KEY = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
BLOCK_S = 0.4
TEST_GPU = core.GpuClass("Test GPU", 8, matrix_dim=8, capacity_passes=2, capacity_rounds=2, capacity_block_size=4,
                         fp64_matrix_dim=4, fp64_passes=1, fp64_block_size=2, spot_checks=2)
POLICY = core.AuditPolicy(auditor_hotkey="5Auditor", mode="score_gate", epoch_blocks=360, windows_per_epoch=360,
                          draft_probability=1.0, group_stress_fraction=0.0, lead_blocks=15,
                          proof_challenge_delay_blocks=4, deadline_s=0.6, transport_grace_s=0.2, payload_deadline_s=4.0,
                          gpu_classes=(TEST_GPU,))

BENCH = r'''
import json, sys, time, pathlib
sys.path.insert(0, sys.argv[1])
from tests.capacity.cpu_bench import CpuBench
_, _, behaviour, out, lease, spec, challenge_file, start_file = sys.argv
out = pathlib.Path(out)
while not pathlib.Path(start_file).exists():
    time.sleep(0.005)
seed = json.loads(pathlib.Path(start_file).read_text())["seed_hex"]
bench = CpuBench(json.loads(spec), lease, seed)
if behaviour == "late":
    time.sleep(1.0)
(out / f"{lease}_final_timing.json").write_text(json.dumps(bench.final_timing))
while not pathlib.Path(challenge_file).exists():
    time.sleep(0.005)
challenge = pathlib.Path(challenge_file).read_text()
if behaviour == "forge":
    challenge = "00" * 32
(out / f"{lease}_final.json").write_text(json.dumps({"proof_payload": bench.prove(challenge)}))
'''


def block_hash(n: int) -> bytes:
    return hashlib.sha256(f"block-{n}".encode()).digest()


class Chain:
    def __init__(self, start: int) -> None:
        self.n, self.lock = start, threading.Lock()

    def tick(self) -> None:
        with self.lock:
            self.n += 1

    def head(self):
        with self.lock:
            return self.n, block_hash(self.n)

    def block_hash(self, n):
        with self.lock:
            return block_hash(n) if n <= self.n else None


    def finalized_block(self) -> int:
        with self.lock:
            return self.n - 2

    def auditor_url(self) -> str:
        return "http://auditor"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("audit")
    script = tmp / "bench.py"
    script.write_text(BENCH)
    chain = Chain(start=1000)
    auditor = Auditor(netuid=405, policy=POLICY, chain=chain, db_path=tmp / "auditor.sqlite", block_time_s=BLOCK_S)
    client = TestClient(create_ingest_app(auditor, AUDITOR_KEY))
    miners = {}
    for behaviour in ("honest", "late", "forge"):
        kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        url = f"http://{len(miners) + 10}.0.113.5:8091/e/gpu0"

        def bench_command(audit, *, out_dir, challenge_file, start_file, behaviour=behaviour, **_):
            return [sys.executable, str(script), str(REPO), behaviour, str(out_dir), audit.lease,
                    json.dumps(audit.workload_spec), str(challenge_file), str(start_file)]

        endpoint = AuditEndpoint("gpu0", url, "kev-4b", "Test GPU", 8, 0)
        worker = AuditWorker(netuid=405, keypair=kp, endpoints=[endpoint], policy=POLICY, chain=chain,
                             drain=DrainState(), client=client, bench_command=bench_command, poll_s=0.01,
                             block_time_s=BLOCK_S, work_dir=tmp)
        miners[behaviour] = (kp, worker, core.Slot(405, kp.ss58_address, "gpu0", url, "kev-4b", "Test GPU", 8))
    auditor.update_slots([slot for _, _, slot in miners.values()])
    holds = HoldTracker(netuid=405, policy=POLICY, chain=chain, block_time_s=BLOCK_S)
    holds.update_slots([slot for _, _, slot in miners.values()])
    drained_seen = set()

    # Everyone starts at this block; skip the start-up replay of earlier blocks.
    auditor._last_block = holds._last_block = chain.n - 1
    auditor.poll_once()
    holds.poll_once()
    for _, worker, _ in miners.values():
        worker._last_block = chain.n - 1
        worker.start()
    # One window is enough: every slot is drafted at the first block and busy until its payload cooldown ends.
    for _ in range(45):
        chain.tick()
        auditor.poll_once()
        holds.poll_once()
        drained_seen |= {b for b, (kp, _, _) in miners.items() if holds.drained(kp.ss58_address, "gpu0")}
        time.sleep(BLOCK_S)
    for _, worker, _ in miners.values():
        worker.stop()
        worker.join_audits(10)
    for _ in range(12):  # past B_proof + payload deadline, then finality
        chain.tick()
        auditor.poll_once()
        time.sleep(0.05)
    while chain.n < 3 * 360 + 1:  # into epoch 3, whose verdicts count the closed epoch 2
        chain.tick()
        auditor.poll_once()
    auditor.drained_seen_by_follower = drained_seen
    yield auditor, miners
    auditor.stop()


def outcomes(auditor: Auditor, hotkey: str) -> list[str]:
    rows = auditor._db.execute("SELECT outcome, reason FROM slots WHERE hotkey = ? ORDER BY rowid", (hotkey,))
    return [r["outcome"] for r in rows]


def test_honest_miner_passes(run) -> None:
    auditor, miners = run
    kp = miners["honest"][0]
    assert outcomes(auditor, kp.ss58_address) and set(outcomes(auditor, kp.ss58_address)) == {PASSED}
    assert (kp.ss58_address, "gpu0") not in auditor.gated(3)


def test_late_final_receipt_is_a_timing_miss(run) -> None:
    auditor, miners = run
    kp = miners["late"][0]
    rows = auditor._db.execute("SELECT outcome, reason FROM slots WHERE hotkey = ?", (kp.ss58_address,)).fetchall()
    assert [(r["outcome"], r["reason"]) for r in rows] == [(TIMING_MISS, "deadline_exceeded")]


def test_forged_proof_is_invalid_and_gated(run) -> None:
    auditor, miners = run
    kp = miners["forge"][0]
    assert INVALID in outcomes(auditor, kp.ss58_address)
    assert auditor.gated(3)[(kp.ss58_address, "gpu0")] == "invalid_proof"
    assert (kp.ss58_address, "gpu0") not in auditor.gated(2)  # the epoch it happened in is still open there


def test_windows_are_confirmed_before_counting(run) -> None:
    auditor, _ = run
    statuses = {r["chain_status"] for r in auditor._db.execute("SELECT chain_status FROM windows")}
    assert statuses == {"confirmed"}


def test_ingest_rejects_unknown_and_unsigned_artifacts(run) -> None:
    auditor, miners = run
    kp = miners["honest"][0]
    client = TestClient(create_ingest_app(auditor, AUDITOR_KEY))
    artifact = {"protocol_version": core.PROTOCOL_VERSION, "artifact_type": "capacity_audit_final_receipt",
                "audit_id": "ff" * 32, "slot_id": "ee" * 32, "hotkey": kp.ss58_address}
    unsigned = client.post("/capacity/v1/receipt", json=artifact)
    assert unsigned.status_code == 400 and unsigned.json()["error"] == "invalid miner_signature"
    artifact["miner_signature"] = core.sign_artifact(artifact, kp)
    unknown = client.post("/capacity/v1/receipt", json=artifact)
    assert unknown.status_code == 400 and unknown.json()["error"] == "unknown audit slot"
    wrong_path = client.post("/capacity/v1/proof", json=artifact,
                             headers={core.HDR_AUDIT_ID: "ff" * 32, core.HDR_SLOT_ID: "ee" * 32})
    assert wrong_path.status_code == 400 and wrong_path.json()["error"] == "unknown audit slot"


def test_follower_sees_the_same_holds(run) -> None:
    auditor, miners = run
    assert auditor.drained_seen_by_follower == set(miners)


def test_signed_verdicts_reach_followers(run) -> None:
    auditor, miners = run
    client = TestClient(create_ingest_app(auditor, AUDITOR_KEY))
    document = client.get("/capacity/v1/verdicts", params={"epoch": 3}).json()
    gated = accepted_verdicts(document, auditor_hotkey=AUDITOR_KEY.ss58_address, netuid=405, epoch=3)
    assert gated == auditor.gated(3) and (miners["forge"][0].ss58_address, "gpu0") in gated
    with pytest.raises(ValueError, match="not signed by the auditor"):
        accepted_verdicts(document, auditor_hotkey=miners["honest"][0].ss58_address, netuid=405, epoch=3)
    with pytest.raises(ValueError, match="another netuid or epoch"):
        accepted_verdicts(document, auditor_hotkey=AUDITOR_KEY.ss58_address, netuid=405, epoch=2)
    assert client.get("/capacity/v1/verdicts", params={"epoch": 4}).status_code == 400  # not started
