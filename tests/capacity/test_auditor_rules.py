"""Auditor and delivery rules that keep validator-side gaps from counting against miners."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time

import httpx

from verathos.capacity import core
from verathos.capacity.auditor import NEUTRAL, PENDING, Auditor
from verathos.capacity.miner import _Publisher

GPU = core.GpuClass("Test GPU", 8, capacity_passes=2)
POLICY = core.AuditPolicy(epoch_blocks=360, windows_per_epoch=360, draft_probability=1.0, group_stress_fraction=0.0,
                          gpu_classes=(GPU,))


class Chain:
    def __init__(self, n: int) -> None:
        self.n = n

    def head(self):
        return self.n, self.block_hash(self.n)

    def block_hash(self, n):
        return hashlib.sha256(f"block-{n}".encode()).digest() if n <= self.n else None


    def finalized_block(self) -> int:
        return self.n - 2


def auditor(tmp_path, n: int = 1000) -> tuple[Auditor, Chain]:
    chain = Chain(n)
    a = Auditor(netuid=405, policy=POLICY, chain=chain, db_path=tmp_path / "a.sqlite")
    a.update_slots([core.Slot(405, f"hk{i}", "gpu0", f"http://203.0.113.{i}:8091/e/gpu0", "kev-4b", "Test GPU", 8)
                    for i in range(3)])
    return a, chain


def windows(a: Auditor) -> list[tuple[str, str, str]]:
    return [(r["chain_status"], r["outcome"], r["reason"]) for r in a._db.execute(
        "SELECT w.chain_status, s.outcome, s.reason FROM slots s JOIN windows w USING (audit_id) ORDER BY s.rowid")]


def test_start_up_replay_records_running_windows_as_void(tmp_path) -> None:
    a, _ = auditor(tmp_path)
    a.poll_once()
    rows = windows(a)
    assert rows and all(status == "void" and outcome == NEUTRAL for status, outcome, _ in rows)
    assert {reason for *_, reason in rows} == {"auditor_late"}
    assert a._busy_slots(1000)  # the replayed windows still hold their slots, as on the miners


def test_policy_change_voids_windows_while_it_propagates(tmp_path) -> None:
    a, chain = auditor(tmp_path)
    a._last_block = chain.n - 1
    a.poll_once()
    assert windows(a) and {o for _, o, _ in windows(a)} == {PENDING}
    a.set_policy(dataclasses.replace(POLICY, deadline_s=20.0))
    a._db.execute("DELETE FROM slots")  # free the slots so the next window drafts them again
    chain.n += 1
    a.poll_once()
    assert windows(a) and set(windows(a)) == {("void", NEUTRAL, "policy_transition")}


def test_unknown_slot_stops_only_that_slots_delivery() -> None:
    delivered = []

    def handler(request: httpx.Request) -> httpx.Response:
        artifact = json.loads(request.content)
        if artifact["slot_id"] == "a":
            return httpx.Response(400, json={"error": "unknown audit slot"})
        delivered.append(artifact["slot_id"])
        return httpx.Response(200, json={"ok": True})

    class View:
        def auditor_url(self) -> str:
            return "http://auditor"

    publisher = _Publisher(View(), httpx.Client(transport=httpx.MockTransport(handler)))
    until = time.time() + 5
    for slot in ("a", "a", "b"):
        publisher.receipts.publish(core.RECEIPT_PATH, {"audit_id": "w1", "slot_id": slot}, until=until,
                                   retry_delay_s=0.01)
    assert publisher.wait_idle(5)
    publisher.stop()
    assert delivered == ["b"]


def test_renamed_endpoints_still_count_against_their_hotkey(tmp_path) -> None:
    from verathos.capacity.auditor import TIMING_MISS

    a, chain = auditor(tmp_path)
    a.policy = dataclasses.replace(POLICY, mode="score_gate")
    a._last_block = chain.n - 1
    for i, endpoint_id in enumerate(("old-a", "old-b")):  # one miss each, under names no longer listed
        a._db.execute("INSERT INTO windows (audit_id, epoch, selection_block, start_block, proof_block, "
                      "selection_hash, chain_status) VALUES (?, 1, 0, 0, 0, '', 'confirmed')", (f"w{i}",))
        a._db.execute("INSERT INTO slots (audit_id, slot_id, hotkey, endpoint_id, gpu_class, lease_id, workload_spec, "
                      "pass_count, deadline_s, outcome) VALUES (?, ?, 'hk0', ?, 'Test GPU', '', '{}', 2, 30, ?)",
                      (f"w{i}", f"s{i}", endpoint_id, TIMING_MISS))
    assert a.gated(2) == {("hk0", "gpu0"): "hotkey_timing"}


def test_restart_neutralises_slots_that_were_still_open(tmp_path) -> None:
    a, chain = auditor(tmp_path)
    a._last_block = chain.n - 1
    a.poll_once()
    assert {o for _, o, _ in windows(a)} == {PENDING}
    again = Auditor(netuid=405, policy=POLICY, chain=chain, db_path=tmp_path / "a.sqlite")
    assert {(o, r) for _, o, r in windows(again)} == {(NEUTRAL, "auditor_restarted")}


def test_policy_change_also_voids_windows_opened_just_before_it(tmp_path) -> None:
    a, chain = auditor(tmp_path)
    a._last_block = chain.n - 1
    a.poll_once()
    a.set_policy(dataclasses.replace(POLICY, deadline_s=20.0))
    assert set(windows(a)) == {("void", NEUTRAL, "policy_transition")}


def test_proof_uploads_must_be_signed_by_the_slot_hotkey(tmp_path) -> None:
    from bittensor_wallet import Keypair

    from verathos.capacity.auditor import TIMING_PASS
    from verathos.protocol import signing

    owner, other = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(2))
    a, chain = auditor(tmp_path)
    a.update_slots([core.Slot(405, owner.ss58_address, "gpu0", "http://203.0.113.9:8091/e/gpu0", "kev-4b",
                              "Test GPU", 8)])
    a._last_block = chain.n - 1
    a.poll_once()
    row = a._db.execute("SELECT audit_id, slot_id FROM slots").fetchone()
    a._db.execute("UPDATE slots SET outcome = ?", (TIMING_PASS,))
    a._db.execute("UPDATE windows SET proof_hash = 'aa', proof_observed_at = 0")
    claim = core.upload_claim(row["audit_id"], row["slot_id"])
    forged = signing.sign(other, core.UPLOAD_DOMAIN, claim)
    assert a.begin_upload(row["audit_id"], row["slot_id"], forged).status == 401
    signed = signing.sign(owner, core.UPLOAD_DOMAIN, claim)
    assert a.begin_upload(row["audit_id"], row["slot_id"], signed) is None
    assert a.begin_upload(row["audit_id"], row["slot_id"], signed).status == 409  # one upload at a time
    a.end_upload(row["audit_id"], row["slot_id"])
    assert a.begin_upload(row["audit_id"], row["slot_id"], signed) is None


def test_a_failed_manifest_fetch_keeps_the_last_slots() -> None:
    from verathos.protocol.manifest import Manifest
    from verathos.validator.capacity import Capacity
    from verathos.validator.discovery import DiscoveredMiner, Neuron

    class Sink:
        slots: list = []

        def update_slots(self, slots) -> None:
            self.slots = slots

    sink = Sink()
    cap = Capacity(netuid=405, policy=POLICY, auditor_hotkey="", chain=None, holds=sink)
    manifest = Manifest(netuid=405, mechid=1, hotkey="hkA", catalog_digest="0" * 64, endpoints=[{
        "endpoint_id": "gpu0", "url": "http://203.0.113.5:80/e/gpu0", "model_id": "kev-4b", "max_concurrency": 1,
        "hardware": {"kind": "cuda", "device_id": "GPU-a", "gpu_name": "Test GPU", "vram_gb": 8}}])
    miner = DiscoveredMiner(1, "hkA", "http://203.0.113.5:80", manifest)
    cap.update_slots([miner], [Neuron(1, "hkA", "203.0.113.5", 80, "c1")])
    cap.update_slots([], [Neuron(1, "hkA", "203.0.113.5", 80, "c1")])  # fetch failed, same commitment
    assert [s.hotkey for s in sink.slots] == ["hkA"]
    cap.update_slots([], [Neuron(1, "hkA", "203.0.113.5", 80, "c2")])  # new commitment not fetched: dropped
    assert sink.slots == []
