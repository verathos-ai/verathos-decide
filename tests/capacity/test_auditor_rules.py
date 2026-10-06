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
POLICY = core.AuditPolicy(windows_per_epoch=360, cohort_fraction=360.0, max_drain_fraction=1.0,
                          group_stress_fraction=0.0, gpu_classes=(GPU,))


class Chain:
    def __init__(self, n: int) -> None:
        self.n = n

    def head(self):
        return self.n, self.block_hash(self.n)

    def block_hash(self, n):
        return hashlib.sha256(f"block-{n}".encode()).digest() if n <= self.n else None

    def epoch_blocks(self) -> int:
        return 360

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
