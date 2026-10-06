"""Capacity audit core: identity, selection, signing, and the stored reference vectors for the beacon and proof
derivations (``fixtures/capacity_vectors.json``) that the GPU bench and the verifier depend on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from bittensor_wallet import Keypair

from verathos.capacity import core

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "capacity_vectors.json").read_text())


def h(i: int) -> str:
    return hashlib.sha256(f"block-{i}".encode()).hexdigest()


def test_proof_derivations_match_reference_vectors() -> None:
    for block, slot_hex, gpu, expected in VECTORS["proof_seed"]:
        assert core.derive_proof_seed(block, slot_hex, gpu) == expected
    for root, block, lease, slot_hex, gpu, expected in VECTORS["challenge_seed"]:
        assert core.derive_proof_challenge_seed(root, block, lease, slot_hex, gpu) == expected
    for root, lease, gpu, passes, expected in VECTORS["sampled_pass"]:
        assert core.derive_sampled_pass_index(root, lease, gpu, passes) == expected
    for block, epoch, expected in VECTORS["audit_seed"]:
        assert core.derive_audit_seed(block, epoch) == expected
    roots, expected = VECTORS["transcript_root"]
    assert core.transcript_root(roots) == expected


def test_beacon_and_gpu_rows_match_reference_vectors() -> None:
    policy = core.AuditPolicy(windows_per_epoch=5, lead_blocks=5, proof_challenge_delay_blocks=4)
    for block, block_hash, triggered, fits in VECTORS["window"]:
        assert core.window_triggered(block, block_hash, 360, policy) == triggered
        assert core.window_fits_epoch(block, 360, policy) == fits
    for name, (vram, spec) in VECTORS["workload_specs"].items():
        assert core.gpu_workload_spec(core.match_gpu_class(name, vram, policy)) == spec
    for host, expected in VECTORS["registered_domain"].items():
        assert core.registered_domain_from_host(host) == expected


def slot(hotkey: str, endpoint_id: str = "gpu0", url: str = "http://203.0.113.5:8091/e/gpu0", model: str = "kev-4b") -> core.Slot:
    return core.Slot(405, hotkey, endpoint_id, url, model, "NVIDIA GeForce RTX 5090", 32)


def test_slot_and_lease_identity() -> None:
    a, b = slot("hkA"), slot("hkA", "gpu1")
    assert a.slot_id != b.slot_id != slot("hkB").slot_id
    assert a.slot_id == slot("hkA").slot_id
    assert core.lease_id(a, 1) != core.lease_id(a, 2)


def test_selection_is_reproducible_by_each_miner() -> None:
    policy = core.AuditPolicy(cohort_fraction=0.5, windows_per_epoch=5)
    slots = [slot(f"hk{i}", url=f"http://198.51.{i // 50}.{i % 50}:8091/e/gpu0") for i in range(200)]
    for i in range(30):
        seed = core.derive_audit_seed(h(i), 1)
        cohort = {s.slot_id for s in core.select_slots(slots, seed, policy)}
        assert cohort == {s.slot_id for s in slots if core.slot_selected(s, seed, policy)}


def test_co_hosted_slots_are_drafted_together() -> None:
    """Group stress: endpoints behind one IP share grouping tokens and are selected together about twice as often
    as independent endpoints, on top of the base draw. Over repeated windows this drafts co-hosted slots into the
    same window, which is what lets the sieve catch several endpoints on one machine."""
    policy = core.AuditPolicy(cohort_fraction=0.5, windows_per_epoch=5)
    co_a, co_b = slot("hkX", "gpu0", "http://203.0.113.9:8091/e/gpu0"), slot("hkY", "gpu0", "http://203.0.113.9:8092/e/gpu0")
    far_a, far_b = slot("hkP", "gpu0", "http://192.0.2.1:8091/e/gpu0"), slot("hkQ", "gpu0", "http://198.51.100.7:8091/e/gpu0")
    together_co = together_far = 0
    for i in range(4000):
        seed = core.derive_audit_seed(h(i), 7)
        together_co += core.slot_selected(co_a, seed, policy) and core.slot_selected(co_b, seed, policy)
        together_far += core.slot_selected(far_a, seed, policy) and core.slot_selected(far_b, seed, policy)
    assert together_co > 2 * together_far  # measured 849 vs 367 of 4000


def test_artifact_signing() -> None:
    kp = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    artifact = {"protocol_version": core.PROTOCOL_VERSION, "slot_id": "ab" * 32, "pass_count": 337}
    artifact["miner_signature"] = core.sign_artifact(artifact, kp)
    assert core.verify_artifact_signature(artifact, kp.ss58_address)
    assert not core.verify_artifact_signature({**artifact, "pass_count": 1}, kp.ss58_address)
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    assert not core.verify_artifact_signature(artifact, other.ss58_address)


def test_policy_validation() -> None:
    core.AuditPolicy().validate()
    with pytest.raises(ValueError):
        core.AuditPolicy(deadline_s=60).validate()
    with pytest.raises(ValueError):
        core.AuditPolicy(group_stress_fraction=0.9).validate()
