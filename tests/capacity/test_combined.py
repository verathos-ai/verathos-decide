"""Combined capacity proof end to end on the CPU: a reference implementation of the bench's lanes builds a complete
proof (capacity lane with its Q block sumcheck and Pallas-field GEMM proof, plus an FP64 lane), and the validator-side
verifier accepts it and rejects tampering."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tests.capacity.cpu_bench import CpuBench
from verathos.capacity.proof.combined import WORKLOAD_VERSION, final_commitment, verify_combined_proof

LEASE = "lease-combined"
PROOF_SEED = "21" * 32
B_PROOF = "43" * 32
SPEC = {
    "workload_version": WORKLOAD_VERSION,
    "capacity_matrix_dim": 8, "capacity_block_size": 4, "capacity_passes": 2, "capacity_rounds": 2,
    "capacity_warmup_passes": 1, "transition_mix_rounds": 0, "transition_fanout": 1,
    "capacity_tail_passes": 0, "capacity_tail_rounds": 1, "capacity_tail_warmup_passes": 1,
    "capacity_tail_transition_mix_rounds": 1, "capacity_tail_transition_fanout": 1,
    "fp64_matrix_dim": 4, "fp64_passes": 1, "fp64_rounds": 1, "fp64_warmup_passes": 1, "fp64_block_size": 2,
    "spot_checks": 2, "pass_count": 3,
}


@pytest.fixture(scope="module")
def audit() -> tuple[dict, dict]:
    bench = CpuBench(SPEC, LEASE, PROOF_SEED)
    return bench.prove(B_PROOF), final_commitment(bench.final_timing)


def verify(proof: dict, commit: dict, **overrides) -> tuple[bool, str]:
    args = {"workload_spec": SPEC, "lease_id": LEASE, "gpu_index": 0, "proof_seed_hex": PROOF_SEED,
            "b_proof_seed_hex": B_PROOF, **overrides}
    return verify_combined_proof(proof=proof, final_commit=commit, **args)


def test_honest_proof_verifies(audit) -> None:
    assert verify(*audit) == (True, "ok")


def test_proof_is_bound_to_the_chain_challenge_and_proof_seed(audit) -> None:
    proof, commit = audit
    assert verify(proof, commit, b_proof_seed_hex="44" * 32) == (False, "b_proof_seed_mismatch")
    assert not verify(proof, commit, proof_seed_hex="22" * 32)[0]
    assert not verify(proof, commit, lease_id="other-lease")[0]


def test_workload_must_match_the_validator_schedule(audit) -> None:
    proof, commit = audit
    smaller = {**SPEC, "capacity_passes": 1, "pass_count": 2}
    assert verify(proof, commit, workload_spec=smaller) == (False, "capacity_params_do_not_match_scheduled_workload")


def test_proof_must_match_the_signed_final_commitment(audit) -> None:
    proof, commit = audit
    assert verify(proof, {**commit, "capacity_transcript_root": "00" * 32}) == \
        (False, "final_capacity_transcript_root_mismatch")
    assert verify(proof, {}) == (False, "missing_final_combined_commitment")


def _flip_last_byte(hex_text: str) -> str:
    return hex_text[:-2] + f"{int(hex_text[-2:], 16) ^ 1:02x}"


@pytest.mark.parametrize("mutate, reason", [
    (lambda p: p["capacity_proof"]["block_proof"]["q_block_i32"][0].__setitem__(0, 7), "capacity_"),
    (lambda p: p["capacity_proof"]["gemm_proof"].__setitem__(
        "payload", _flip_last_byte(p["capacity_proof"]["gemm_proof"]["payload"])
     ), "capacity_gemm"),
    (lambda p: p["capacity_proof"].pop("gemm_band_openings"), "capacity_missing_gemm_proof"),
    (lambda p: p["fp64_proof"]["spots"][0].__setitem__("value", p["fp64_proof"]["spots"][0]["value"] + 2), "fp64_"),
    (lambda p: p["fp64_proof"].__setitem__("seed", p["fp64_proof"]["seed"] + 1), "fp64_seed"),
    (lambda p: p.__setitem__("sample_seed", "00" * 32), "sample_seed_mismatch"),
])
def test_tampering_is_rejected(audit, mutate, reason) -> None:
    proof, commit = audit
    tampered = deepcopy(proof)
    mutate(tampered)
    ok, why = verify(tampered, commit)
    assert not ok and why.startswith(reason), why
