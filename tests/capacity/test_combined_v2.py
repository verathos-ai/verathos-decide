"""Production-envelope tests for strict hot-capacity proof v2."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

from verathos.capacity.core import derive_sampled_pass_index
from verathos.capacity.proof.combined import (
    COMBINED_PROOF_FORMAT_V2,
    COMBINED_WORKLOAD_VERSION,
    _combined_transcript_root,
    _sample_seed_from_b_proof,
    combined_commitment_from_final_timing,
    combined_proof_protocol_version,
    verify_combined_proof_payload,
)
from verathos.capacity.proof.gemm_v2 import (
    capacity_gemm_v2_context_from_workspace_proof,
    prove_capacity_gemm_v2,
)
from verathos.capacity.proof.workspace import (
    WORKSPACE_PROOF_FORMAT,
    WorkspaceMerkleTree,
    derive_challenge_block_index,
    derive_sampled_round_index,
    pass_root_from_round_roots,
    serialize_merkle_path,
    workspace_i8_value_after_mix,
    workspace_seed_for,
)


def _matmul(a, b):
    return [
        [
            sum(int(a[i][k]) * int(b[k][j]) for k in range(len(b)))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def _combined_v2_fixture():
    lease_id = "combined-capacity-v2"
    gpu_index = 0
    proof_seed = "21" * 32
    b_proof_seed = "43" * 32
    matrix_dim = 4
    block_size = 2
    capacity_params = {
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": 1,
        "rounds": 1,
        "transition_mix_rounds": 0,
        "transition_fanout": 1,
    }
    fp64_params = {
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": 0,
        "rounds": 1,
    }
    capacity_transcript = "65" * 32
    fp64_transcript = "87" * 32
    combined_root = _combined_transcript_root(
        capacity_transcript=capacity_transcript,
        capacity_tail_transcript="",
        fp64_transcript=fp64_transcript,
        capacity_params=capacity_params,
        capacity_tail_params=None,
        fp64_params=fp64_params,
        proof_protocol_version=2,
    )
    sample_seed = _sample_seed_from_b_proof(
        b_proof_seed=b_proof_seed,
        combined_transcript=combined_root,
    )
    workspace_seed = workspace_seed_for(lease_id, gpu_index, proof_seed)
    pass_index = derive_sampled_pass_index(sample_seed, lease_id, gpu_index, 1)
    round_index = derive_sampled_round_index(
        sample_seed,
        lease_id,
        gpu_index,
        pass_index,
        1,
    )
    a = [
        [
            workspace_i8_value_after_mix(
                workspace_seed=workspace_seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=0,
                row=row,
                col=column,
                matrix_dim=matrix_dim,
                memory_mix_rounds=0,
            )
            for column in range(matrix_dim)
        ]
        for row in range(matrix_dim)
    ]
    b = [
        [
            workspace_i8_value_after_mix(
                workspace_seed=workspace_seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=1,
                row=row,
                col=column,
                matrix_dim=matrix_dim,
                memory_mix_rounds=0,
            )
            for column in range(matrix_dim)
        ]
        for row in range(matrix_dim)
    ]
    q = _matmul(a, b)
    q_tree = WorkspaceMerkleTree.from_matrix_i32(q, block_size)
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=q_tree.root.hex(),
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=sample_seed,
    )
    row_block, column_block = divmod(block_index, blocks_per_row)
    q_block = [
        row[column_block * block_size : (column_block + 1) * block_size]
        for row in q[row_block * block_size : (row_block + 1) * block_size]
    ]
    q_path = serialize_merkle_path(q_tree.get_path(row_block, column_block))
    lane_proof = {
        "format": WORKSPACE_PROOF_FORMAT,
        "workload_version": "fixed_workspace_merkle_gemm_v1",
        "memory_mix_rounds": 0,
        "transition_mix_rounds": 0,
        "transition_fanout": 1,
        "proof_challenge_seed": sample_seed,
        "proof_challenge_public": True,
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "rounds": 1,
        "passes": 1,
        "spot_checks": 1,
        "workspace_seed": workspace_seed,
        "root_chain": [
            {
                "pass_index": 0,
                "pass_root": pass_root_from_round_roots(0, [q_tree.root.hex()]),
                "round_roots": [q_tree.root.hex()],
            }
        ],
        "sampled": {
            "pass_index": pass_index,
            "round_index": round_index,
            "round_root": q_tree.root.hex(),
            "block_index": block_index,
        },
        "block_proof": {
            "bi": row_block,
            "bj": column_block,
            "q_block_i32": q_block,
            "merkle_path": q_path,
        },
    }
    context = capacity_gemm_v2_context_from_workspace_proof(
        proof=lane_proof,
        capacity_params=capacity_params,
        lease_id=lease_id,
        gpu_index=gpu_index,
        proof_seed_hex=proof_seed,
        challenge_seed_hex=sample_seed,
        transcript_root_hex=combined_root,
    )
    lane_proof["gemm_v2"] = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    ).to_mapping()
    lane_proof["gemm_v2_state_openings"] = []
    combined = {
        "format": COMBINED_PROOF_FORMAT_V2,
        "proof_protocol_version": 2,
        "workload_version": COMBINED_WORKLOAD_VERSION,
        "lease_id": lease_id,
        "gpu_index": gpu_index,
        "pass_count": 1,
        "capacity_params": capacity_params,
        "capacity_tail_params": None,
        "fp64_params": fp64_params,
        "capacity_transcript_root": capacity_transcript,
        "capacity_tail_transcript_root": "",
        "fp64_transcript_root": fp64_transcript,
        "combined_transcript_root": combined_root,
        "b_proof_seed": b_proof_seed,
        "b_proof_source": "challenge_file",
        "sample_seed": sample_seed,
        "capacity_proof": lane_proof,
        "capacity_tail_proof": None,
        "fp64_proof": {},
    }
    final_artifact = {
        "combined": {
            "format": COMBINED_PROOF_FORMAT_V2,
            "proof_protocol_version": 2,
            "combined_transcript_root": combined_root,
            "capacity_transcript_root": capacity_transcript,
            "capacity_tail_transcript_root": "",
            "fp64_transcript_root": fp64_transcript,
            "capacity_params": capacity_params,
            "capacity_tail_params": None,
            "fp64_params": fp64_params,
        }
    }
    verifier_args = {
        "final_artifact": final_artifact,
        "expected_combined_transcript_root": combined_root,
        "lease_id": lease_id,
        "gpu_index": gpu_index,
        "proof_seed_hex": proof_seed,
        "proof_challenge_seed_hex": b_proof_seed,
        "expected_workload_spec": {
            "workload_version": COMBINED_WORKLOAD_VERSION,
            "capacity_matrix_dim": matrix_dim,
            "capacity_block_size": block_size,
            "capacity_passes": 1,
            "capacity_rounds": 1,
            "transition_mix_rounds": 0,
            "transition_fanout": 1,
            "capacity_tail_passes": 0,
            "capacity_tail_rounds": 1,
            "capacity_tail_transition_mix_rounds": 1,
            "capacity_tail_transition_fanout": 1,
            "fp64_matrix_dim": matrix_dim,
            "fp64_block_size": block_size,
            "fp64_passes": 0,
            "fp64_rounds": 1,
            "spot_checks": 1,
            "pass_count": 1,
        },
    }
    return combined, verifier_args


def test_combined_v2_dispatches_through_strict_arithmetic_verifier():
    proof, verifier_args = _combined_v2_fixture()
    with patch(
        "verathos.capacity.proof.combined.verify_workspace_proof_payload",
        return_value=(True, "ok"),
    ):
        assert verify_combined_proof_payload(proof=proof, **verifier_args) == (
            True,
            "ok",
        )


def test_child_final_timing_normalizes_to_validator_commitment_schema():
    proof, verifier_args = _combined_v2_fixture()
    committed = dict(verifier_args["final_artifact"]["combined"])
    committed.update(
        {
            "workload_version": COMBINED_WORKLOAD_VERSION,
            "pass_count": 1,
            "workspace_mode": "sequential_reuse",
            "timed_cuda_component_s": 12.5,
            "timed_wall_s": 12.75,
        }
    )
    final_timing = dict(committed)
    final_timing["proof_format"] = final_timing.pop("format")

    assert combined_commitment_from_final_timing(final_timing) == committed


def test_combined_v2_malformed_or_tampered_payload_fails_closed():
    proof, verifier_args = _combined_v2_fixture()
    tampered = deepcopy(proof)
    encoded = tampered["capacity_proof"]["gemm_v2"]["payload"]
    tampered["capacity_proof"]["gemm_v2"]["payload"] = encoded[:-2] + (
        "00" if encoded[-2:] != "00" else "01"
    )
    with patch(
        "verathos.capacity.proof.combined.verify_workspace_proof_payload",
        return_value=(True, "ok"),
    ):
        ok, reason = verify_combined_proof_payload(
            proof=tampered,
            **verifier_args,
        )
    assert not ok
    assert "gemm_v2" in reason

    malformed = deepcopy(proof)
    malformed["capacity_proof"].pop("gemm_v2_state_openings")
    with patch(
        "verathos.capacity.proof.combined.verify_workspace_proof_payload",
        return_value=(True, "ok"),
    ):
        ok, reason = verify_combined_proof_payload(
            proof=malformed,
            **verifier_args,
        )
    assert (ok, reason) == (False, "capacity_missing_gemm_v2_state_openings")


def test_combined_protocol_version_cannot_be_relabelled():
    proof, _verifier_args = _combined_v2_fixture()
    assert combined_proof_protocol_version(proof) == 2
    proof["proof_protocol_version"] = 1
    assert combined_proof_protocol_version(proof) is None


def test_combined_v2_requires_pre_challenge_final_commitment():
    proof, verifier_args = _combined_v2_fixture()
    verifier_args["final_artifact"] = {}

    assert verify_combined_proof_payload(
        proof=proof,
        **verifier_args,
    ) == (False, "missing_final_combined_commitment")


def test_combined_v2_cannot_choose_smaller_than_scheduled_workload():
    proof, verifier_args = _combined_v2_fixture()
    verifier_args["expected_workload_spec"] = dict(
        verifier_args["expected_workload_spec"],
        capacity_matrix_dim=8,
    )
    with patch(
        "verathos.capacity.proof.combined.verify_workspace_proof_payload",
        return_value=(True, "ok"),
    ):
        assert verify_combined_proof_payload(
            proof=proof,
            **verifier_args,
        ) == (False, "capacity_params_do_not_match_scheduled_workload")


def test_combined_v2_passes_exact_scheduled_spot_count_to_workspace_verifier():
    proof, verifier_args = _combined_v2_fixture()
    with patch(
        "verathos.capacity.proof.combined.verify_workspace_proof_payload",
        return_value=(True, "ok"),
    ) as workspace_verify:
        assert verify_combined_proof_payload(
            proof=proof,
            **verifier_args,
        ) == (True, "ok")
    assert workspace_verify.call_args.kwargs["expected_spot_checks"] == 1
    assert workspace_verify.call_args.kwargs["proof_protocol_version"] == 2
