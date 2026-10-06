"""Protocol-v2 hot-capacity arithmetic adapter tests."""

from __future__ import annotations

import base64
from dataclasses import replace

import numpy as np
import pytest

from verathos.capacity.core import derive_sampled_pass_index
from verathos.capacity.v2 import (
    CAPACITY_GEMM_V2_FORMAT,
    CapacityGemmV2Context,
    CapacityGemmV2FormatError,
    CapacityGemmV2Payload,
    prove_capacity_gemm_v2,
    required_transition_state_blocks,
    verify_capacity_gemm_v2,
)
from verathos.capacity.workspace_proof import (
    WORKSPACE_PROOF_FORMAT,
    WorkspaceMerkleTree,
    WorkspaceStateMerkleTree,
    derive_challenge_block_index,
    derive_sampled_round_index,
    serialize_merkle_path,
    workspace_i8_value_after_mix,
    workspace_i8_values_after_mix,
    workspace_q_block_hash_i32,
    workspace_seed_for,
    workspace_state_block_hash_i8,
    verify_workspace_gemm_block_proof,
    verify_workspace_proof_payload,
    workspace_transcript_root_v2,
)
from verathos.zk.crypto.gemm_v2_batch import GemmV2BatchProof, GemmV2BatchStatement
from verathos.zk.crypto.pcs_v2 import PCSNativeError


@pytest.mark.parametrize("matrix_id", [0, 1])
@pytest.mark.parametrize("memory_mix_rounds", [0, 1, 3])
def test_vectorized_workspace_values_match_scalar_reference(
    matrix_id, memory_mix_rounds
):
    matrix_dim = 17
    indices = np.array(
        [[0, 1, 16], [17, 117, matrix_dim * matrix_dim - 1]],
        dtype=np.uint64,
    )
    actual = workspace_i8_values_after_mix(
        workspace_seed=0xFEDCBA9876543210,
        pass_index=123,
        round_index=7,
        matrix_id=matrix_id,
        element_indices=indices,
        memory_mix_rounds=memory_mix_rounds,
    )
    expected = np.array(
        [
            [
                workspace_i8_value_after_mix(
                    workspace_seed=0xFEDCBA9876543210,
                    pass_index=123,
                    round_index=7,
                    matrix_id=matrix_id,
                    row=int(index) // matrix_dim,
                    col=int(index) % matrix_dim,
                    matrix_dim=matrix_dim,
                    memory_mix_rounds=memory_mix_rounds,
                )
                for index in row
            ]
            for row in indices
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(actual, expected)


def _matmul(a, b):
    return [
        [
            sum(int(a[i][k]) * int(b[k][j]) for k in range(len(b)))
            for j in range(len(b[0]))
        ]
        for i in range(len(a))
    ]


def _deterministic_fixture(
    *,
    false_output: bool = False,
    memory_mix_rounds: int = 0,
    matrix_dim: int = 4,
):
    lease_id = "capacity-lease-v2"
    proof_seed = bytes.fromhex("12" * 32)
    challenge_seed = bytes.fromhex("34" * 32)
    transcript_root = bytes.fromhex("56" * 32)
    gpu_index = 0
    block_size = 2
    pass_count = 3
    rounds_per_pass = 2
    workspace_seed = workspace_seed_for(lease_id, gpu_index, proof_seed.hex())
    pass_index = derive_sampled_pass_index(
        challenge_seed.hex(), lease_id, gpu_index, pass_count
    )
    round_index = derive_sampled_round_index(
        challenge_seed.hex(),
        lease_id,
        gpu_index,
        pass_index,
        rounds_per_pass,
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
                memory_mix_rounds=memory_mix_rounds,
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
                memory_mix_rounds=memory_mix_rounds,
            )
            for column in range(matrix_dim)
        ]
        for row in range(matrix_dim)
    ]
    q = _matmul(a, b)
    if false_output:
        q = [[value + 1 for value in row] for row in q]
    q_tree = WorkspaceMerkleTree.from_matrix_i32(q, block_size)
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=q_tree.root.hex(),
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=challenge_seed.hex(),
    )
    row_block, column_block = divmod(block_index, blocks_per_row)
    context = CapacityGemmV2Context(
        workload_version="fixed_workspace_merkle_gemm_v2",
        lease_id=lease_id,
        gpu_index=gpu_index,
        proof_seed=proof_seed,
        challenge_seed=challenge_seed,
        transcript_root=transcript_root,
        round_root=q_tree.root,
        workspace_seed=workspace_seed,
        pass_count=pass_count,
        rounds_per_pass=rounds_per_pass,
        pass_index=pass_index,
        round_index=round_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        row_block=row_block,
        column_block=column_block,
        memory_mix_rounds=memory_mix_rounds,
    )
    q_block = [
        row[column_block * block_size : (column_block + 1) * block_size]
        for row in q[row_block * block_size : (row_block + 1) * block_size]
    ]
    q_path = serialize_merkle_path(q_tree.get_path(row_block, column_block))
    return context, q_block, q_path


def _state_opening(a, b, tree, block_index, block_size):
    blocks_per_row = len(a) // block_size
    block_row, block_column = divmod(block_index, blocks_per_row)
    a_block = np.asarray(
        [
            row[block_column * block_size : (block_column + 1) * block_size]
            for row in a[block_row * block_size : (block_row + 1) * block_size]
        ],
        dtype=np.int8,
    )
    b_block = np.asarray(
        [
            row[block_column * block_size : (block_column + 1) * block_size]
            for row in b[block_row * block_size : (block_row + 1) * block_size]
        ],
        dtype=np.int8,
    )
    packed = np.empty((block_size * 2, block_size), dtype=np.uint8)
    packed[0::2, :] = a_block.view(np.uint8)
    packed[1::2, :] = b_block.view(np.uint8)
    return {
        "block_index": block_index,
        "state_block_i8": "base64:"
        + base64.b64encode(packed.tobytes()).decode("ascii"),
        "merkle_path": serialize_merkle_path(tree.get_path(block_index)),
    }


def test_workspace_state_block_hash_accepts_int8_arrays():
    a_block = np.asarray([[-128, -1], [0, 127]], dtype=np.int8)
    b_block = np.asarray([[127, 0], [-1, -128]], dtype=np.int8)
    assert workspace_state_block_hash_i8(
        a_block,
        b_block,
    ) == workspace_state_block_hash_i8(
        a_block.tolist(),
        b_block.tolist(),
    )


def _transition_fixture():
    lease_id = "capacity-transition-v2"
    proof_seed = bytes.fromhex("9a" * 32)
    challenge_seed = bytes.fromhex("bc" * 32)
    transcript_root = bytes.fromhex("de" * 32)
    gpu_index = 1
    matrix_dim = 4
    block_size = 2
    pass_count = 2
    rounds_per_pass = 2
    workspace_seed = workspace_seed_for(lease_id, gpu_index, proof_seed.hex())
    pass_index = derive_sampled_pass_index(
        challenge_seed.hex(), lease_id, gpu_index, pass_count
    )
    round_index = derive_sampled_round_index(
        challenge_seed.hex(),
        lease_id,
        gpu_index,
        pass_index,
        rounds_per_pass,
    )
    a = [
        [-7, 2, 5, 9],
        [3, -4, 8, 1],
        [6, 0, -3, 2],
        [4, 7, 1, -5],
    ]
    b = [
        [2, -1, 3, 5],
        [7, 4, -2, 1],
        [-3, 6, 8, 0],
        [5, 2, 1, -4],
    ]
    q = _matmul(a, b)
    q_tree = WorkspaceMerkleTree.from_matrix_i32(q, block_size)
    state_tree = WorkspaceStateMerkleTree.from_state_i8_matrices(a, b, block_size)
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=q_tree.root.hex(),
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=challenge_seed.hex(),
    )
    row_block, column_block = divmod(block_index, blocks_per_row)
    context = CapacityGemmV2Context(
        workload_version="fixed_workspace_transition_merkle_gemm_v2",
        lease_id=lease_id,
        gpu_index=gpu_index,
        proof_seed=proof_seed,
        challenge_seed=challenge_seed,
        transcript_root=transcript_root,
        round_root=q_tree.root,
        workspace_seed=workspace_seed,
        pass_count=pass_count,
        rounds_per_pass=rounds_per_pass,
        pass_index=pass_index,
        round_index=round_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        row_block=row_block,
        column_block=column_block,
        transition_mix_rounds=1,
        transition_fanout=8,
        state_root=state_tree.root,
    )
    q_block = [
        row[column_block * block_size : (column_block + 1) * block_size]
        for row in q[row_block * block_size : (row_block + 1) * block_size]
    ]
    q_path = serialize_merkle_path(q_tree.get_path(row_block, column_block))
    openings = [
        _state_opening(a, b, state_tree, index, block_size)
        for index in required_transition_state_blocks(context)
    ]
    return context, q_block, q_path, openings


@pytest.mark.parametrize("memory_mix_rounds", [0, 2])
def test_deterministic_workspace_v2_roundtrip(memory_mix_rounds):
    context, q_block, q_path = _deterministic_fixture(
        memory_mix_rounds=memory_mix_rounds
    )
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    decoded = CapacityGemmV2Payload.from_mapping(payload.to_mapping())
    ok, reason = verify_capacity_gemm_v2(
        context=context,
        payload=decoded,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    assert (ok, reason) == (True, "ok")


def test_non_power_of_two_inner_dimension_is_zero_padded_soundly():
    context, q_block, q_path = _deterministic_fixture(matrix_dim=6)
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    assert verify_capacity_gemm_v2(
        context=context,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    ) == (True, "ok")


def test_transition_workspace_uses_exact_authenticated_band_openings():
    context, q_block, q_path, openings = _transition_fixture()
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
        state_openings=openings,
    )
    assert verify_capacity_gemm_v2(
        context=context,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
        state_openings=openings,
    ) == (True, "ok")

    ok, reason = verify_capacity_gemm_v2(
        context=context,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
        state_openings=openings[:-1],
    )
    assert not ok
    assert "exact required block set" in reason


def test_v2_workspace_transcript_commits_every_transition_state_root():
    q_roots = [
        {
            "pass_index": 0,
            "pass_root": "11" * 32,
            "round_roots": ["22" * 32],
        }
    ]
    state_roots = [
        {
            "pass_index": 0,
            "round_state_roots": [["33" * 32, "44" * 32]],
        }
    ]
    committed = workspace_transcript_root_v2(q_roots, state_roots)
    changed = [
        {
            "pass_index": 0,
            "round_state_roots": [["33" * 32, "45" * 32]],
        }
    ]
    assert workspace_transcript_root_v2(q_roots, changed) != committed


def test_valid_output_commitment_with_false_gemm_is_rejected():
    context, q_block, q_path = _deterministic_fixture(false_output=True)
    with pytest.raises(PCSNativeError, match="does not satisfy"):
        prove_capacity_gemm_v2(
            context=context,
            q_block_i32=q_block,
            q_merkle_path=q_path,
        )


def test_prover_supplied_terminal_values_cannot_replace_openings():
    context, q_block, q_path = _deterministic_fixture()
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    proof = GemmV2BatchProof.from_canonical_bytes(
        payload.arithmetic_proof,
        expected_blocks=1,
        expected_rounds=context.arithmetic_rounds,
    )
    forged = replace(
        proof,
        final_x_evals=((proof.final_x_evals[0] + 1) % (1 << 200),),
    )
    forged_payload = CapacityGemmV2Payload(
        payload.statement_digest, forged.canonical_bytes()
    )
    ok, _reason = verify_capacity_gemm_v2(
        context=context,
        payload=forged_payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    assert not ok


def test_empty_sumcheck_rounds_are_rejected():
    context, q_block, q_path = _deterministic_fixture()
    statement = context.statement(workspace_q_block_hash_i32(q_block))
    empty = GemmV2BatchProof(
        claimed_y_evals=(0,),
        rounds=(),
        final_x_evals=(0,),
        final_w_evals=(0,),
    )
    payload = CapacityGemmV2Payload(
        GemmV2BatchStatement((statement,)).digest(),
        empty.canonical_bytes(),
    )
    ok, reason = verify_capacity_gemm_v2(
        context=context,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    assert not ok
    assert "round count" in reason


def test_context_requires_fixed_nonzero_challenge_set_and_derived_coordinates():
    context, _q_block, _q_path = _deterministic_fixture()
    with pytest.raises(CapacityGemmV2FormatError, match="challenge_count"):
        replace(context, challenge_count=0)
    with pytest.raises(CapacityGemmV2FormatError, match="block coordinates"):
        replace(
            context,
            column_block=(context.column_block + 1)
            % (context.matrix_dim // context.block_size),
        )
    with pytest.raises(CapacityGemmV2FormatError, match="rounds_per_pass"):
        replace(context, rounds_per_pass=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload_version", "different_workspace_v2"),
        ("transcript_root", bytes.fromhex("aa" * 32)),
        ("memory_mix_rounds", 1),
    ],
)
def test_statement_context_is_not_replayable(field, value):
    context, q_block, q_path = _deterministic_fixture()
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    changed = replace(context, **{field: value})
    ok, reason = verify_capacity_gemm_v2(
        context=changed,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    assert not ok
    assert "statement digest" in reason


def test_legacy_or_ambiguous_mapping_is_not_protocol_v2():
    with pytest.raises(CapacityGemmV2FormatError):
        CapacityGemmV2Payload.from_mapping(
            {
                "format": "fixed_workspace_merkle_gemm_v1",
                "protocol_version": 1,
                "payload": "00",
            }
        )
    with pytest.raises(CapacityGemmV2FormatError):
        CapacityGemmV2Payload.from_mapping(
            {
                "format": CAPACITY_GEMM_V2_FORMAT,
                "protocol_version": 2,
                "payload": "00",
                "rounds": 0,
            }
        )


def test_q_opening_and_parser_fail_closed():
    context, q_block, q_path = _deterministic_fixture()
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=q_block,
        q_merkle_path=q_path,
    )
    bad_path = {
        "leaf_index": q_path["leaf_index"],
        "siblings": [list(item) for item in q_path["siblings"]],
    }
    bad_path["siblings"][0][1] = not bad_path["siblings"][0][1]
    ok, reason = verify_capacity_gemm_v2(
        context=context,
        payload=payload,
        q_block_i32=q_block,
        q_merkle_path=bad_path,
    )
    assert not ok
    assert "direction" in reason

    encoded = payload.canonical_bytes()
    with pytest.raises(CapacityGemmV2FormatError):
        CapacityGemmV2Payload.from_canonical_bytes(encoded + b"\x00")


def test_reachable_workspace_parser_rejects_zero_spots_and_malformed_data():
    lease_id = "legacy-parser-check"
    proof_seed = "42" * 32
    workspace_seed = workspace_seed_for(lease_id, 0, proof_seed)
    base = {
        "format": WORKSPACE_PROOF_FORMAT,
        "matrix_dim": 4,
        "block_size": 2,
        "rounds": 1,
        "spot_checks": 0,
        "workspace_seed": workspace_seed,
        "root_chain": [],
    }
    ok, reason = verify_workspace_proof_payload(
        proof=base,
        expected_transcript_root="00" * 32,
        expected_pass0_root="00" * 32,
        expected_final_root="00" * 32,
        lease_id=lease_id,
        gpu_index=0,
        proof_seed_hex=proof_seed,
        pass_count=1,
    )
    assert (ok, reason) == (False, "invalid_workspace_proof_parameters")

    malformed = dict(base, spot_checks=1, root_chain=[None])
    ok, reason = verify_workspace_proof_payload(
        proof=malformed,
        expected_transcript_root="00" * 32,
        expected_pass0_root="00" * 32,
        expected_final_root="00" * 32,
        lease_id=lease_id,
        gpu_index=0,
        proof_seed_hex=proof_seed,
        pass_count=1,
    )
    assert (ok, reason) == (False, "malformed_workspace_proof")


def test_reachable_workspace_parser_rejects_unscheduled_spot_count():
    lease_id = "scheduled-spot-checks"
    proof_seed = "24" * 32
    proof = {
        "format": WORKSPACE_PROOF_FORMAT,
        "matrix_dim": 4,
        "block_size": 2,
        "rounds": 1,
        "spot_checks": 1,
        "workspace_seed": workspace_seed_for(lease_id, 0, proof_seed),
        "root_chain": [],
    }
    ok, reason = verify_workspace_proof_payload(
        proof=proof,
        expected_transcript_root="00" * 32,
        expected_pass0_root="00" * 32,
        expected_final_root="00" * 32,
        lease_id=lease_id,
        gpu_index=0,
        proof_seed_hex=proof_seed,
        pass_count=1,
        expected_spot_checks=2,
    )
    assert (ok, reason) == (
        False,
        "spot_check_count_does_not_match_scheduled_workload",
    )


def test_reachable_workspace_parser_requires_exact_nonzero_arithmetic_rounds():
    q = [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    tree = WorkspaceMerkleTree.from_matrix_i32(q, 2)
    q_block = [[1, 2], [3, 4]]
    block_proof = {
        "bi": 0,
        "bj": 0,
        "leaf_hash": workspace_q_block_hash_i32(q_block).hex(),
        "merkle_path": serialize_merkle_path(tree.get_path(0, 0)),
        "m_bits": 1,
        "n_bits": 1,
        "k_bits": 0,
        "q_block_i32": q_block,
        "sumcheck_proof": {
            "claimed_sum": 0,
            "rounds": [],
            "final_A": 0,
            "final_B": 0,
        },
        "spot_A": [{"row": 0, "col": 0, "value": 0}],
        "spot_B": [{"row": 0, "col": 0, "value": 0}],
    }
    ok, reason = verify_workspace_gemm_block_proof(
        block_proof_data=block_proof,
        round_root_hex=tree.root.hex(),
        workspace_seed=1,
        pass_index=0,
        round_index=0,
        matrix_dim=4,
        block_size=2,
        spot_checks=1,
        expected_block_index=0,
    )
    assert (ok, reason) == (False, "gemm_dimension_or_round_count_mismatch")

    block_proof["k_bits"] = 2
    block_proof["sumcheck_proof"]["rounds"] = [[0, 0, 0], [0, 0, 0]]
    ok, reason = verify_workspace_gemm_block_proof(
        block_proof_data=block_proof,
        round_root_hex=tree.root.hex(),
        workspace_seed=1,
        pass_index=0,
        round_index=0,
        matrix_dim=4,
        block_size=2,
        spot_checks=1,
        expected_block_index=1,
    )
    assert (ok, reason) == (False, "gemm_block_index_mismatch")
