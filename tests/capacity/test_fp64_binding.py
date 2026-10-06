"""Strict parent binding tests for the FP64 hot-capacity child proof."""

from __future__ import annotations

from copy import deepcopy

import pytest

from verathos.capacity.proof.combined import _verify_fp64_identity_proof
from verathos.capacity.bench import bench_fp64_identity as fp64


def _valid_child_case() -> tuple[dict, dict, dict]:
    lease_id = "fp64-v2-parent-binding"
    gpu_index = 0
    proof_seed = "19" * 32
    challenge_seed = "ab" * 32
    matrix_dim = 4
    block_size = 2
    passes = 1
    rounds = 1
    spot_checks = 2
    seed = fp64.seed_for(lease_id, gpu_index, proof_seed)
    c_matrix = [
        [
            fp64.fp64_dot_value(
                seed=seed,
                pass_index=0,
                round_index=0,
                row=row,
                col=column,
                matrix_dim=matrix_dim,
            )
            for column in range(matrix_dim)
        ]
        for row in range(matrix_dim)
    ]
    blocks_per_row = matrix_dim // block_size
    blocks = []
    for block_row in range(blocks_per_row):
        for block_column in range(blocks_per_row):
            blocks.append(
                [
                    row[
                        block_column * block_size : (block_column + 1) * block_size
                    ]
                    for row in c_matrix[
                        block_row * block_size : (block_row + 1) * block_size
                    ]
                ]
            )
    levels = [
        [fp64.fp64_leaf_hash(fp64.fp64_c_block_hash(block)) for block in blocks]
    ]
    while len(levels[-1]) > 1:
        current = levels[-1]
        levels.append(
            [
                fp64.fp64_parent_hash(
                    current[index],
                    current[index + 1] if index + 1 < len(current) else current[index],
                )
                for index in range(0, len(current), 2)
            ]
        )
    root = levels[-1][0]
    root_chain = [{"pass_index": 0, "round_roots": [root.hex()]}]
    transcript_root = fp64.transcript_root(
        seed=seed,
        matrix_dim=matrix_dim,
        block_size=block_size,
        passes=passes,
        rounds=rounds,
        root_chain=root_chain,
    )
    sample = fp64.derive_sample(
        challenge_hex=challenge_seed,
        transcript_hex=transcript_root,
        passes=passes,
        rounds=rounds,
        total_blocks=len(blocks),
        block_size=block_size,
        spot_checks=spot_checks,
    )
    block_index = int(sample["block_index"])
    block_row, block_column = divmod(block_index, blocks_per_row)
    c_block = blocks[block_index]
    path_siblings = []
    path_index = block_index
    for level in levels[:-1]:
        if path_index & 1:
            sibling_index = path_index - 1
            is_left = True
        else:
            sibling_index = path_index + 1
            is_left = False
            if sibling_index >= len(level):
                sibling_index = path_index
        path_siblings.append([level[sibling_index].hex(), is_left])
        path_index //= 2
    leaf_hash = levels[0][block_index]
    spots = []
    for offset in sample["spot_offsets"]:
        local_row, local_column = divmod(int(offset), block_size)
        row = block_row * block_size + local_row
        column = block_column * block_size + local_column
        spots.append(
            {
                "offset": int(offset),
                "row": row,
                "col": column,
                "value": c_block[local_row][local_column],
            }
        )
    proof = {
        "format": fp64.PROOF_FORMAT,
        "workload_version": fp64.WORKLOAD_VERSION,
        "seed": seed,
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": passes,
        "rounds": rounds,
        "root_chain": root_chain,
        "proof_challenge_seed": challenge_seed,
        "transcript_root": transcript_root,
        "sampled": {
            "pass_index": 0,
            "round_index": 0,
            "block_index": block_index,
            "bi": block_row,
            "bj": block_column,
        },
        "c_block_i64": c_block,
        "leaf_hash": leaf_hash.hex(),
        "merkle_path": {
            "leaf_index": block_index,
            "siblings": path_siblings,
        },
        "spots": spots,
    }
    params = {
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": passes,
        "rounds": rounds,
    }
    verifier_args = {
        "fp64_params": params,
        "expected_transcript_root": transcript_root,
        "expected_challenge_seed": challenge_seed,
        "expected_spot_checks": spot_checks,
        "lease_id": lease_id,
        "gpu_index": gpu_index,
        "proof_seed_hex": proof_seed,
    }
    return proof, params, verifier_args


def test_fp64_v2_child_is_bound_to_parent_context():
    proof, _params, verifier_args = _valid_child_case()
    assert _verify_fp64_identity_proof(proof, **verifier_args) == (True, "ok")


@pytest.mark.parametrize(
    ("field_name", "replacement", "reason"),
    (
        ("seed", 7, "seed_does_not_match_scheduled_workload"),
        ("matrix_dim", 8, "matrix_dim_does_not_match_scheduled_workload"),
        ("block_size", 4, "block_size_does_not_match_scheduled_workload"),
        ("passes", 2, "passes_does_not_match_scheduled_workload"),
        ("rounds", 2, "rounds_does_not_match_scheduled_workload"),
        ("proof_challenge_seed", "cd" * 32, "proof_challenge_seed_mismatch"),
        ("transcript_root", "ef" * 32, "transcript_root_parent_mismatch"),
    ),
)
def test_fp64_v2_child_cannot_substitute_parent_fields(
    field_name: str,
    replacement: object,
    reason: str,
):
    proof, _params, verifier_args = _valid_child_case()
    tampered = deepcopy(proof)
    tampered[field_name] = replacement
    assert _verify_fp64_identity_proof(tampered, **verifier_args) == (
        False,
        reason,
    )


def test_fp64_v2_child_requires_exact_scheduled_spot_set():
    proof, _params, verifier_args = _valid_child_case()
    tampered = deepcopy(proof)
    tampered["spots"] = tampered["spots"][:1]
    assert _verify_fp64_identity_proof(tampered, **verifier_args) == (
        False,
        "spot_check_count_does_not_match_scheduled_workload",
    )

    tampered = deepcopy(proof)
    tampered["spots"][0]["offset"] = (
        int(tampered["spots"][0]["offset"]) + 1
    ) % (int(tampered["block_size"]) ** 2)
    ok, reason = _verify_fp64_identity_proof(tampered, **verifier_args)
    assert not ok
    assert reason == "spot_offset_mismatch"


def test_fp64_v2_child_merkle_path_is_bound_to_challenged_leaf_index():
    proof, _params, _verifier_args = _valid_child_case()
    tampered = deepcopy(proof)
    tampered["merkle_path"]["siblings"][0][1] = not tampered["merkle_path"][
        "siblings"
    ][0][1]
    assert fp64.verify_fp64_identity_proof(tampered) == (
        False,
        "merkle_path_direction_mismatch",
    )
