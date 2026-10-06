"""Combined capacity proof end to end on the CPU: a reference implementation of the bench's lanes builds a complete
proof (capacity lane with sumcheck block proof and PCS GEMM proof, plus an FP64 lane), and the validator-side
verifier accepts it and rejects tampering."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from verathos.capacity.bench.combined import _attach_gemm_proof
from verathos.capacity.core import derive_sampled_pass_index
from verathos.capacity.proof import fp64
from verathos.capacity.proof.combined import (
    COMBINED_PROOF_FORMAT,
    WORKLOAD_VERSION,
    combined_transcript_root,
    final_commitment,
    lane_params,
    sample_seed,
    verify_combined_proof,
)
from verathos.capacity.proof.workspace import (
    WORKSPACE_PROOF_FORMAT,
    WorkspaceMerkleTree,
    derive_challenge_block_index,
    derive_sampled_round_index,
    pass_root_from_round_roots,
    serialize_gemm_block_proof,
    workspace_i8_values,
    workspace_seed_for,
    workspace_transcript_root_v2,
)
from verathos.capacity.zk.config import Config
from verathos.capacity.zk.crypto.transcript import Transcript
from verathos.capacity.zk.prover.gemm_fast import GEMMProverFast

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


def _workspace(seed: int, pass_index: int, round_index: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.arange(n * n, dtype=np.uint64).reshape(n, n)
    a, b = (torch.from_numpy(workspace_i8_values(workspace_seed=seed, pass_index=pass_index, round_index=round_index,
                                                 matrix_id=m, element_indices=indices)) for m in (0, 1))
    return a, b, a.to(torch.int32) @ b.to(torch.int32)


def _capacity_lane(params: dict) -> tuple[str, callable]:
    """Commit the capacity lane like the timed run; return the proof builder for the post-timing opening."""
    n, bs, passes, rounds = params["matrix_dim"], params["block_size"], params["passes"], params["rounds"]
    seed = workspace_seed_for(LEASE, 0, PROOF_SEED)
    chain = []
    for p in range(passes):
        roots = [WorkspaceMerkleTree.from_matrix_i32(_workspace(seed, p, r, n)[2].tolist(), bs).root.hex()
                 for r in range(rounds)]
        chain.append({"pass_index": p, "pass_root": pass_root_from_round_roots(p, roots), "round_roots": roots})
    transcript = workspace_transcript_root_v2(chain, [])

    def build(sample_hex: str, combined_root: str) -> dict:
        p = derive_sampled_pass_index(sample_hex, LEASE, 0, passes)
        r = derive_sampled_round_index(sample_hex, LEASE, 0, p, rounds)
        a, b, q = _workspace(seed, p, r, n)
        tree = WorkspaceMerkleTree.from_matrix_i32(q.tolist(), bs)
        block_index = derive_challenge_block_index(round_root_hex=tree.root.hex(), lease_id=LEASE, gpu_index=0,
                                                   pass_index=p, round_index=r, blocks_per_row=n // bs,
                                                   challenge_seed_hex=sample_hex)
        bi, bj = divmod(block_index, n // bs)
        transcript_fs = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
        transcript_fs.absorb(b"pass_index", p)
        transcript_fs.absorb(b"round_index", r)
        transcript_fs.absorb_bytes(b"round_root", tree.root)
        block = GEMMProverFast(Config(block_size=bs, spot_checks=SPEC["spot_checks"]))._prove_block_fast(
            X=a, W=b, Y=q, Y_merkle=tree, bi=bi, bj=bj, transcript=transcript_fs)
        q_block = q[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs].tolist()
        proof = {
            "format": WORKSPACE_PROOF_FORMAT, "workload_version": "fixed_workspace_merkle_gemm_v1",
            "transition_mix_rounds": 0, "transition_fanout": 1, "proof_challenge_seed": sample_hex,
            "proof_challenge_public": True, "matrix_dim": n, "block_size": bs, "rounds": rounds, "passes": passes,
            "spot_checks": SPEC["spot_checks"], "workspace_seed": seed, "root_chain": chain,
            "sampled": {"pass_index": p, "round_index": r, "round_root": tree.root.hex(), "block_index": block_index},
            "block_proof": serialize_gemm_block_proof(block, q_block_i32=q_block),
        }
        _attach_gemm_proof(proof, params=params, lease_id=LEASE, proof_seed_hex=PROOF_SEED, sample_seed_hex=sample_hex,
                           combined_root=combined_root)
        return proof

    return transcript, build


def _fp64_lane(params: dict) -> tuple[str, callable]:
    n, bs = params["matrix_dim"], params["block_size"]
    seed = fp64.seed_for(LEASE, 0, PROOF_SEED)
    c = [[fp64.fp64_dot_value(seed=seed, pass_index=0, round_index=0, row=i, col=j, matrix_dim=n) for j in range(n)]
         for i in range(n)]
    per_row = n // bs
    blocks = [[row[bj * bs:(bj + 1) * bs] for row in c[bi * bs:(bi + 1) * bs]]
              for bi in range(per_row) for bj in range(per_row)]
    levels = [[fp64.fp64_leaf_hash(fp64.fp64_c_block_hash(blk)) for blk in blocks]]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([fp64.fp64_parent_hash(cur[i], cur[min(i + 1, len(cur) - 1)]) for i in range(0, len(cur), 2)])
    chain = [{"pass_index": 0, "round_roots": [levels[-1][0].hex()]}]
    transcript = fp64.transcript_root(seed=seed, matrix_dim=n, block_size=bs, passes=1, rounds=1, root_chain=chain)

    def build(sample_hex: str) -> dict:
        sample = fp64.derive_sample(challenge_hex=sample_hex, transcript_hex=transcript, passes=1, rounds=1,
                                    total_blocks=len(blocks), block_size=bs, spot_checks=SPEC["spot_checks"])
        index = sample["block_index"]
        bi, bj = divmod(index, per_row)
        siblings, idx = [], index
        for level in levels[:-1]:
            sib = idx - 1 if idx & 1 else min(idx + 1, len(level) - 1)
            siblings.append([level[sib].hex(), bool(idx & 1)])
            idx //= 2
        spots = []
        for offset in sample["spot_offsets"]:
            lr, lc = divmod(offset, bs)
            spots.append({"offset": offset, "row": bi * bs + lr, "col": bj * bs + lc, "value": blocks[index][lr][lc]})
        return {
            "format": fp64.PROOF_FORMAT, "workload_version": fp64.WORKLOAD_VERSION, "seed": seed, "matrix_dim": n,
            "block_size": bs, "passes": 1, "rounds": 1, "root_chain": chain, "proof_challenge_seed": sample_hex,
            "transcript_root": transcript,
            "sampled": {"pass_index": 0, "round_index": 0, "block_index": index, "bi": bi, "bj": bj},
            "c_block_i64": blocks[index], "leaf_hash": levels[0][index].hex(),
            "merkle_path": {"leaf_index": index, "siblings": siblings}, "spots": spots,
        }

    return transcript, build


@pytest.fixture(scope="module")
def audit() -> tuple[dict, dict]:
    capacity, tail, fp64_params, _ = lane_params(SPEC)
    cap_transcript, build_capacity = _capacity_lane(capacity)
    fp64_transcript, build_fp64 = _fp64_lane(fp64_params)
    root = combined_transcript_root(capacity_transcript=cap_transcript, capacity_tail_transcript="",
                                    fp64_transcript=fp64_transcript, capacity_params=capacity,
                                    capacity_tail_params=tail, fp64_params=fp64_params)
    final_timing = {
        "workload_version": WORKLOAD_VERSION, "pass_count": SPEC["pass_count"], "combined_transcript_root": root,
        "capacity_transcript_root": cap_transcript, "capacity_tail_transcript_root": "",
        "fp64_transcript_root": fp64_transcript, "capacity_params": capacity, "capacity_tail_params": tail,
        "fp64_params": fp64_params, "timed_cuda_component_s": 20.0, "timed_wall_s": 20.5,
    }
    seed = sample_seed(b_proof_seed=B_PROOF, combined_transcript=root)
    proof = {
        "format": COMBINED_PROOF_FORMAT, "workload_version": WORKLOAD_VERSION, "lease_id": LEASE, "gpu_index": 0,
        "pass_count": SPEC["pass_count"], "capacity_params": capacity, "capacity_tail_params": tail,
        "fp64_params": fp64_params, "capacity_transcript_root": cap_transcript, "capacity_tail_transcript_root": "",
        "fp64_transcript_root": fp64_transcript, "combined_transcript_root": root, "b_proof_seed": B_PROOF,
        "sample_seed": seed, "capacity_proof": build_capacity(seed, root), "capacity_tail_proof": None,
        "fp64_proof": build_fp64(seed),
    }
    return proof, final_commitment(final_timing)


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
    (lambda p: p["capacity_proof"]["gemm_v2"].__setitem__("payload", _flip_last_byte(p["capacity_proof"]["gemm_v2"]
                                                                                      ["payload"])), "capacity_gemm"),
    (lambda p: p["capacity_proof"].pop("gemm_v2_state_openings"), "capacity_missing_gemm_proof"),
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
