"""CPU reference of the GPU bench for tests: commits a deterministic-workspace capacity lane and an FP64 lane, then
builds the combined proof for a B_proof seed exactly as ``bench.combined`` does on the GPU. Small workloads only."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verathos.capacity.bench.combined import _attach_gemm_proof
from verathos.capacity.core import derive_sampled_pass_index
from verathos.capacity.proof import fp64
from verathos.capacity.proof.combined import (
    COMBINED_PROOF_FORMAT,
    WORKLOAD_VERSION,
    combined_transcript_root,
    lane_params,
    sample_seed,
)
from verathos.capacity.proof.workspace import (
    WORKSPACE_PROOF_FORMAT,
    WorkspaceMerkleTree,
    block_transcript,
    derive_challenge_block_index,
    derive_sampled_round_index,
    pass_root_from_round_roots,
    serialize_gemm_block_proof,
    workspace_i8_values,
    workspace_seed_for,
    workspace_transcript_root,
)
from verathos.capacity.zk.block_prover import prove_block


def _workspace(seed: int, pass_index: int, round_index: int, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.arange(n * n, dtype=np.uint64).reshape(n, n)
    a, b = (torch.from_numpy(workspace_i8_values(workspace_seed=seed, pass_index=pass_index, round_index=round_index,
                                                 matrix_id=m, element_indices=indices)) for m in (0, 1))
    return a, b, a.to(torch.int32) @ b.to(torch.int32)


class CpuBench:
    """One audit of one slot: ``final_timing`` after construction, ``prove(b_proof_seed)`` after the challenge."""

    def __init__(self, spec: dict[str, Any], lease_id: str, proof_seed_hex: str) -> None:
        self.spec, self.lease, self.proof_seed = spec, lease_id, proof_seed_hex
        self.capacity, self.tail, self.fp64_params, self.spot_checks = lane_params(spec)
        if self.tail is not None or self.capacity["transition_mix_rounds"]:
            raise ValueError("the CPU bench covers deterministic workspaces without a tail lane")
        self._commit_capacity()
        self._commit_fp64()
        self.root = combined_transcript_root(
            capacity_transcript=self.cap_transcript, capacity_tail_transcript="", fp64_transcript=self.fp64_transcript,
            capacity_params=self.capacity, capacity_tail_params=None, fp64_params=self.fp64_params)
        self.final_timing = {
            "lease_id": lease_id, "root": self.root, "pass0_root": self.cap_chain[0]["pass_root"],
            "transcript_root": self.root, "workload_version": WORKLOAD_VERSION, "pass_count": spec["pass_count"],
            "combined_transcript_root": self.root, "capacity_transcript_root": self.cap_transcript,
            "capacity_tail_transcript_root": "", "fp64_transcript_root": self.fp64_transcript,
            "capacity_params": self.capacity, "capacity_tail_params": None, "fp64_params": self.fp64_params,
            "timed_cuda_component_s": 20.0, "timed_wall_s": 20.5,
        }

    def _commit_capacity(self) -> None:
        n, bs = self.capacity["matrix_dim"], self.capacity["block_size"]
        self.ws_seed = workspace_seed_for(self.lease, 0, self.proof_seed)
        self.cap_chain = []
        for p in range(self.capacity["passes"]):
            roots = [WorkspaceMerkleTree.from_matrix_i32(_workspace(self.ws_seed, p, r, n)[2].tolist(), bs).root.hex()
                     for r in range(self.capacity["rounds"])]
            self.cap_chain.append({"pass_index": p, "pass_root": pass_root_from_round_roots(p, roots),
                                   "round_roots": roots})
        self.cap_transcript = workspace_transcript_root(self.cap_chain, [])

    def _commit_fp64(self) -> None:
        self.fp64_transcript = ""
        if not self.fp64_params["passes"]:
            return
        n, bs = self.fp64_params["matrix_dim"], self.fp64_params["block_size"]
        if self.fp64_params["passes"] != 1 or self.fp64_params["rounds"] != 1:
            raise ValueError("the CPU bench covers one FP64 pass of one round")
        self.fp64_seed = fp64.seed_for(self.lease, 0, self.proof_seed)
        c = [[fp64.fp64_dot_value(seed=self.fp64_seed, pass_index=0, round_index=0, row=i, col=j, matrix_dim=n)
              for j in range(n)] for i in range(n)]
        per_row = n // bs
        self.fp64_blocks = [[row[bj * bs:(bj + 1) * bs] for row in c[bi * bs:(bi + 1) * bs]]
                            for bi in range(per_row) for bj in range(per_row)]
        levels = [[fp64.fp64_leaf_hash(fp64.fp64_c_block_hash(blk)) for blk in self.fp64_blocks]]
        while len(levels[-1]) > 1:
            cur = levels[-1]
            levels.append([fp64.fp64_parent_hash(cur[i], cur[min(i + 1, len(cur) - 1)])
                           for i in range(0, len(cur), 2)])
        self.fp64_levels = levels
        self.fp64_chain = [{"pass_index": 0, "round_roots": [levels[-1][0].hex()]}]
        self.fp64_transcript = fp64.transcript_root(seed=self.fp64_seed, matrix_dim=n, block_size=bs, passes=1,
                                                    rounds=1, root_chain=self.fp64_chain)

    def _capacity_proof(self, sample_hex: str) -> dict[str, Any]:
        n, bs, passes, rounds = (self.capacity[k] for k in ("matrix_dim", "block_size", "passes", "rounds"))
        p = derive_sampled_pass_index(sample_hex, self.lease, 0, passes)
        r = derive_sampled_round_index(sample_hex, self.lease, 0, p, rounds)
        a, b, q = _workspace(self.ws_seed, p, r, n)
        tree = WorkspaceMerkleTree.from_matrix_i32(q.tolist(), bs)
        block_index = derive_challenge_block_index(round_root_hex=tree.root.hex(), lease_id=self.lease, gpu_index=0,
                                                   pass_index=p, round_index=r, blocks_per_row=n // bs,
                                                   challenge_seed_hex=sample_hex)
        bi, bj = divmod(block_index, n // bs)
        block = prove_block(x=a, w=b, y=q, q_tree=tree, bi=bi, bj=bj, block_size=bs, spot_checks=self.spot_checks,
                            transcript=block_transcript(p, r, tree.root))
        proof = {
            "format": WORKSPACE_PROOF_FORMAT, "workload_version": "fixed_workspace_merkle_gemm_v1",
            "transition_mix_rounds": 0, "transition_fanout": 1, "proof_challenge_seed": sample_hex,
            "proof_challenge_public": True, "matrix_dim": n, "block_size": bs, "rounds": rounds, "passes": passes,
            "spot_checks": self.spot_checks, "workspace_seed": self.ws_seed, "root_chain": self.cap_chain,
            "sampled": {"pass_index": p, "round_index": r, "round_root": tree.root.hex(), "block_index": block_index},
            "block_proof": serialize_gemm_block_proof(
                block, q_block_i32=q[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs].tolist()),
        }
        _attach_gemm_proof(proof, params=self.capacity, lease_id=self.lease, proof_seed_hex=self.proof_seed,
                           sample_seed_hex=sample_hex, combined_root=self.root)
        return proof

    def _fp64_proof(self, sample_hex: str) -> dict[str, Any]:
        if not self.fp64_params["passes"]:
            return {}
        n, bs = self.fp64_params["matrix_dim"], self.fp64_params["block_size"]
        sample = fp64.derive_sample(challenge_hex=sample_hex, transcript_hex=self.fp64_transcript, passes=1, rounds=1,
                                    total_blocks=len(self.fp64_blocks), block_size=bs, spot_checks=self.spot_checks)
        index = sample["block_index"]
        bi, bj = divmod(index, n // bs)
        siblings, idx = [], index
        for level in self.fp64_levels[:-1]:
            siblings.append([level[idx - 1 if idx & 1 else min(idx + 1, len(level) - 1)].hex(), bool(idx & 1)])
            idx //= 2
        block = self.fp64_blocks[index]
        spots = [{"offset": o, "row": bi * bs + o // bs, "col": bj * bs + o % bs, "value": block[o // bs][o % bs]}
                 for o in sample["spot_offsets"]]
        return {
            "format": fp64.PROOF_FORMAT, "workload_version": fp64.WORKLOAD_VERSION, "seed": self.fp64_seed,
            "matrix_dim": n, "block_size": bs, "passes": 1, "rounds": 1, "root_chain": self.fp64_chain,
            "proof_challenge_seed": sample_hex, "transcript_root": self.fp64_transcript,
            "sampled": {"pass_index": 0, "round_index": 0, "block_index": index, "bi": bi, "bj": bj},
            "c_block_i64": block, "leaf_hash": self.fp64_levels[0][index].hex(),
            "merkle_path": {"leaf_index": index, "siblings": siblings}, "spots": spots,
        }

    def prove(self, b_proof_seed_hex: str) -> dict[str, Any]:
        seed = sample_seed(b_proof_seed=b_proof_seed_hex, combined_transcript=self.root)
        return {
            "format": COMBINED_PROOF_FORMAT, "workload_version": WORKLOAD_VERSION, "lease_id": self.lease,
            "gpu_index": 0, "pass_count": self.spec["pass_count"], "capacity_params": self.capacity,
            "capacity_tail_params": None, "fp64_params": self.fp64_params,
            "capacity_transcript_root": self.cap_transcript, "capacity_tail_transcript_root": "",
            "fp64_transcript_root": self.fp64_transcript, "combined_transcript_root": self.root,
            "b_proof_seed": b_proof_seed_hex, "sample_seed": seed, "capacity_proof": self._capacity_proof(seed),
            "capacity_tail_proof": None, "fp64_proof": self._fp64_proof(seed),
        }
