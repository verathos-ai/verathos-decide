"""FP64 identity lane, GPU side: committed root chains and the post-timing opening (see ``proof.fp64``)."""

from __future__ import annotations

from typing import Any

import torch

from verathos.capacity.bench.workspace import tensor_bytes, tensor_hex
from verathos.capacity.proof.fp64 import (
    PROOF_FORMAT,
    WORKLOAD_VERSION,
    derive_sample,
    fp64_c_block_hash,
    fp64_leaf_hash,
    level_sizes,
)


class FlatMerkleTree:
    """Merkle tree laid out level by level as emitted by ``fp64_c_merkle_commit_f64``."""

    def __init__(self, flat_tree: bytes, num_leaves: int) -> None:
        nodes = len(flat_tree) // 32
        if len(flat_tree) % 32 or nodes != sum(level_sizes(num_leaves)):
            raise ValueError("flat tree does not match the leaf count")
        self.num_leaves = num_leaves
        self.nodes = [flat_tree[i * 32:(i + 1) * 32] for i in range(nodes)]

    @property
    def root(self) -> bytes:
        return self.nodes[-1]

    def path(self, leaf_index: int) -> dict[str, Any]:
        siblings, idx, offset = [], leaf_index, 0
        for size in level_sizes(self.num_leaves)[:-1]:
            sibling = idx - 1 if idx % 2 else min(idx + 1, size - 1)
            siblings.append([self.nodes[offset + sibling].hex(), bool(idx % 2)])
            offset += size
            idx >>= 1
        return {"leaf_index": leaf_index, "siblings": siblings}


def root_chain_from_tensor(round_roots: torch.Tensor, passes: int, rounds: int) -> list[dict[str, Any]]:
    roots = [tensor_hex(round_roots[i]) for i in range(passes * rounds)]
    return [{"pass_index": p, "round_roots": roots[p * rounds:(p + 1) * rounds]} for p in range(passes)]


def build_fp64_proof(*, ext: Any, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, seed: int, matrix_dim: int,
                     block_size: int, passes: int, rounds: int, root_chain: list[dict[str, Any]], challenge_hex: str,
                     transcript_hex: str, spot_checks: int) -> dict[str, Any]:
    """Recompute the challenged round, re-commit it and open one C block with its spot entries."""
    blocks_per_row = matrix_dim // block_size
    total_blocks = blocks_per_row * blocks_per_row
    sample = derive_sample(challenge_hex=challenge_hex, transcript_hex=transcript_hex, passes=passes, rounds=rounds,
                           total_blocks=total_blocks, block_size=block_size, spot_checks=spot_checks)
    pass_index, round_index, block_index = sample["pass_index"], sample["round_index"], sample["block_index"]
    expected_root = root_chain[pass_index]["round_roots"][round_index]

    round_root = torch.empty((1, 32), dtype=torch.uint8, device=c.device)
    ext.run_fp64_identity_round_committed(a, b, c, round_root, seed, pass_index, round_index, block_size, 0)
    if tensor_hex(round_root[0]) != expected_root:
        raise RuntimeError("challenged FP64 round does not reproduce its committed root")
    root_tensor, flat_tensor = ext.fp64_c_merkle_commit_f64(c, block_size, 0)
    tree = FlatMerkleTree(tensor_bytes(flat_tensor), total_blocks)
    if tensor_hex(root_tensor) != expected_root or tree.root.hex() != expected_root:
        raise RuntimeError("FP64 commitment does not reproduce the committed root")

    bi, bj = divmod(block_index, blocks_per_row)
    block = c[bi * block_size:(bi + 1) * block_size, bj * block_size:(bj + 1) * block_size].detach().cpu()
    c_block = [[int(round(float(v))) for v in row] for row in block.tolist()]
    leaf_hash = fp64_leaf_hash(fp64_c_block_hash(c_block))
    if tree.nodes[block_index] != leaf_hash:
        raise RuntimeError("FP64 leaf hash does not match the committed tree")

    spots = []
    for offset in sample["spot_offsets"]:
        local_r, local_c = divmod(offset, block_size)
        spots.append({"offset": offset, "row": bi * block_size + local_r, "col": bj * block_size + local_c,
                      "value": c_block[local_r][local_c]})
    return {
        "format": PROOF_FORMAT,
        "workload_version": WORKLOAD_VERSION,
        "seed": seed,
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": passes,
        "rounds": rounds,
        "root_chain": root_chain,
        "proof_challenge_seed": challenge_hex,
        "transcript_root": transcript_hex,
        "sampled": {"pass_index": pass_index, "round_index": round_index, "block_index": block_index, "bi": bi,
                    "bj": bj},
        "c_block_i64": c_block,
        "leaf_hash": leaf_hash.hex(),
        "merkle_path": tree.path(block_index),
        "spots": spots,
    }
