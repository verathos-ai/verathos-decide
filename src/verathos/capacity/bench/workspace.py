"""Capacity lane: resident int8 workspace, committed GEMM rounds and sampled block proofs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from verathos.capacity.proof.workspace import (
    WorkspaceMerkleTree,
    WorkspaceStateMerkleTree,
    derive_challenge_block_index,
    serialize_gemm_block_proof,
    serialize_merkle_path,
    state_block_index_for_element,
)

BLOCK_TRANSCRIPT_LABEL = b"verathos-hot-capacity-workspace-gemm-v1"


def load_extension() -> Any:
    """The workspace CUDA extension, built in place by ``verathos-build-native``."""
    from verathos.capacity.bench import _workspace_cuda

    return _workspace_cuda


@dataclass
class Workspace:
    a: torch.Tensor
    b: torch.Tensor
    q: torch.Tensor
    partial: torch.Tensor
    root: torch.Tensor
    round_roots: torch.Tensor


def allocate_workspace(n: int, passes: int, rounds: int, device: torch.device) -> Workspace:
    blocks = (n * n + 255) // 256
    return Workspace(
        a=torch.empty((n, n), dtype=torch.int8, device=device),
        b=torch.empty((n, n), dtype=torch.int8, device=device),
        q=torch.empty((n, n), dtype=torch.int32, device=device),
        partial=torch.empty((blocks,), dtype=torch.int64, device=device),
        root=torch.empty((4,), dtype=torch.int64, device=device),
        round_roots=torch.empty((max(1, passes * rounds), 32), dtype=torch.uint8, device=device),
    )


def tensor_hex(tensor: torch.Tensor) -> str:
    return tensor_bytes(tensor).hex()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _block_rows(tensor: torch.Tensor, bi: int, bj: int, block_size: int) -> list[list[int]]:
    block = tensor[bi * block_size:(bi + 1) * block_size, bj * block_size:(bj + 1) * block_size]
    return [[int(v) for v in row] for row in block.detach().cpu().tolist()]


def state_opening(a: torch.Tensor, b: torch.Tensor, tree: WorkspaceStateMerkleTree, block_index: int,
                  matrix_dim: int, block_size: int) -> dict[str, Any]:
    """Merkle-opened int8 A/B block of the resident transition state."""
    bi, bj = divmod(int(block_index), matrix_dim // block_size)
    return {
        "block_index": int(block_index),
        "a_block_i8": _block_rows(a, bi, bj, block_size),
        "b_block_i8": _block_rows(b, bi, bj, block_size),
        "merkle_path": serialize_merkle_path(tree.get_path(int(block_index))),
    }


def state_tree(ext: Any, a: torch.Tensor, b: torch.Tensor, block_size: int) -> tuple[str, WorkspaceStateMerkleTree]:
    root_tensor, flat_tree_tensor = ext.state_merkle_commit_i8(a, b, block_size, 0)
    blocks_per_row = int(a.size(0)) // int(block_size)
    tree = WorkspaceStateMerkleTree.from_flat_tree(tensor_bytes(flat_tree_tensor), blocks_per_row * blocks_per_row)
    return tensor_hex(root_tensor), tree


def gemm_state_blocks(block_proof: dict[str, Any], matrix_dim: int, block_size: int) -> list[int]:
    """State blocks holding the A/B spot-check entries of a block proof."""
    spots = list(block_proof["spot_A"]) + list(block_proof["spot_B"])
    return sorted({state_block_index_for_element(int(s["row"]) * matrix_dim + int(s["col"]), matrix_dim, block_size)
                   for s in spots})


def challenged_block_proof(*, ext: Any, ws: Workspace, round_root: str, pass_index: int, round_index: int,
                           lease_id: str, gpu_index: int, matrix_dim: int, block_size: int, spot_checks: int,
                           challenge_seed_hex: str) -> dict[str, Any]:
    """Sumcheck proof for the challenged Q block of the round resident in ``ws``.

    ``ws.q`` must hold the round whose committed root is ``round_root``; the Q tree is rebuilt on the GPU and
    checked against it before proving.
    """
    from verathos.capacity.zk.config import Config
    from verathos.capacity.zk.crypto.transcript import Transcript
    from verathos.capacity.zk.prover.gemm_fast import GEMMProverFast

    root_tensor, flat_tree_tensor = ext.q_merkle_commit_i32(ws.q, block_size, 0)
    merkle_root = tensor_hex(root_tensor)
    if merkle_root != round_root:
        raise RuntimeError(f"resident Q root mismatch: {merkle_root} != {round_root}")
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=round_root,
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=challenge_seed_hex,
    )
    bi, bj = divmod(block_index, blocks_per_row)
    tree = WorkspaceMerkleTree.from_flat_tree(tensor_bytes(flat_tree_tensor), blocks_per_row * blocks_per_row)

    transcript = Transcript(BLOCK_TRANSCRIPT_LABEL)
    transcript.absorb(b"pass_index", pass_index)
    transcript.absorb(b"round_index", round_index)
    transcript.absorb_bytes(b"round_root", bytes.fromhex(round_root))
    prover = GEMMProverFast(Config(block_size=block_size, spot_checks=spot_checks))
    block_proof = prover._prove_block_fast(X=ws.a, W=ws.b, Y=ws.q, Y_merkle=tree, bi=bi, bj=bj,
                                           transcript=transcript)
    return {
        "block_index": block_index,
        "block_proof": serialize_gemm_block_proof(block_proof, q_block_i32=_block_rows(ws.q, bi, bj, block_size)),
    }


def replay_round(*, ext: Any, ws: Workspace, seed: int, pass_index: int, round_index: int) -> None:
    """Recompute the deterministic workspace up to ``round_index`` of ``pass_index`` (no transition mixing)."""
    for r in range(round_index + 1):
        ext.run_workspace_round(ws.a, ws.b, ws.q, ws.partial, ws.root, seed, pass_index, r, 0)
    torch.cuda.synchronize()
