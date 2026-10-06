#!/usr/bin/env python3
"""Benchmark a fixed-workspace hot-capacity CUDA workload.

This is an implementation prototype, not the production protocol. It proves the
critical property we need before calibration: bounded VRAM without per-pass
allocator flushing or repeated tensor allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

from verathos.capacity.core import derive_sampled_pass_index, root_words_digest, transcript_root  # noqa: E402
from verathos.capacity.workspace_proof import (  # noqa: E402
    WORKSPACE_COMPACT_PROOF_FORMAT,
    WORKSPACE_MIXED_PROOF_FORMAT,
    WORKSPACE_PROOF_FORMAT,
    WORKSPACE_TRANSITION_PROOF_FORMAT,
    WorkspaceMerkleTree,
    WorkspaceStateMerkleTree,
    derive_challenge_block_index,
    derive_sampled_round_index,
    derive_sampled_transition_mix_index,
    derive_transition_element_index,
    pass_root_from_round_roots,
    preload_workspace_proof_verifier,
    serialize_gemm_block_proof,
    serialize_merkle_path,
    transition_source_indices,
    verify_workspace_proof_payload,
    workspace_i8_value,
    workspace_transcript_root,
)


def _self_command() -> list[str]:
    script = SCRIPT_DIR / "bench_workspace.py"
    if script.exists():
        return [sys.executable, str(script)]
    return [sys.executable, "-c", "from verathos.capacity.bench.bench_workspace import main; main()"]


def workload_version_for_mix(memory_mix_rounds: int) -> str:
    return "fixed_workspace_mixed_v2" if int(memory_mix_rounds) > 0 else "fixed_workspace_merkle_gemm_v1"


def proof_format_for_mix(memory_mix_rounds: int) -> str:
    return WORKSPACE_MIXED_PROOF_FORMAT if int(memory_mix_rounds) > 0 else WORKSPACE_PROOF_FORMAT


def workload_version_for_args(args: argparse.Namespace) -> str:
    if int(getattr(args, "transition_mix_rounds", 0) or 0) > 0:
        if int(getattr(args, "transition_fanout", 1) or 1) > 1:
            return "fixed_workspace_transition_fanout_v3"
        return "fixed_workspace_transition_mix_v3"
    return workload_version_for_mix(int(getattr(args, "memory_mix_rounds", 0) or 0))


def proof_format_for_args(args: argparse.Namespace) -> str:
    if int(getattr(args, "transition_mix_rounds", 0) or 0) > 0:
        return WORKSPACE_TRANSITION_PROOF_FORMAT
    return proof_format_for_mix(int(getattr(args, "memory_mix_rounds", 0) or 0))


def _import_ext():
    # Prefer the extension built beside this benchmark. The repo root may carry
    # an older calibration artifact from a previous local build.
    # The compiled extension is built into this directory by build.py.
    while str(SCRIPT_DIR) in sys.path:
        sys.path.remove(str(SCRIPT_DIR))
    sys.path.insert(0, str(SCRIPT_DIR))
    import hot_capacity_workspace_cuda as ext  # type: ignore

    return ext


def matrix_dim_for_workspace_mib(target_mib: float) -> int:
    # A int8 + B int8 + Q int32 = 6 bytes per matrix cell. Partial/root are small.
    budget = int(target_mib * 1024 * 1024 * 0.965)
    n = int(math.sqrt(budget / 6.0))
    return max(256, (n // 256) * 256)


def seed_for(lease_id: str, gpu_index: int, seed_hex: str = "") -> int:
    import hashlib

    h = hashlib.sha256()
    h.update(b"VERATHOS_FIXED_WORKSPACE_V1")
    if seed_hex:
        raw = seed_hex[2:] if seed_hex.startswith("0x") else seed_hex
        h.update(bytes.fromhex(raw))
    h.update(lease_id.encode())
    h.update(int(gpu_index).to_bytes(4, "little"))
    return int.from_bytes(h.digest()[:8], "little")


@dataclass
class Workspace:
    a: torch.Tensor
    b: torch.Tensor
    q: torch.Tensor
    partial: torch.Tensor
    root: torch.Tensor
    roots: torch.Tensor
    round_roots: torch.Tensor


@dataclass
class CapturedRound:
    pass_index: int
    round_index: int
    round_root: str
    block_index: int
    flat_tree: bytes
    q_block: torch.Tensor
    a_rows: torch.Tensor
    b_cols: torch.Tensor


def allocate_workspace(n: int, passes: int, rounds: int, device: str) -> Workspace:
    total = n * n
    blocks = (total + 255) // 256
    return Workspace(
        a=torch.empty((n, n), dtype=torch.int8, device=device),
        b=torch.empty((n, n), dtype=torch.int8, device=device),
        q=torch.empty((n, n), dtype=torch.int32, device=device),
        partial=torch.empty((blocks,), dtype=torch.int64, device=device),
        root=torch.empty((4,), dtype=torch.int64, device=device),
        roots=torch.empty((max(1, passes), 4), dtype=torch.int64, device=device),
        round_roots=torch.empty((max(1, passes * rounds), 32), dtype=torch.uint8, device=device),
    )


def _tensor_u8_hex(tensor: torch.Tensor) -> str:
    return tensor.detach().cpu().contiguous().numpy().tobytes().hex()


def _tensor_u8_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _q_block_rows(q: torch.Tensor, bi: int, bj: int, block_size: int) -> list[list[int]]:
    block = q[
        bi * block_size:(bi + 1) * block_size,
        bj * block_size:(bj + 1) * block_size,
    ].detach().cpu().tolist()
    return [[int(v) for v in row] for row in block]


def _matrix_i8_rows(tensor: torch.Tensor) -> list[list[int]]:
    return [[int(v) for v in row] for row in tensor.detach().cpu().tolist()]


def _filled_state_matrices(
    *,
    seed: int,
    pass_index: int,
    round_index: int,
    matrix_dim: int,
) -> tuple[list[list[int]], list[list[int]]]:
    n = int(matrix_dim)
    a_rows = [
        [
            workspace_i8_value(
                workspace_seed=seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=0,
                row=i,
                col=j,
                matrix_dim=n,
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    b_rows = [
        [
            workspace_i8_value(
                workspace_seed=seed,
                pass_index=pass_index,
                round_index=round_index,
                matrix_id=1,
                row=i,
                col=j,
                matrix_dim=n,
            )
            for j in range(n)
        ]
        for i in range(n)
    ]
    return a_rows, b_rows


def _state_block_index_for_element(element_index: int, matrix_dim: int, block_size: int) -> int:
    row = int(element_index) // int(matrix_dim)
    col = int(element_index) % int(matrix_dim)
    blocks_per_row = int(matrix_dim) // int(block_size)
    return (row // int(block_size)) * blocks_per_row + (col // int(block_size))


def _state_opening(
    a_rows: list[list[int]],
    b_rows: list[list[int]],
    tree: WorkspaceStateMerkleTree,
    block_index: int,
    matrix_dim: int,
    block_size: int,
) -> dict[str, Any]:
    blocks_per_row = int(matrix_dim) // int(block_size)
    bi = int(block_index) // blocks_per_row
    bj = int(block_index) % blocks_per_row
    a_block = [
        row[bj * block_size:(bj + 1) * block_size]
        for row in a_rows[bi * block_size:(bi + 1) * block_size]
    ]
    b_block = [
        row[bj * block_size:(bj + 1) * block_size]
        for row in b_rows[bi * block_size:(bi + 1) * block_size]
    ]
    return {
        "block_index": int(block_index),
        "a_block_i8": a_block,
        "b_block_i8": b_block,
        "merkle_path": serialize_merkle_path(tree.get_path(int(block_index))),
    }


def _i8_block_rows_from_tensor(
    tensor: torch.Tensor,
    block_index: int,
    matrix_dim: int,
    block_size: int,
) -> list[list[int]]:
    blocks_per_row = int(matrix_dim) // int(block_size)
    bi = int(block_index) // blocks_per_row
    bj = int(block_index) % blocks_per_row
    block = tensor[
        bi * block_size:(bi + 1) * block_size,
        bj * block_size:(bj + 1) * block_size,
    ].detach().cpu().tolist()
    return [[int(v) for v in row] for row in block]


def _state_opening_from_tensors(
    a: torch.Tensor,
    b: torch.Tensor,
    tree: WorkspaceStateMerkleTree,
    block_index: int,
    matrix_dim: int,
    block_size: int,
) -> dict[str, Any]:
    return {
        "block_index": int(block_index),
        "a_block_i8": _i8_block_rows_from_tensor(a, int(block_index), matrix_dim, block_size),
        "b_block_i8": _i8_block_rows_from_tensor(b, int(block_index), matrix_dim, block_size),
        "merkle_path": serialize_merkle_path(tree.get_path(int(block_index))),
    }


def _state_tree_from_resident_tensors(ext, a: torch.Tensor, b: torch.Tensor, block_size: int):
    root_tensor, flat_tree_tensor = ext.state_merkle_commit_i8(a, b, block_size, 0)
    root_hex = _tensor_u8_hex(root_tensor)
    flat_tree = _tensor_u8_bytes(flat_tree_tensor)
    n = int(a.size(0))
    blocks_per_row = n // int(block_size)
    tree = WorkspaceStateMerkleTree.from_flat_tree(flat_tree, blocks_per_row * blocks_per_row)
    return root_hex, tree


def _normalize_hex_32(text: str) -> str:
    raw = str(text or "").strip()
    raw = raw[2:] if raw.startswith("0x") else raw
    if len(raw) != 64:
        raise ValueError("expected 32-byte hex challenge")
    bytes.fromhex(raw)
    return raw


def _resolve_challenge_seed(args: argparse.Namespace, fallback_transcript: str) -> tuple[str, bool]:
    if getattr(args, "challenge_hex", ""):
        return _normalize_hex_32(args.challenge_hex), True
    challenge_file_text = str(getattr(args, "challenge_file", "") or "")
    if challenge_file_text:
        challenge_file = pathlib.Path(challenge_file_text)
        deadline = time.perf_counter() + max(0.0, float(getattr(args, "challenge_timeout_s", 0.0) or 0.0))
        while time.perf_counter() <= deadline:
            if challenge_file.exists():
                return _normalize_hex_32(challenge_file.read_text()), True
            time.sleep(0.25)
        return "", False
    return fallback_transcript, False


def _resolve_public_challenge_seed(args: argparse.Namespace) -> str:
    """Resolve an explicitly pre-run public challenge.

    A challenge file represents the production B_proof flow and may not exist
    until after final_timing is published. Never wait for that file before the
    timed workload, otherwise the benchmark would delay B_start work until
    B_proof. The no-replay capture path is diagnostic-only and requires an
    already-known --challenge-hex.
    """
    if getattr(args, "challenge_hex", ""):
        return _normalize_hex_32(args.challenge_hex)
    return ""


def _build_block_proof_from_capture(
    *,
    captured: CapturedRound,
    matrix_dim: int,
    block_size: int,
    spot_checks: int,
) -> dict[str, Any]:
    blocks_per_row = int(matrix_dim) // int(block_size)
    bi = int(captured.block_index) // blocks_per_row
    bj = int(captured.block_index) % blocks_per_row
    tree = WorkspaceMerkleTree.from_flat_tree(
        captured.flat_tree,
        blocks_per_row * blocks_per_row,
    )

    from verathos.zk.config import Config
    from verathos.zk.crypto.transcript import Transcript
    from verathos.zk.prover.gemm_fast import GEMMProverFast
    from verathos.zk.types import GEMMBlockProof

    transcript = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
    transcript.absorb(b"pass_index", int(captured.pass_index))
    transcript.absorb(b"round_index", int(captured.round_index))
    transcript.absorb_bytes(b"round_root", bytes.fromhex(captured.round_root))

    leaf_hash = tree.get_leaf(bi, bj)
    merkle_path = tree.get_path(bi, bj)
    transcript.absorb(b"bi", bi)
    transcript.absorb(b"bj", bj)
    transcript.absorb_bytes(b"leaf_hash", leaf_hash)

    prover = GEMMProverFast(Config(block_size=block_size, spot_checks=spot_checks))
    sumcheck_proof, _challenges = prover._sumcheck_for_block_fast(
        X_rows=captured.a_rows,
        W_cols=captured.b_cols,
        Y_block=captured.q_block,
        transcript=transcript,
    )
    row_start = bi * int(block_size)
    col_start = bj * int(block_size)
    spot_x = prover._sample_spots_with_offset(
        captured.a_rows,
        transcript,
        b"spot_X",
        row_offset=row_start,
        col_offset=0,
    )
    spot_w = prover._sample_spots_with_offset(
        captured.b_cols,
        transcript,
        b"spot_W",
        row_offset=0,
        col_offset=col_start,
    )
    m, k_dim = captured.a_rows.shape
    _, n_cols = captured.b_cols.shape
    block_proof = GEMMBlockProof(
        bi=bi,
        bj=bj,
        leaf_hash=leaf_hash,
        merkle_path=merkle_path,
        sumcheck_proof=sumcheck_proof,
        spot_X=spot_x,
        spot_W=spot_w,
        m_bits=max(1, (int(m) - 1).bit_length()),
        n_bits=max(1, (int(n_cols) - 1).bit_length()),
        k_bits=max(1, (int(k_dim) - 1).bit_length()),
    )
    return {
        "block_index": int(captured.block_index),
        "block_proof": serialize_gemm_block_proof(
            block_proof,
            q_block_i32=captured.q_block.tolist(),
        ),
    }


def _run_merkle_pass_with_optional_capture(
    *,
    ext,
    ws: Workspace,
    seed: int,
    pass_index: int,
    rounds: int,
    block_size: int,
    lease_id: str,
    gpu_index: int,
    challenge_seed: str,
    capture_round: int | None,
    memory_mix_rounds: int = 0,
) -> CapturedRound | None:
    n = int(ws.a.size(0))
    blocks_per_row = n // int(block_size)
    captured: CapturedRound | None = None
    for round_index in range(int(rounds)):
        if memory_mix_rounds > 0:
            ext.run_workspace_round_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root,
                seed, pass_index, round_index, memory_mix_rounds, 0,
            )
        else:
            ext.run_workspace_round(
                ws.a, ws.b, ws.q, ws.partial, ws.root,
                seed, pass_index, round_index, 0,
            )
        root_offset = int(pass_index) * int(rounds) + round_index
        if capture_round is not None and round_index == int(capture_round):
            root_tensor, flat_tree_tensor = ext.q_merkle_commit_i32(ws.q, block_size, 0)
            round_root = _tensor_u8_hex(root_tensor)
            ws.round_roots[root_offset].copy_(root_tensor.to(device=ws.round_roots.device))
            block_index = derive_challenge_block_index(
                round_root_hex=round_root,
                lease_id=lease_id,
                gpu_index=gpu_index,
                pass_index=pass_index,
                round_index=round_index,
                blocks_per_row=blocks_per_row,
                challenge_seed_hex=challenge_seed,
            )
            bi = block_index // blocks_per_row
            bj = block_index % blocks_per_row
            captured = CapturedRound(
                pass_index=int(pass_index),
                round_index=int(round_index),
                round_root=round_root,
                block_index=int(block_index),
                flat_tree=_tensor_u8_bytes(flat_tree_tensor),
                q_block=ws.q[
                    bi * block_size:(bi + 1) * block_size,
                    bj * block_size:(bj + 1) * block_size,
                ].detach().cpu().contiguous(),
                a_rows=ws.a[
                    bi * block_size:(bi + 1) * block_size,
                    :
                ].detach().cpu().contiguous(),
                b_cols=ws.b[
                    :,
                    bj * block_size:(bj + 1) * block_size,
                ].detach().cpu().contiguous(),
            )
        else:
            root_tensor = ext.q_merkle_root_i32(ws.q, block_size, 0)
            ws.round_roots[root_offset].copy_(root_tensor.to(device=ws.round_roots.device))
    return captured


def _build_sampled_block_proof(
    *,
    ext,
    ws: Workspace,
    seed: int,
    pass_index: int,
    round_index: int,
    round_root: str,
    lease_id: str,
    gpu_index: int,
    matrix_dim: int,
    block_size: int,
    spot_checks: int,
    challenge_seed_hex: str = "",
    memory_mix_rounds: int = 0,
) -> dict[str, Any]:
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
    bi = block_index // blocks_per_row
    bj = block_index % blocks_per_row

    for r in range(round_index + 1):
        if memory_mix_rounds > 0:
            ext.run_workspace_round_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root,
                seed, pass_index, r, memory_mix_rounds, 0,
            )
        else:
            ext.run_workspace_round(
                ws.a, ws.b, ws.q, ws.partial, ws.root,
                seed, pass_index, r, 0,
            )
    torch.cuda.synchronize()
    root_tensor, flat_tree_tensor = ext.q_merkle_commit_i32(ws.q, block_size, 0)
    recomputed_root = _tensor_u8_hex(root_tensor)
    if recomputed_root != round_root:
        raise RuntimeError(
            f"sampled round root mismatch: {recomputed_root} != {round_root}"
        )
    flat_tree = _tensor_u8_bytes(flat_tree_tensor)
    tree = WorkspaceMerkleTree.from_flat_tree(flat_tree, blocks_per_row * blocks_per_row)

    from verathos.zk.config import Config
    from verathos.zk.crypto.transcript import Transcript
    from verathos.zk.prover.gemm_fast import GEMMProverFast

    transcript = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
    transcript.absorb(b"pass_index", pass_index)
    transcript.absorb(b"round_index", round_index)
    transcript.absorb_bytes(b"round_root", bytes.fromhex(round_root))
    prover = GEMMProverFast(Config(block_size=block_size, spot_checks=spot_checks))
    block_proof = prover._prove_block_fast(
        X=ws.a,
        W=ws.b,
        Y=ws.q,
        Y_merkle=tree,
        bi=bi,
        bj=bj,
        transcript=transcript,
    )
    return {
        "block_index": block_index,
        "block_proof": serialize_gemm_block_proof(
            block_proof,
            q_block_i32=_q_block_rows(ws.q, bi, bj, block_size),
        ),
    }


def _build_resident_block_proof(
    *,
    ext,
    ws: Workspace,
    pass_index: int,
    round_index: int,
    lease_id: str,
    gpu_index: int,
    matrix_dim: int,
    block_size: int,
    spot_checks: int,
) -> dict[str, Any]:
    root_tensor, flat_tree_tensor = ext.q_merkle_commit_i32(ws.q, block_size, 0)
    merkle_root = _tensor_u8_hex(root_tensor)
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=merkle_root,
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
    )
    bi = block_index // blocks_per_row
    bj = block_index % blocks_per_row
    flat_tree = _tensor_u8_bytes(flat_tree_tensor)
    tree = WorkspaceMerkleTree.from_flat_tree(flat_tree, blocks_per_row * blocks_per_row)

    from verathos.zk.config import Config
    from verathos.zk.crypto.transcript import Transcript
    from verathos.zk.prover.gemm_fast import GEMMProverFast

    transcript = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
    transcript.absorb(b"pass_index", pass_index)
    transcript.absorb(b"round_index", round_index)
    transcript.absorb_bytes(b"round_root", bytes.fromhex(merkle_root))
    prover = GEMMProverFast(Config(block_size=block_size, spot_checks=spot_checks))
    block_proof = prover._prove_block_fast(
        X=ws.a,
        W=ws.b,
        Y=ws.q,
        Y_merkle=tree,
        bi=bi,
        bj=bj,
        transcript=transcript,
    )
    return {
        "merkle_root": merkle_root,
        "block_index": block_index,
        "block_proof": serialize_gemm_block_proof(
            block_proof,
            q_block_i32=_q_block_rows(ws.q, bi, bj, block_size),
        ),
    }


def _build_resident_challenged_block_proof(
    *,
    ext,
    ws: Workspace,
    round_root: str,
    pass_index: int,
    round_index: int,
    lease_id: str,
    gpu_index: int,
    matrix_dim: int,
    block_size: int,
    spot_checks: int,
    challenge_seed_hex: str,
) -> dict[str, Any]:
    root_tensor, flat_tree_tensor = ext.q_merkle_commit_i32(ws.q, block_size, 0)
    merkle_root = _tensor_u8_hex(root_tensor)
    if merkle_root != str(round_root):
        raise RuntimeError(f"resident Q root mismatch: {merkle_root} != {round_root}")
    blocks_per_row = matrix_dim // block_size
    block_index = derive_challenge_block_index(
        round_root_hex=merkle_root,
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=pass_index,
        round_index=round_index,
        blocks_per_row=blocks_per_row,
        challenge_seed_hex=challenge_seed_hex,
    )
    bi = block_index // blocks_per_row
    bj = block_index % blocks_per_row
    flat_tree = _tensor_u8_bytes(flat_tree_tensor)
    tree = WorkspaceMerkleTree.from_flat_tree(flat_tree, blocks_per_row * blocks_per_row)

    from verathos.zk.config import Config
    from verathos.zk.crypto.transcript import Transcript
    from verathos.zk.prover.gemm_fast import GEMMProverFast

    transcript = Transcript(b"verathos-hot-capacity-workspace-gemm-v1")
    transcript.absorb(b"pass_index", pass_index)
    transcript.absorb(b"round_index", round_index)
    transcript.absorb_bytes(b"round_root", bytes.fromhex(merkle_root))
    prover = GEMMProverFast(Config(block_size=block_size, spot_checks=spot_checks))
    block_proof = prover._prove_block_fast(
        X=ws.a,
        W=ws.b,
        Y=ws.q,
        Y_merkle=tree,
        bi=bi,
        bj=bj,
        transcript=transcript,
    )
    return {
        "merkle_root": merkle_root,
        "block_index": block_index,
        "block_proof": serialize_gemm_block_proof(
            block_proof,
            q_block_i32=_q_block_rows(ws.q, bi, bj, block_size),
        ),
    }


def _child_compact(args: argparse.Namespace, ext, ws: Workspace, seed: int, out_dir: pathlib.Path) -> None:
    started = time.perf_counter()
    pass_ms: list[float] = []

    if args.passes > 0:
        ms = ext.run_workspace_pass(
            ws.a, ws.b, ws.q, ws.partial, ws.root,
            seed, 0, args.rounds, 0,
        )
        torch.cuda.synchronize()
        root_words = [int(x) for x in ws.root.detach().cpu().tolist()]
        ws.roots[0].copy_(ws.root)
        pass_ms.append(float(ms))
        (out_dir / f"{args.lease_id}_pass0.json").write_text(json.dumps({
            "lease_id": args.lease_id,
            "pass_index": 0,
            "root": root_words,
            "t": time.time(),
        }))

    chain_ms = 0.0
    if args.passes > 1:
        chain_ms = ext.run_workspace_chain(
            ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots,
            seed, 1, args.passes - 1, args.rounds, 0,
        )
        torch.cuda.synchronize()
        root_words = [int(x) for x in ws.root.detach().cpu().tolist()]
        pass_ms.extend([float(chain_ms) / float(args.passes - 1)] * (args.passes - 1))
    elif args.passes == 1:
        root_words = [int(x) for x in ws.root.detach().cpu().tolist()]
    else:
        root_words = []

    torch.cuda.synchronize()
    root_chain: list[list[int]] = []
    if args.passes > 0:
        root_chain = [
            [int(x) for x in row[:4]]
            for row in ws.roots[:args.passes].detach().cpu().tolist()
        ]
    final_transcript = transcript_root(root_chain) if root_chain else ""
    timed_elapsed_s = time.perf_counter() - started
    (out_dir / f"{args.lease_id}_final_timing.json").write_text(json.dumps({
        "lease_id": args.lease_id,
        "passes": args.passes,
        "root": root_words,
        "transcript_root": final_transcript,
        "proof_format": WORKSPACE_COMPACT_PROOF_FORMAT,
        "t": time.time(),
    }))

    proof_payload: dict[str, Any] = {}
    proof_payload_s = 0.0
    if root_chain:
        proof_started = time.perf_counter()
        sampled_pass = derive_sampled_pass_index(
            final_transcript,
            args.lease_id,
            args.gpu_index,
            args.passes,
        )
        ext.run_workspace_pass(
            ws.a, ws.b, ws.q, ws.partial, ws.root,
            seed, sampled_pass, args.rounds, 0,
        )
        torch.cuda.synchronize()
        recomputed_root = [int(x) for x in ws.root.detach().cpu().tolist()]
        if recomputed_root != root_chain[sampled_pass]:
            raise RuntimeError(
                f"sampled compact root mismatch: {recomputed_root} != {root_chain[sampled_pass]}"
            )
        round_index = int(args.rounds) - 1
        sampled = _build_resident_block_proof(
            ext=ext,
            ws=ws,
            pass_index=sampled_pass,
            round_index=round_index,
            lease_id=args.lease_id,
            gpu_index=args.gpu_index,
            matrix_dim=ws.a.size(0),
            block_size=int(args.block_size),
            spot_checks=int(args.spot_checks),
        )
        proof_payload_s = time.perf_counter() - proof_started
        proof_payload = {
            "format": WORKSPACE_COMPACT_PROOF_FORMAT,
            "timing_root_format": "fixed_workspace_root_chain_v0",
            "matrix_dim": int(ws.a.size(0)),
            "block_size": int(args.block_size),
            "rounds": int(args.rounds),
            "passes": int(args.passes),
            "spot_checks": int(args.spot_checks),
            "workspace_seed": int(seed),
            "root_chain": root_chain,
            "sampled": {
                "pass_index": sampled_pass,
                "round_index": round_index,
                "compact_root": root_chain[sampled_pass],
                "compact_root_digest": root_words_digest(root_chain[sampled_pass]),
                "merkle_root": sampled["merkle_root"],
                "block_index": sampled["block_index"],
            },
            "block_proof": sampled["block_proof"],
        }

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final = {
        "proof_mode": "compact",
        "lease_id": args.lease_id,
        "gpu_index": args.gpu_index,
        "matrix_dim": args.matrix_dim or matrix_dim_for_workspace_mib(args.workspace_mib),
        "workspace_mib_target": args.workspace_mib,
        "workspace_bytes_nominal": 6 * ws.a.size(0) * ws.a.size(0) + 8 * ((ws.a.size(0) * ws.a.size(0) + 255) // 256) + 32,
        "passes": args.passes,
        "rounds": args.rounds,
        "block_size": int(args.block_size),
        "spot_checks": int(args.spot_checks),
        "chain_ms": chain_ms,
        "elapsed_s": time.perf_counter() - started,
        "timed_elapsed_s": timed_elapsed_s,
        "mean_pass_ms": sum(pass_ms) / len(pass_ms) if pass_ms else 0.0,
        "p50_pass_ms": sorted(pass_ms)[len(pass_ms) // 2] if pass_ms else 0.0,
        "peak_allocated_mib": peak_alloc / (1024 * 1024),
        "peak_reserved_mib": peak_reserved / (1024 * 1024),
        "final_root": root_words,
        "root_chain": root_chain,
        "timing_transcript_root": final_transcript,
        "proof_payload": proof_payload,
        "proof_payload_s": proof_payload_s,
    }
    (out_dir / f"{args.lease_id}_final.json").write_text(json.dumps(final, indent=2))


def _child_merkle(args: argparse.Namespace, ext, ws: Workspace, seed: int, out_dir: pathlib.Path) -> None:
    public_challenge_seed = _resolve_public_challenge_seed(args)
    if public_challenge_seed:
        return _child_merkle_public_challenge(args, ext, ws, seed, out_dir, public_challenge_seed)

    started = time.perf_counter()
    pass_ms: list[float] = []
    block_size = int(args.block_size)
    n = int(ws.a.size(0))
    root_chain: list[dict[str, Any]] = []

    if args.passes > 0:
        if args.memory_mix_rounds > 0:
            ms = ext.run_workspace_pass_committed_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.round_roots,
                seed, 0, args.rounds, block_size, args.memory_mix_rounds, 0,
            )
        else:
            ms = ext.run_workspace_pass_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.round_roots,
                seed, 0, args.rounds, block_size, 0,
            )
        torch.cuda.synchronize()
        ws.roots[0].copy_(ws.root)
        pass_ms.append(float(ms))
        pass0_round_roots = [
            _tensor_u8_hex(ws.round_roots[round_index])
            for round_index in range(args.rounds)
        ]
        pass0_root = pass_root_from_round_roots(0, pass0_round_roots)
        (out_dir / f"{args.lease_id}_pass0.json").write_text(json.dumps({
            "lease_id": args.lease_id,
            "pass_index": 0,
            "root": pass0_root,
            "round_roots": pass0_round_roots,
            "t": time.time(),
        }))

    chain_ms = 0.0
    if args.passes > 1:
        if args.memory_mix_rounds > 0:
            chain_ms = ext.run_workspace_chain_committed_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, 1, args.passes - 1, args.rounds, block_size, args.memory_mix_rounds, 0,
            )
        else:
            chain_ms = ext.run_workspace_chain_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, 1, args.passes - 1, args.rounds, block_size, 0,
            )
        torch.cuda.synchronize()
        pass_ms.extend([float(chain_ms) / float(args.passes - 1)] * (args.passes - 1))
    elif args.passes == 1:
        torch.cuda.synchronize()

    if args.passes > 0:
        round_root_rows = ws.round_roots[:args.passes * args.rounds].detach().cpu().tolist()
        for pass_index in range(args.passes):
            start = pass_index * args.rounds
            round_roots = [
                bytes(int(x) for x in round_root_rows[start + round_index][:32]).hex()
                for round_index in range(args.rounds)
            ]
            pass_root = pass_root_from_round_roots(pass_index, round_roots)
            root_chain.append({
                "pass_index": pass_index,
                "pass_root": pass_root,
                "round_roots": round_roots,
            })

    final_root = root_chain[-1]["pass_root"] if root_chain else ""
    pass0_root = root_chain[0]["pass_root"] if root_chain else ""
    final_transcript = workspace_transcript_root(root_chain) if root_chain else ""
    timed_elapsed_s = time.perf_counter() - started
    (out_dir / f"{args.lease_id}_final_timing.json").write_text(json.dumps({
        "lease_id": args.lease_id,
        "passes": args.passes,
        "root": final_root,
        "pass0_root": pass0_root,
        "transcript_root": final_transcript,
        "proof_format": proof_format_for_mix(args.memory_mix_rounds),
        "t": time.time(),
    }))

    proof_payload = {}
    proof_payload_s = 0.0
    challenge_seed, challenge_is_public = (
        ("", False) if args.skip_proof_payload else _resolve_challenge_seed(args, final_transcript)
    )
    if root_chain and challenge_seed:
        proof_started = time.perf_counter()
        sampled_pass = derive_sampled_pass_index(challenge_seed, args.lease_id, args.gpu_index, args.passes)
        sampled_round = derive_sampled_round_index(
            challenge_seed,
            args.lease_id,
            args.gpu_index,
            sampled_pass,
            args.rounds,
        )
        sampled_round_root = root_chain[sampled_pass]["round_roots"][sampled_round]
        sampled = _build_sampled_block_proof(
            ext=ext,
            ws=ws,
            seed=seed,
            pass_index=sampled_pass,
            round_index=sampled_round,
            round_root=sampled_round_root,
            lease_id=args.lease_id,
            gpu_index=args.gpu_index,
            matrix_dim=n,
            block_size=block_size,
            spot_checks=args.spot_checks,
            challenge_seed_hex=challenge_seed,
            memory_mix_rounds=args.memory_mix_rounds,
        )
        proof_payload_s = time.perf_counter() - proof_started
        proof_payload = {
            "format": proof_format_for_mix(args.memory_mix_rounds),
            "workload_version": workload_version_for_mix(args.memory_mix_rounds),
            "memory_mix_rounds": int(args.memory_mix_rounds),
            "proof_challenge_seed": challenge_seed,
            "proof_challenge_public": bool(challenge_is_public),
            "matrix_dim": n,
            "block_size": block_size,
            "rounds": args.rounds,
            "passes": args.passes,
            "spot_checks": args.spot_checks,
            "workspace_seed": int(seed),
            "root_chain": root_chain,
            "sampled": {
                "pass_index": sampled_pass,
                "round_index": sampled_round,
                "round_root": sampled_round_root,
                "block_index": sampled["block_index"],
            },
            "block_proof": sampled["block_proof"],
        }

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final = {
        "proof_mode": "merkle",
        "workload_version": workload_version_for_mix(args.memory_mix_rounds),
        "memory_mix_rounds": int(args.memory_mix_rounds),
        "lease_id": args.lease_id,
        "gpu_index": args.gpu_index,
        "matrix_dim": n,
        "workspace_mib_target": args.workspace_mib,
        "workspace_bytes_nominal": 6 * n * n + 8 * ((n * n + 255) // 256) + 32,
        "passes": args.passes,
        "rounds": args.rounds,
        "block_size": block_size,
        "spot_checks": args.spot_checks,
        "elapsed_s": time.perf_counter() - started,
        "timed_elapsed_s": timed_elapsed_s,
        "mean_pass_ms": sum(pass_ms) / len(pass_ms) if pass_ms else 0.0,
        "p50_pass_ms": sorted(pass_ms)[len(pass_ms) // 2] if pass_ms else 0.0,
        "peak_allocated_mib": peak_alloc / (1024 * 1024),
        "peak_reserved_mib": peak_reserved / (1024 * 1024),
        "pass0_root": pass0_root,
        "final_root": final_root,
        "root_chain": root_chain,
        "timing_transcript_root": final_transcript,
        "proof_payload": proof_payload,
        "proof_payload_s": proof_payload_s,
    }
    (out_dir / f"{args.lease_id}_final.json").write_text(json.dumps(final, indent=2))


def _child_merkle_public_challenge(
    args: argparse.Namespace,
    ext,
    ws: Workspace,
    seed: int,
    out_dir: pathlib.Path,
    challenge_seed: str,
) -> None:
    """Run the Merkle proof mode when the proof challenge is public pre-run.

    This avoids the old post-final_timing replay: the sampled pass/round is
    known up front, so the benchmark captures only that round's verifier
    opening material while the calibrated chain is running. The expensive proof
    object construction still happens after final_timing.
    """
    started = time.perf_counter()
    pass_ms: list[float] = []
    block_size = int(args.block_size)
    n = int(ws.a.size(0))
    sampled_pass = derive_sampled_pass_index(
        challenge_seed,
        args.lease_id,
        args.gpu_index,
        args.passes,
    )
    sampled_round = derive_sampled_round_index(
        challenge_seed,
        args.lease_id,
        args.gpu_index,
        sampled_pass,
        args.rounds,
    )
    captured: CapturedRound | None = None

    def _pass_round_roots(pass_index: int) -> list[str]:
        rows = ws.round_roots[
            pass_index * args.rounds:(pass_index + 1) * args.rounds
        ].detach().cpu().tolist()
        return [bytes(int(x) for x in row[:32]).hex() for row in rows]

    if args.passes > 0:
        if sampled_pass == 0:
            pass_started = time.perf_counter()
            captured = _run_merkle_pass_with_optional_capture(
                ext=ext,
                ws=ws,
                seed=seed,
                pass_index=0,
                rounds=args.rounds,
                block_size=block_size,
                lease_id=args.lease_id,
                gpu_index=args.gpu_index,
                challenge_seed=challenge_seed,
                capture_round=sampled_round,
                memory_mix_rounds=args.memory_mix_rounds,
            )
            torch.cuda.synchronize()
            pass_ms.append((time.perf_counter() - pass_started) * 1000.0)
        else:
            if args.memory_mix_rounds > 0:
                ms = ext.run_workspace_pass_committed_mixed(
                    ws.a, ws.b, ws.q, ws.partial, ws.root, ws.round_roots,
                    seed, 0, args.rounds, block_size, args.memory_mix_rounds, 0,
                )
            else:
                ms = ext.run_workspace_pass_committed(
                    ws.a, ws.b, ws.q, ws.partial, ws.root, ws.round_roots,
                    seed, 0, args.rounds, block_size, 0,
                )
            torch.cuda.synchronize()
            ws.roots[0].copy_(ws.root)
            pass_ms.append(float(ms))

        pass0_round_roots = _pass_round_roots(0)
        pass0_root = pass_root_from_round_roots(0, pass0_round_roots)
        (out_dir / f"{args.lease_id}_pass0.json").write_text(json.dumps({
            "lease_id": args.lease_id,
            "pass_index": 0,
            "root": pass0_root,
            "round_roots": pass0_round_roots,
            "t": time.time(),
        }))

    if args.passes > 0 and sampled_pass > 1:
        pre_passes = sampled_pass - 1
        if args.memory_mix_rounds > 0:
            chain_ms = ext.run_workspace_chain_committed_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, 1, pre_passes, args.rounds, block_size, args.memory_mix_rounds, 0,
            )
        else:
            chain_ms = ext.run_workspace_chain_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, 1, pre_passes, args.rounds, block_size, 0,
            )
        torch.cuda.synchronize()
        pass_ms.extend([float(chain_ms) / float(pre_passes)] * int(pre_passes))

    if args.passes > 0 and sampled_pass > 0:
        pass_started = time.perf_counter()
        captured = _run_merkle_pass_with_optional_capture(
            ext=ext,
            ws=ws,
            seed=seed,
            pass_index=sampled_pass,
            rounds=args.rounds,
            block_size=block_size,
            lease_id=args.lease_id,
            gpu_index=args.gpu_index,
            challenge_seed=challenge_seed,
            capture_round=sampled_round,
            memory_mix_rounds=args.memory_mix_rounds,
        )
        torch.cuda.synchronize()
        pass_ms.append((time.perf_counter() - pass_started) * 1000.0)

    if args.passes > 0 and sampled_pass + 1 < args.passes:
        post_start = sampled_pass + 1
        post_passes = args.passes - post_start
        if args.memory_mix_rounds > 0:
            chain_ms = ext.run_workspace_chain_committed_mixed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, post_start, post_passes, args.rounds, block_size, args.memory_mix_rounds, 0,
            )
        else:
            chain_ms = ext.run_workspace_chain_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.roots, ws.round_roots,
                seed, post_start, post_passes, args.rounds, block_size, 0,
            )
        torch.cuda.synchronize()
        pass_ms.extend([float(chain_ms) / float(post_passes)] * int(post_passes))

    root_chain: list[dict[str, Any]] = []
    if args.passes > 0:
        round_root_rows = ws.round_roots[:args.passes * args.rounds].detach().cpu().tolist()
        for pass_index in range(args.passes):
            start = pass_index * args.rounds
            round_roots = [
                bytes(int(x) for x in round_root_rows[start + round_index][:32]).hex()
                for round_index in range(args.rounds)
            ]
            pass_root = pass_root_from_round_roots(pass_index, round_roots)
            root_chain.append({
                "pass_index": pass_index,
                "pass_root": pass_root,
                "round_roots": round_roots,
            })

    if captured is None and args.passes > 0:
        raise RuntimeError("public challenge capture did not produce sampled proof material")

    final_root = root_chain[-1]["pass_root"] if root_chain else ""
    pass0_root = root_chain[0]["pass_root"] if root_chain else ""
    final_transcript = workspace_transcript_root(root_chain) if root_chain else ""
    timed_elapsed_s = time.perf_counter() - started
    (out_dir / f"{args.lease_id}_final_timing.json").write_text(json.dumps({
        "lease_id": args.lease_id,
        "passes": args.passes,
        "root": final_root,
        "pass0_root": pass0_root,
        "transcript_root": final_transcript,
        "proof_format": proof_format_for_mix(args.memory_mix_rounds),
        "proof_challenge_seed": challenge_seed,
        "proof_challenge_public": True,
        "t": time.time(),
    }))

    proof_payload = {}
    proof_payload_s = 0.0
    if root_chain and captured is not None:
        proof_started = time.perf_counter()
        sampled = _build_block_proof_from_capture(
            captured=captured,
            matrix_dim=n,
            block_size=block_size,
            spot_checks=args.spot_checks,
        )
        proof_payload_s = time.perf_counter() - proof_started
        sampled_round_root = root_chain[sampled_pass]["round_roots"][sampled_round]
        if sampled_round_root != captured.round_root:
            raise RuntimeError(
                f"captured sampled round root mismatch: {captured.round_root} != {sampled_round_root}"
            )
        proof_payload = {
            "format": proof_format_for_mix(args.memory_mix_rounds),
            "workload_version": workload_version_for_mix(args.memory_mix_rounds),
            "memory_mix_rounds": int(args.memory_mix_rounds),
            "proof_challenge_seed": challenge_seed,
            "proof_challenge_public": True,
            "matrix_dim": n,
            "block_size": block_size,
            "rounds": args.rounds,
            "passes": args.passes,
            "spot_checks": args.spot_checks,
            "workspace_seed": int(seed),
            "root_chain": root_chain,
            "sampled": {
                "pass_index": sampled_pass,
                "round_index": sampled_round,
                "round_root": sampled_round_root,
                "block_index": sampled["block_index"],
            },
            "block_proof": sampled["block_proof"],
        }

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final = {
        "proof_mode": "merkle",
        "workload_version": workload_version_for_mix(args.memory_mix_rounds),
        "memory_mix_rounds": int(args.memory_mix_rounds),
        "proof_capture": "public_challenge_no_replay",
        "lease_id": args.lease_id,
        "gpu_index": args.gpu_index,
        "matrix_dim": n,
        "workspace_mib_target": args.workspace_mib,
        "workspace_bytes_nominal": 6 * n * n + 8 * ((n * n + 255) // 256) + 32,
        "passes": args.passes,
        "rounds": args.rounds,
        "block_size": block_size,
        "spot_checks": args.spot_checks,
        "elapsed_s": time.perf_counter() - started,
        "timed_elapsed_s": timed_elapsed_s,
        "mean_pass_ms": sum(pass_ms) / len(pass_ms) if pass_ms else 0.0,
        "p50_pass_ms": sorted(pass_ms)[len(pass_ms) // 2] if pass_ms else 0.0,
        "peak_allocated_mib": peak_alloc / (1024 * 1024),
        "peak_reserved_mib": peak_reserved / (1024 * 1024),
        "pass0_root": pass0_root,
        "final_root": final_root,
        "root_chain": root_chain,
        "timing_transcript_root": final_transcript,
        "proof_payload": proof_payload,
        "proof_payload_s": proof_payload_s,
    }
    (out_dir / f"{args.lease_id}_final.json").write_text(json.dumps(final, indent=2))


def _child_transition(args: argparse.Namespace, ext, ws: Workspace, seed: int, out_dir: pathlib.Path) -> None:
    if int(args.memory_mix_rounds) != 0:
        raise ValueError("--transition-mix-rounds and --memory-mix-rounds are mutually exclusive")

    pass_ms: list[float] = []
    block_size = int(args.block_size)
    n = int(ws.a.size(0))
    transition_mix_rounds = int(args.transition_mix_rounds)
    transition_fanout = int(args.transition_fanout)
    if transition_fanout <= 0:
        raise ValueError("--transition-fanout must be > 0")
    warmup_passes = max(0, int(args.warmup_passes))
    allocated_passes = max(1, int(args.passes), warmup_passes)
    state_roots = torch.empty(
        (allocated_passes * args.rounds * (transition_mix_rounds + 1), 32),
        dtype=torch.uint8,
        device=ws.a.device,
    )

    warmup_ms = 0.0
    warmup_wall_s = 0.0
    if warmup_passes:
        torch.cuda.synchronize()
        warmup_started = time.perf_counter()
        for warmup_pass in range(warmup_passes):
            warmup_ms += float(ext.run_workspace_pass_transition_committed(
                ws.a,
                ws.b,
                ws.q,
                ws.partial,
                ws.root,
                state_roots,
                ws.round_roots,
                seed ^ 0xA5A55A5A12345678,
                warmup_pass,
                args.rounds,
                block_size,
                transition_mix_rounds,
                transition_fanout,
                0,
            ))
        torch.cuda.synchronize()
        warmup_wall_s = time.perf_counter() - warmup_started

    torch.cuda.synchronize()
    started = time.perf_counter()
    for pass_index in range(int(args.passes)):
        ms = ext.run_workspace_pass_transition_committed(
            ws.a,
            ws.b,
            ws.q,
            ws.partial,
            ws.root,
            state_roots,
            ws.round_roots,
            seed,
            pass_index,
            args.rounds,
            block_size,
            transition_mix_rounds,
            transition_fanout,
            0,
        )
        torch.cuda.synchronize()
        pass_ms.append(float(ms))
        if pass_index == 0:
            pass0_round_roots = [
                _tensor_u8_hex(ws.round_roots[round_index])
                for round_index in range(args.rounds)
            ]
            pass0_root = pass_root_from_round_roots(0, pass0_round_roots)
            (out_dir / f"{args.lease_id}_pass0.json").write_text(json.dumps({
                "lease_id": args.lease_id,
                "pass_index": 0,
                "root": pass0_root,
                "round_roots": pass0_round_roots,
                "t": time.time(),
            }))

    root_chain: list[dict[str, Any]] = []
    state_root_chain: list[dict[str, Any]] = []
    if args.passes > 0:
        round_root_rows = ws.round_roots[:args.passes * args.rounds].detach().cpu().tolist()
        state_root_rows = state_roots[
            :args.passes * args.rounds * (transition_mix_rounds + 1)
        ].detach().cpu().tolist()
        for pass_index in range(args.passes):
            round_roots = []
            round_state_roots = []
            for round_index in range(args.rounds):
                round_offset = pass_index * args.rounds + round_index
                round_roots.append(
                    bytes(int(x) for x in round_root_rows[round_offset][:32]).hex()
                )
                state_offset = round_offset * (transition_mix_rounds + 1)
                round_state_roots.append([
                    bytes(int(x) for x in state_root_rows[state_offset + mix_idx][:32]).hex()
                    for mix_idx in range(transition_mix_rounds + 1)
                ])
            root_chain.append({
                "pass_index": pass_index,
                "pass_root": pass_root_from_round_roots(pass_index, round_roots),
                "round_roots": round_roots,
            })
            state_root_chain.append({
                "pass_index": pass_index,
                "round_state_roots": round_state_roots,
            })

    final_root = root_chain[-1]["pass_root"] if root_chain else ""
    pass0_root = root_chain[0]["pass_root"] if root_chain else ""
    final_transcript = workspace_transcript_root(root_chain) if root_chain else ""
    timed_elapsed_s = time.perf_counter() - started
    timed_cuda_component_s = sum(pass_ms) / 1000.0
    (out_dir / f"{args.lease_id}_final_timing.json").write_text(json.dumps({
        "lease_id": args.lease_id,
        "passes": args.passes,
        "root": final_root,
        "pass0_root": pass0_root,
        "transcript_root": final_transcript,
        "proof_format": WORKSPACE_TRANSITION_PROOF_FORMAT,
        "transition_mix_rounds": transition_mix_rounds,
        "transition_fanout": transition_fanout,
        "t": time.time(),
    }))

    proof_payload: dict[str, Any] = {}
    proof_payload_s = 0.0
    challenge_seed, challenge_is_public = (
        ("", False) if args.skip_proof_payload else _resolve_challenge_seed(args, final_transcript)
    )
    if root_chain and challenge_seed:
        proof_started = time.perf_counter()
        sampled_pass = derive_sampled_pass_index(challenge_seed, args.lease_id, args.gpu_index, args.passes)
        sampled_round = derive_sampled_round_index(
            challenge_seed,
            args.lease_id,
            args.gpu_index,
            sampled_pass,
            args.rounds,
        )
        sampled_mix = derive_sampled_transition_mix_index(
            challenge_seed,
            args.lease_id,
            args.gpu_index,
            sampled_pass,
            sampled_round,
            transition_mix_rounds,
        )
        sampled_round_root = root_chain[sampled_pass]["round_roots"][sampled_round]
        sampled_state_roots = state_root_chain[sampled_pass]["round_state_roots"][sampled_round]

        sampled_state_tensor = torch.empty(
            (transition_mix_rounds + 1, 32),
            dtype=torch.uint8,
            device=ws.a.device,
        )
        sampled_round_tensor = torch.empty((1, 32), dtype=torch.uint8, device=ws.a.device)
        ext.run_workspace_round_transition_committed(
            ws.a,
            ws.b,
            ws.q,
            ws.partial,
            ws.root,
            sampled_state_tensor,
            sampled_round_tensor,
            seed,
            sampled_pass,
            sampled_round,
            block_size,
            transition_mix_rounds,
            transition_fanout,
            0,
        )
        torch.cuda.synchronize()
        recomputed_round_root = _tensor_u8_hex(sampled_round_tensor[0])
        if recomputed_round_root != sampled_round_root:
            raise RuntimeError(f"sampled transition round root mismatch: {recomputed_round_root} != {sampled_round_root}")
        recomputed_state_roots = [
            _tensor_u8_hex(sampled_state_tensor[idx])
            for idx in range(transition_mix_rounds + 1)
        ]
        if recomputed_state_roots != sampled_state_roots:
            raise RuntimeError("sampled transition state root mismatch")

        sampled_block = _build_resident_challenged_block_proof(
            ext=ext,
            ws=ws,
            round_root=sampled_round_root,
            pass_index=sampled_pass,
            round_index=sampled_round,
            lease_id=args.lease_id,
            gpu_index=args.gpu_index,
            matrix_dim=n,
            block_size=block_size,
            spot_checks=args.spot_checks,
            challenge_seed_hex=challenge_seed,
        )
        final_state_root, final_tree = _state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
        expected_final_state_root = sampled_state_roots[-1]
        if final_state_root != expected_final_state_root:
            raise RuntimeError(
                f"sampled final state root mismatch: {final_state_root} != {expected_final_state_root}"
            )
        gemm_state_blocks = {
            _state_block_index_for_element(int(spot["row"]) * n + int(spot["col"]), n, block_size)
            for spot in sampled_block["block_proof"]["spot_A"]
        }
        gemm_state_blocks.update(
            _state_block_index_for_element(int(spot["row"]) * n + int(spot["col"]), n, block_size)
            for spot in sampled_block["block_proof"]["spot_B"]
        )
        gemm_state_openings = [
            _state_opening_from_tensors(ws.a, ws.b, final_tree, bidx, n, block_size)
            for bidx in sorted(gemm_state_blocks)
        ]

        ext.run_workspace_round_transition_state(
            ws.a,
            ws.b,
            ws.q,
            seed,
            sampled_pass,
            sampled_round,
            transition_mix_rounds,
            transition_fanout,
            sampled_mix,
            0,
        )
        torch.cuda.synchronize()
        before_root = sampled_state_roots[sampled_mix]
        before_state_root, before_tree = _state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
        if before_state_root != before_root:
            raise RuntimeError(f"sampled before state root mismatch: {before_state_root} != {before_root}")
        ext.run_workspace_round_transition_state(
            ws.a,
            ws.b,
            ws.q,
            seed,
            sampled_pass,
            sampled_round,
            transition_mix_rounds,
            transition_fanout,
            sampled_mix + 1,
            0,
        )
        torch.cuda.synchronize()
        after_root = sampled_state_roots[sampled_mix + 1]
        after_state_root, after_tree = _state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
        if after_state_root != after_root:
            raise RuntimeError(f"sampled after state root mismatch: {after_state_root} != {after_root}")
        element_index = derive_transition_element_index(
            before_root_hex=before_root,
            after_root_hex=after_root,
            lease_id=args.lease_id,
            gpu_index=args.gpu_index,
            pass_index=sampled_pass,
            round_index=sampled_round,
            mix_index=sampled_mix,
            matrix_dim=n,
            challenge_seed_hex=challenge_seed,
        )
        source_pairs = transition_source_indices(
            workspace_seed=seed,
            pass_index=sampled_pass,
            round_index=sampled_round,
            mix_index=sampled_mix,
            element_index=element_index,
            matrix_dim=n,
            fanout=transition_fanout,
        )
        transition_before_blocks = {
            _state_block_index_for_element(element_index, n, block_size),
        }
        for j, k in source_pairs:
            transition_before_blocks.add(_state_block_index_for_element(j, n, block_size))
            transition_before_blocks.add(_state_block_index_for_element(k, n, block_size))
        transition_after_blocks = {
            _state_block_index_for_element(element_index, n, block_size),
        }
        ext.run_workspace_round_transition_state(
            ws.a,
            ws.b,
            ws.q,
            seed,
            sampled_pass,
            sampled_round,
            transition_mix_rounds,
            transition_fanout,
            sampled_mix,
            0,
        )
        torch.cuda.synchronize()
        transition_before_openings = [
            _state_opening_from_tensors(ws.a, ws.b, before_tree, bidx, n, block_size)
            for bidx in sorted(transition_before_blocks)
        ]
        ext.run_workspace_round_transition_state(
            ws.a,
            ws.b,
            ws.q,
            seed,
            sampled_pass,
            sampled_round,
            transition_mix_rounds,
            transition_fanout,
            sampled_mix + 1,
            0,
        )
        torch.cuda.synchronize()
        transition_after_openings = [
            _state_opening_from_tensors(ws.a, ws.b, after_tree, bidx, n, block_size)
            for bidx in sorted(transition_after_blocks)
        ]
        proof_payload = {
            "format": WORKSPACE_TRANSITION_PROOF_FORMAT,
            "workload_version": workload_version_for_args(args),
            "memory_mix_rounds": 0,
            "transition_mix_rounds": transition_mix_rounds,
            "transition_fanout": transition_fanout,
            "proof_challenge_seed": challenge_seed,
            "proof_challenge_public": bool(challenge_is_public),
            "matrix_dim": n,
            "block_size": block_size,
            "rounds": args.rounds,
            "passes": args.passes,
            "spot_checks": args.spot_checks,
            "workspace_seed": int(seed),
            "root_chain": root_chain,
            "state_root_chain": state_root_chain,
            "sampled": {
                "pass_index": sampled_pass,
                "round_index": sampled_round,
                "round_root": sampled_round_root,
                "block_index": sampled_block["block_index"],
            },
            "block_proof": sampled_block["block_proof"],
            "gemm_state_openings": gemm_state_openings,
            "sampled_transition": {
                "pass_index": sampled_pass,
                "round_index": sampled_round,
                "mix_index": sampled_mix,
                "before_root": before_root,
                "after_root": after_root,
                "opening": {
                    "element_index": element_index,
                    "before_blocks": transition_before_openings,
                    "after_blocks": transition_after_openings,
                },
            },
        }
        proof_payload_s = time.perf_counter() - proof_started

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    final = {
        "proof_mode": "merkle",
        "workload_version": workload_version_for_args(args),
        "memory_mix_rounds": 0,
        "transition_mix_rounds": transition_mix_rounds,
        "transition_fanout": transition_fanout,
        "lease_id": args.lease_id,
        "gpu_index": args.gpu_index,
        "matrix_dim": n,
        "workspace_mib_target": args.workspace_mib,
        "workspace_bytes_nominal": 6 * n * n + 8 * ((n * n + 255) // 256) + 32,
        "passes": args.passes,
        "rounds": args.rounds,
        "block_size": block_size,
        "spot_checks": args.spot_checks,
        "elapsed_s": time.perf_counter() - started,
        "timed_elapsed_s": timed_elapsed_s,
        "timed_cuda_component_s": timed_cuda_component_s,
        "warmup_passes": warmup_passes,
        "warmup_timed_s": warmup_ms / 1000.0,
        "warmup_wall_s": warmup_wall_s,
        "mean_pass_ms": sum(pass_ms) / len(pass_ms) if pass_ms else 0.0,
        "p50_pass_ms": sorted(pass_ms)[len(pass_ms) // 2] if pass_ms else 0.0,
        "peak_allocated_mib": peak_alloc / (1024 * 1024),
        "peak_reserved_mib": peak_reserved / (1024 * 1024),
        "pass0_root": pass0_root,
        "final_root": final_root,
        "root_chain": root_chain,
        "state_root_chain": state_root_chain,
        "timing_transcript_root": final_transcript,
        "proof_payload": proof_payload,
        "proof_payload_s": proof_payload_s,
    }
    (out_dir / f"{args.lease_id}_final.json").write_text(json.dumps(final, indent=2))


def child(args: argparse.Namespace) -> None:
    ext = _import_ext()
    torch.cuda.set_device(0)
    device = "cuda:0"
    n = args.matrix_dim or matrix_dim_for_workspace_mib(args.workspace_mib)
    ws = allocate_workspace(n, args.passes, args.rounds, device)
    torch.cuda.synchronize()
    pre_timing_used = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = seed_for(args.lease_id, args.gpu_index, args.seed_hex)
    if args.ready_file:
        pathlib.Path(args.ready_file).write_text(json.dumps({
            "lease_id": args.lease_id,
            "gpu_index": args.gpu_index,
            "t": time.time(),
            "pre_timing_used_bytes": int(pre_timing_used),
        }))
    if args.start_file:
        start_path = pathlib.Path(args.start_file)
        deadline = time.perf_counter() + max(0.0, float(args.start_timeout_s))
        while not start_path.exists():
            if time.perf_counter() > deadline:
                raise TimeoutError(f"timed out waiting for hot-start file: {start_path}")
            time.sleep(0.01)
    if args.proof_mode == "compact":
        _child_compact(args, ext, ws, seed, out_dir)
    elif int(args.transition_mix_rounds) > 0:
        _child_transition(args, ext, ws, seed, out_dir)
    else:
        _child_merkle(args, ext, ws, seed, out_dir)


def parent(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir or tempfile.mkdtemp(prefix="verathos_fixed_workspace_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()] or ["0"]
    cmd_base = [
        *_self_command(),
        "--child",
        "--workspace-mib", str(args.workspace_mib),
        "--matrix-dim", str(args.matrix_dim),
        "--passes", str(args.passes),
        "--rounds", str(args.rounds),
        "--block-size", str(args.block_size),
        "--spot-checks", str(args.spot_checks),
        "--proof-mode", str(args.proof_mode),
        "--memory-mix-rounds", str(args.memory_mix_rounds),
        "--transition-mix-rounds", str(args.transition_mix_rounds),
        "--transition-fanout", str(args.transition_fanout),
        "--warmup-passes", str(args.warmup_passes),
        "--skip-proof-payload" if args.skip_proof_payload else "--no-skip-proof-payload",
        "--challenge-hex", str(args.challenge_hex),
        "--out-dir", str(out_dir),
    ]
    procs = []
    challenge_files: dict[str, pathlib.Path] = {}
    challenge_due: dict[str, float] = {}
    challenge_written: set[str] = set()
    ready_files: dict[str, pathlib.Path] = {}
    verifier_preload: dict[str, Any] = {}
    if args.verify_proof:
        verifier_preload = preload_workspace_proof_verifier()
    started = time.perf_counter()
    start_file = out_dir / "hot_start_go.json"
    for i in range(args.leases):
        lease_id = f"lease-{i:03d}"
        device = devices[0] if args.same_gpu else devices[i % len(devices)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONPATH"] = f"{SCRIPT_DIR}:{env.get('PYTHONPATH', '')}"
        challenge_file = str(args.challenge_file)
        if args.simulate_b_proof_delay_s >= 0 and not args.challenge_hex:
            challenge_path = out_dir / f"{lease_id}_b_proof_challenge.txt"
            challenge_file = str(challenge_path)
            challenge_files[lease_id] = challenge_path
        ready_file = out_dir / f"{lease_id}_ready.json"
        if args.hot_start_barrier:
            ready_files[lease_id] = ready_file
        cmd = cmd_base + [
            "--lease-id", lease_id,
            "--gpu-index", str(i),
            "--seed-hex", args.seed_hex,
            "--challenge-file", challenge_file,
            "--challenge-timeout-s", str(args.challenge_timeout_s),
        ]
        if args.hot_start_barrier:
            cmd += [
                "--ready-file", str(ready_file),
                "--start-file", str(start_file),
                "--start-timeout-s", str(args.hot_start_timeout_s),
            ]
        procs.append((lease_id, subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))

    if args.hot_start_barrier:
        pending_ready = set(ready_files)
        ready_deadline = time.perf_counter() + max(0.0, float(args.hot_start_timeout_s))
        while pending_ready:
            for lease_id in list(pending_ready):
                if ready_files[lease_id].exists():
                    pending_ready.remove(lease_id)
            for lease_id, proc in procs:
                if lease_id in pending_ready and proc.poll() is not None:
                    pending_ready.remove(lease_id)
            if not pending_ready or time.perf_counter() > ready_deadline:
                break
            time.sleep(0.01)
        start_file.write_text(json.dumps({
            "t": time.time(),
            "ready": sorted(set(ready_files) - pending_ready),
            "missing_ready": sorted(pending_ready),
        }))
        started = time.perf_counter()

    pass0_recv: dict[str, float] = {}
    final_recv: dict[str, float] = {}
    pending_pass0 = {lease_id for lease_id, _ in procs}
    pending_final = {lease_id for lease_id, _ in procs}
    pending_challenge = set(challenge_files)
    while pending_pass0 or pending_final or pending_challenge:
        now = time.perf_counter() - started
        for lease_id in list(pending_pass0):
            if (out_dir / f"{lease_id}_pass0.json").exists():
                pass0_recv[lease_id] = now
                pending_pass0.remove(lease_id)
        for lease_id in list(pending_final):
            if (out_dir / f"{lease_id}_final_timing.json").exists():
                final_recv[lease_id] = now
                if lease_id in challenge_files:
                    challenge_due[lease_id] = now + max(0.0, float(args.simulate_b_proof_delay_s))
                pending_final.remove(lease_id)
        for lease_id in list(pending_challenge):
            due = challenge_due.get(lease_id)
            if due is not None and now >= due:
                digest = hashlib.sha256(
                    b"VERATHOS_BENCH_SIMULATED_B_PROOF_V1"
                    + lease_id.encode()
                    + str(final_recv.get(lease_id, 0.0)).encode()
                ).hexdigest()
                challenge_files[lease_id].write_text(digest)
                challenge_written.add(lease_id)
                pending_challenge.remove(lease_id)
        if pending_pass0 or pending_final or pending_challenge:
            if all(proc.poll() is not None for _, proc in procs):
                break
            time.sleep(0.01)

    children: list[dict[str, Any]] = []
    for lease_id, proc in procs:
        stdout, stderr = proc.communicate(timeout=args.timeout_s)
        final_path = out_dir / f"{lease_id}_final.json"
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        verify_ok = None
        verify_reason = ""
        verify_s = 0.0
        if args.verify_proof and final.get("proof_payload"):
            verify_started = time.perf_counter()
            if str(final.get("proof_mode")) == "merkle":
                verify_ok, verify_reason = verify_workspace_proof_payload(
                    proof=final["proof_payload"],
                    expected_transcript_root=str(final.get("timing_transcript_root") or ""),
                    expected_pass0_root=str(final.get("pass0_root") or ""),
                    expected_final_root=str(final.get("final_root") or ""),
                    lease_id=str(final.get("lease_id") or lease_id),
                    gpu_index=int(final.get("gpu_index") or 0),
                    proof_seed_hex=str(args.seed_hex or ""),
                    pass_count=int(final.get("passes") or 0),
                    proof_challenge_seed_hex=str(
                        (final.get("proof_payload") or {}).get("proof_challenge_seed") or ""
                    ),
                )
            else:
                verify_ok = False
                verify_reason = "verify_proof_unsupported_mode"
            verify_s = time.perf_counter() - verify_started
        summary = dict(final)
        root_chain = summary.get("root_chain")
        if isinstance(root_chain, list) and len(root_chain) > 4:
            summary["root_chain"] = {
                "passes": len(root_chain),
                "first": root_chain[0],
                "last": root_chain[-1],
            }
        state_root_chain = summary.get("state_root_chain")
        if isinstance(state_root_chain, list) and len(state_root_chain) > 4:
            summary["state_root_chain"] = {
                "passes": len(state_root_chain),
                "first": state_root_chain[0],
                "last": state_root_chain[-1],
            }
        proof_payload = summary.get("proof_payload")
        if isinstance(proof_payload, dict):
            summary["proof_payload"] = {
                "format": proof_payload.get("format"),
                "sampled": proof_payload.get("sampled"),
                "json_bytes": len(json.dumps(proof_payload, separators=(",", ":"))),
            }
        children.append({
            "lease_id": lease_id,
            "returncode": proc.returncode,
            "pass0_recv_s": round(pass0_recv.get(lease_id, 0.0), 4),
            "final_timing_recv_s": round(final_recv.get(lease_id, 0.0), 4),
            "simulated_b_proof_written": lease_id in challenge_written,
            "external_delta_s": round(final_recv.get(lease_id, 0.0) - pass0_recv.get(lease_id, 0.0), 4),
            "verify_ok": verify_ok,
            "verify_reason": verify_reason,
            "verify_s": round(verify_s, 6),
            "summary": summary,
            "stderr_tail": stderr[-1000:],
            "stdout_tail": stdout[-500:],
        })

    result = {
        "case": "same_gpu" if args.same_gpu else "device_round_robin",
        "out_dir": str(out_dir),
        "leases": args.leases,
        "workspace_mib": args.workspace_mib,
        "matrix_dim": args.matrix_dim or matrix_dim_for_workspace_mib(args.workspace_mib),
        "passes": args.passes,
        "rounds": args.rounds,
        "memory_mix_rounds": args.memory_mix_rounds,
        "transition_mix_rounds": args.transition_mix_rounds,
        "total_wall_s": round(time.perf_counter() - started, 4),
        "hot_start_barrier": bool(args.hot_start_barrier),
        "verifier_preload": verifier_preload,
        "children": children,
    }
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--out-dir", default="")
    p.add_argument("--lease-id", default="lease-000")
    p.add_argument("--seed-hex", default="")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--ready-file", default="")
    p.add_argument("--start-file", default="")
    p.add_argument("--start-timeout-s", type=float, default=60.0)
    p.add_argument("--devices", default="0")
    p.add_argument("--same-gpu", action="store_true")
    p.add_argument("--hot-start-barrier", action="store_true")
    p.add_argument("--hot-start-timeout-s", type=float, default=60.0)
    p.add_argument("--leases", type=int, default=1)
    p.add_argument("--workspace-mib", type=float, default=500.0)
    p.add_argument("--matrix-dim", type=int, default=0)
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument(
        "--warmup-passes",
        type=int,
        default=1,
        help="hot calibration: run this many untimed passes before measured work",
    )
    p.add_argument(
        "--memory-mix-rounds",
        type=int,
        default=0,
        help="prototype v2: deterministic streaming A/B memory-mix rounds before each GEMM round",
    )
    p.add_argument(
        "--transition-mix-rounds",
        type=int,
        default=0,
        help="prototype v3: transition-dependent A/B state mix rounds before each GEMM round",
    )
    p.add_argument(
        "--transition-fanout",
        type=int,
        default=1,
        help="prototype v3: number of transition source pairs per output element",
    )
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--spot-checks", type=int, default=1)
    p.add_argument("--proof-mode", choices=("compact", "merkle"), default="merkle")
    p.add_argument("--challenge-hex", default="")
    p.add_argument("--challenge-file", default="")
    p.add_argument("--challenge-timeout-s", type=float, default=0.0)
    p.add_argument("--verify-proof", action="store_true")
    p.add_argument("--skip-proof-payload", dest="skip_proof_payload", action="store_true")
    p.add_argument("--no-skip-proof-payload", dest="skip_proof_payload", action="store_false")
    p.set_defaults(skip_proof_payload=False)
    p.add_argument(
        "--simulate-b-proof-delay-s",
        type=float,
        default=-1.0,
        help="parent benchmark only: write a synthetic B_proof challenge file after final_timing",
    )
    p.add_argument("--timeout-s", type=float, default=600.0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.child:
        child(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
