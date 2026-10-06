#!/usr/bin/env python3
"""Combined hot-capacity audit runtime harness.

This local harness fuses the v3 capacity lane and FP64 server-identity lane into
one audit transcript:

1. Run timed v3 transition/fanout capacity work and timed FP64 identity work.
2. Commit both root chains before any sampled opening is known.
3. Derive one sample seed from post-timing B_proof-like randomness plus the
   combined transcript.
4. Build and verify sampled proofs for both lanes.

The miner worker uses this harness to execute one audit recipe and emit the
pass0, final timing, and sampled proof payloads consumed by validators.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import secrets
import sys
import time
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

from verathos.capacity.bench import bench_fp64_identity as fp64  # noqa: E402
from verathos.capacity.bench import bench_workspace as cap  # noqa: E402
from verathos.capacity.proof.workspace import (  # noqa: E402
    WORKSPACE_PROOF_FORMAT,
    WORKSPACE_TRANSITION_PROOF_FORMAT,
    derive_sampled_pass_index,
    derive_sampled_round_index,
    derive_sampled_transition_mix_index,
    derive_transition_element_index,
    serialize_merkle_path,
    transition_source_indices,
    verify_workspace_proof_payload,
    workspace_transcript_root_v2,
)
from verathos.capacity.proof.combined import (  # noqa: E402
    COMBINED_PROOF_FORMAT_V1,
    COMBINED_PROOF_FORMAT_V2,
    CURRENT_COMBINED_PROOF_PROTOCOL_VERSION,
    LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
    combined_commitment_from_final_timing,
    verify_combined_proof_payload,
)
from verathos.capacity.proof.gemm_v2 import (  # noqa: E402
    capacity_gemm_v2_context_from_workspace_proof,
    prove_capacity_gemm_v2,
    required_transition_state_blocks,
)


WORKLOAD_VERSION = "hot_capacity_combined"
# Retained for legacy callers that import the harness constant.
COMBINED_PROOF_FORMAT = COMBINED_PROOF_FORMAT_V1


def _combined_proof_format(proof_protocol_version: int) -> str:
    if int(proof_protocol_version) == LEGACY_COMBINED_PROOF_PROTOCOL_VERSION:
        return COMBINED_PROOF_FORMAT_V1
    if int(proof_protocol_version) == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
        return COMBINED_PROOF_FORMAT_V2
    raise ValueError("unsupported combined proof protocol version")


def _sha256_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _capacity_root_chains(
    *,
    round_roots: torch.Tensor,
    state_roots: torch.Tensor,
    passes: int,
    rounds: int,
    transition_mix_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    round_root_rows = round_roots[:passes * rounds].detach().cpu().tolist()
    state_root_rows = state_roots[:passes * rounds * (transition_mix_rounds + 1)].detach().cpu().tolist()
    root_chain: list[dict[str, Any]] = []
    state_root_chain: list[dict[str, Any]] = []
    for pass_index in range(passes):
        pass_round_roots: list[str] = []
        round_state_roots: list[list[str]] = []
        for round_index in range(rounds):
            round_offset = pass_index * rounds + round_index
            pass_round_roots.append(bytes(int(x) for x in round_root_rows[round_offset][:32]).hex())
            state_offset = round_offset * (transition_mix_rounds + 1)
            round_state_roots.append([
                bytes(int(x) for x in state_root_rows[state_offset + mix_idx][:32]).hex()
                for mix_idx in range(transition_mix_rounds + 1)
            ])
        root_chain.append({
            "pass_index": pass_index,
            "pass_root": cap.pass_root_from_round_roots(pass_index, pass_round_roots),
            "round_roots": pass_round_roots,
        })
        state_root_chain.append({
            "pass_index": pass_index,
            "round_state_roots": round_state_roots,
        })
    return root_chain, state_root_chain


def _capacity_round_root_chain(
    *,
    round_roots: torch.Tensor,
    passes: int,
    rounds: int,
) -> list[dict[str, Any]]:
    round_root_rows = round_roots[:passes * rounds].detach().cpu().tolist()
    root_chain: list[dict[str, Any]] = []
    for pass_index in range(passes):
        pass_round_roots: list[str] = []
        for round_index in range(rounds):
            round_offset = pass_index * rounds + round_index
            pass_round_roots.append(bytes(int(x) for x in round_root_rows[round_offset][:32]).hex())
        root_chain.append({
            "pass_index": pass_index,
            "pass_root": cap.pass_root_from_round_roots(pass_index, pass_round_roots),
            "round_roots": pass_round_roots,
        })
    return root_chain


def combined_transcript_root(
    *,
    capacity_transcript: str,
    capacity_tail_transcript: str,
    fp64_transcript: str,
    capacity_params: dict[str, Any],
    capacity_tail_params: dict[str, Any] | None,
    fp64_params: dict[str, Any],
    proof_protocol_version: int = LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
) -> str:
    proof_protocol_version = int(proof_protocol_version)
    version_fields: dict[str, Any] = {}
    transcript_domain = b"VERATHOS_HOT_CAPACITY_COMBINED_TRANSCRIPT_V1"
    if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
        version_fields["proof_protocol_version"] = proof_protocol_version
        transcript_domain = b"VERATHOS_HOT_CAPACITY_COMBINED_TRANSCRIPT_V2"
    elif proof_protocol_version != LEGACY_COMBINED_PROOF_PROTOCOL_VERSION:
        raise ValueError("unsupported combined proof protocol version")
    payload = json.dumps(
        {
            "version": WORKLOAD_VERSION,
            "capacity_transcript": capacity_transcript,
            "capacity_tail_transcript": capacity_tail_transcript,
            "fp64_transcript": fp64_transcript,
            "capacity_params": capacity_params,
            "capacity_tail_params": capacity_tail_params,
            "fp64_params": fp64_params,
            **version_fields,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return _sha256_hex(transcript_domain, payload)


def sample_seed_from_b_proof(*, b_proof_seed: str, combined_transcript: str) -> str:
    return _sha256_hex(
        b"VERATHOS_HOT_CAPACITY_COMBINED_SAMPLE_V1",
        bytes.fromhex(str(b_proof_seed).removeprefix("0x")),
        bytes.fromhex(str(combined_transcript)),
    )


def _normalize_hex_32(text: str) -> str:
    raw = str(text or "").strip()
    raw = raw[2:] if raw.startswith("0x") else raw
    if len(raw) != 64:
        raise ValueError("expected 32-byte hex challenge")
    bytes.fromhex(raw)
    return raw


def _atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    tmp.replace(path)


def _resolve_b_proof_seed(args: argparse.Namespace) -> tuple[str, str]:
    if str(getattr(args, "b_proof_hex", "") or ""):
        return _normalize_hex_32(str(args.b_proof_hex)), "provided"
    challenge_file_text = str(getattr(args, "challenge_file", "") or "")
    if challenge_file_text:
        challenge_file = pathlib.Path(challenge_file_text)
        deadline = time.perf_counter() + max(0.0, float(getattr(args, "challenge_timeout_s", 0.0) or 0.0))
        last_error = ""
        while time.perf_counter() <= deadline:
            if challenge_file.exists():
                try:
                    return _normalize_hex_32(challenge_file.read_text()), "challenge_file"
                except (OSError, ValueError) as exc:
                    last_error = str(exc)
            time.sleep(0.25)
        if last_error:
            return "", f"invalid_challenge_file:{last_error}"
        return "", "missing_challenge_file"
    if getattr(args, "child", False):
        return "", "missing_child_challenge"
    return secrets.token_hex(32), "local_random_simulated_post_timing"


def _write_ready_and_wait_for_start(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    ready_file = getattr(args, "ready_file", None)
    if ready_file:
        _atomic_write_json(pathlib.Path(ready_file), payload)

    start_file = getattr(args, "start_file", None)
    if not start_file:
        return {}

    start_path = pathlib.Path(start_file)
    deadline = time.perf_counter() + max(0.0, float(getattr(args, "start_timeout_s", 0.0) or 0.0))
    last_error = ""
    while time.perf_counter() <= deadline:
        if not start_path.exists():
            time.sleep(0.01)
            continue
        try:
            data = json.loads(start_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.01)
            continue
        if isinstance(data, dict):
            seed_hex = str(data.get("seed_hex") or data.get("proof_seed") or "").strip()
            if seed_hex:
                args.seed_hex = seed_hex
            return data
        last_error = "start payload is not a JSON object"
        time.sleep(0.01)
        if time.perf_counter() > deadline:
            break
    detail = f" ({last_error})" if last_error else ""
    raise TimeoutError(f"timed out waiting for valid hot-start file: {start_path}{detail}")


def _combined_pass_count(args: argparse.Namespace) -> int:
    return (
        max(0, int(args.capacity_passes))
        + max(0, int(args.capacity_tail_passes))
        + max(0, int(args.fp64_passes))
    )


def _write_child_timing_artifacts(
    *,
    args: argparse.Namespace,
    pass0_root: str,
    final_timing: dict[str, Any],
) -> None:
    out_dir_text = str(getattr(args, "out_dir", "") or "")
    if not out_dir_text:
        return
    out_dir = pathlib.Path(out_dir_text)
    lease = str(args.lease_id)
    _atomic_write_json(
        out_dir / f"{lease}_pass0.json",
        {
            "lease_id": lease,
            "root": pass0_root,
            "pass_index": 0,
            "workload_version": WORKLOAD_VERSION,
            "proof_format": final_timing.get("proof_format"),
            "proof_protocol_version": final_timing.get("proof_protocol_version"),
            "t": time.time(),
        },
    )
    _atomic_write_json(out_dir / f"{lease}_final_timing.json", final_timing)


def _attach_capacity_gemm_v2(
    *,
    proof: dict[str, Any],
    capacity_params: dict[str, Any],
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    challenge_seed: str,
    combined_transcript_root_hex: str,
    state_openings: list[dict[str, Any]],
) -> dict[str, Any]:
    block_proof = proof.get("block_proof")
    if not isinstance(block_proof, dict):
        raise ValueError("workspace block proof is missing")
    context = capacity_gemm_v2_context_from_workspace_proof(
        proof=proof,
        capacity_params=capacity_params,
        lease_id=lease_id,
        gpu_index=gpu_index,
        proof_seed_hex=proof_seed_hex,
        challenge_seed_hex=challenge_seed,
        transcript_root_hex=combined_transcript_root_hex,
    )
    payload = prove_capacity_gemm_v2(
        context=context,
        q_block_i32=block_proof.get("q_block_i32"),
        q_merkle_path=block_proof.get("merkle_path"),
        state_openings=state_openings,
    )
    proof["gemm_v2"] = payload.to_mapping()
    proof["gemm_v2_state_openings"] = state_openings
    return proof


def _transition_band_state_openings(
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    tree: Any,
    block_indices: tuple[int, ...],
    matrix_dim: int,
    block_size: int,
    row_block: int,
    column_block: int,
) -> list[dict[str, Any]]:
    """Copy selected bands once and serialize canonical packed int8 openings."""

    row_start = row_block * block_size
    column_start = column_block * block_size
    a_row = a[row_start : row_start + block_size, :].detach().cpu().contiguous()
    b_row = b[row_start : row_start + block_size, :].detach().cpu().contiguous()
    a_column = (
        a[:, column_start : column_start + block_size]
        .detach()
        .cpu()
        .contiguous()
    )
    b_column = (
        b[:, column_start : column_start + block_size]
        .detach()
        .cpu()
        .contiguous()
    )
    a_row_array = np.asarray(a_row.numpy(), dtype=np.int8)
    b_row_array = np.asarray(b_row.numpy(), dtype=np.int8)
    a_column_array = np.asarray(a_column.numpy(), dtype=np.int8)
    b_column_array = np.asarray(b_column.numpy(), dtype=np.int8)
    blocks_per_row = matrix_dim // block_size
    openings: list[dict[str, Any]] = []
    for block_index in block_indices:
        block_row, block_column = divmod(block_index, blocks_per_row)
        if block_row == row_block:
            local_column = block_column * block_size
            a_block = a_row_array[:, local_column : local_column + block_size]
            b_block = b_row_array[:, local_column : local_column + block_size]
        elif block_column == column_block:
            local_row = block_row * block_size
            a_block = a_column_array[local_row : local_row + block_size, :]
            b_block = b_column_array[local_row : local_row + block_size, :]
        else:
            raise ValueError("state opening is outside the selected v2 bands")
        packed = np.empty((block_size * 2, block_size), dtype=np.uint8)
        packed[0::2, :] = np.asarray(a_block, dtype=np.int8).view(np.uint8)
        packed[1::2, :] = np.asarray(b_block, dtype=np.int8).view(np.uint8)
        openings.append(
            {
                "block_index": block_index,
                "state_block_i8": "base64:"
                + base64.b64encode(packed.tobytes(order="C")).decode("ascii"),
                "merkle_path": serialize_merkle_path(tree.get_path(block_index)),
            }
        )
    return openings


def build_capacity_proof(
    *,
    ext: Any,
    ws: cap.Workspace,
    seed: int,
    root_chain: list[dict[str, Any]],
    state_root_chain: list[dict[str, Any]],
    challenge_seed: str,
    lease_id: str,
    gpu_index: int,
    matrix_dim: int,
    block_size: int,
    passes: int,
    rounds: int,
    spot_checks: int,
    transition_mix_rounds: int,
    transition_fanout: int,
    proof_protocol_version: int = LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
    proof_seed_hex: str = "",
    combined_transcript_root_hex: str = "",
) -> dict[str, Any]:
    sampled_pass = derive_sampled_pass_index(challenge_seed, lease_id, gpu_index, passes)
    sampled_round = derive_sampled_round_index(challenge_seed, lease_id, gpu_index, sampled_pass, rounds)
    sampled_round_root = root_chain[sampled_pass]["round_roots"][sampled_round]

    if transition_mix_rounds <= 0:
        sampled_block = cap._build_sampled_block_proof(
            ext=ext,
            ws=ws,
            seed=seed,
            pass_index=sampled_pass,
            round_index=sampled_round,
            round_root=sampled_round_root,
            lease_id=lease_id,
            gpu_index=gpu_index,
            matrix_dim=matrix_dim,
            block_size=block_size,
            spot_checks=spot_checks,
            challenge_seed_hex=challenge_seed,
            memory_mix_rounds=0,
        )
        proof = {
            "format": WORKSPACE_PROOF_FORMAT,
            "workload_version": "fixed_workspace_merkle_gemm_v1",
            "memory_mix_rounds": 0,
            "transition_mix_rounds": 0,
            "transition_fanout": 1,
            "proof_challenge_seed": challenge_seed,
            "proof_challenge_public": True,
            "matrix_dim": matrix_dim,
            "block_size": block_size,
            "rounds": rounds,
            "passes": passes,
            "spot_checks": spot_checks,
            "workspace_seed": int(seed),
            "root_chain": root_chain,
            "sampled": {
                "pass_index": sampled_pass,
                "round_index": sampled_round,
                "round_root": sampled_round_root,
                "block_index": sampled_block["block_index"],
            },
            "block_proof": sampled_block["block_proof"],
        }
        if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
            proof = _attach_capacity_gemm_v2(
                proof=proof,
                capacity_params={
                    "matrix_dim": matrix_dim,
                    "block_size": block_size,
                    "passes": passes,
                    "rounds": rounds,
                    "transition_mix_rounds": 0,
                    "transition_fanout": 1,
                },
                lease_id=lease_id,
                gpu_index=gpu_index,
                proof_seed_hex=proof_seed_hex,
                challenge_seed=challenge_seed,
                combined_transcript_root_hex=combined_transcript_root_hex,
                state_openings=[],
            )
        return proof

    sampled_mix = derive_sampled_transition_mix_index(
        challenge_seed,
        lease_id,
        gpu_index,
        sampled_pass,
        sampled_round,
        transition_mix_rounds,
    )
    sampled_state_roots = state_root_chain[sampled_pass]["round_state_roots"][sampled_round]

    sampled_state_tensor = torch.empty((transition_mix_rounds + 1, 32), dtype=torch.uint8, device=ws.a.device)
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
    recomputed_round_root = cap._tensor_u8_hex(sampled_round_tensor[0])
    if recomputed_round_root != sampled_round_root:
        raise RuntimeError(f"sampled capacity round root mismatch: {recomputed_round_root} != {sampled_round_root}")
    recomputed_state_roots = [cap._tensor_u8_hex(sampled_state_tensor[idx]) for idx in range(transition_mix_rounds + 1)]
    if recomputed_state_roots != sampled_state_roots:
        raise RuntimeError("sampled capacity state root mismatch")

    sampled_block = cap._build_resident_challenged_block_proof(
        ext=ext,
        ws=ws,
        round_root=sampled_round_root,
        pass_index=sampled_pass,
        round_index=sampled_round,
        lease_id=lease_id,
        gpu_index=gpu_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        spot_checks=spot_checks,
        challenge_seed_hex=challenge_seed,
    )

    final_state_root, final_tree = cap._state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
    expected_final_state_root = sampled_state_roots[-1]
    if final_state_root != expected_final_state_root:
        raise RuntimeError(f"sampled final state root mismatch: {final_state_root} != {expected_final_state_root}")

    gemm_state_blocks = {
        cap._state_block_index_for_element(int(spot["row"]) * matrix_dim + int(spot["col"]), matrix_dim, block_size)
        for spot in sampled_block["block_proof"]["spot_A"]
    }
    gemm_state_blocks.update(
        cap._state_block_index_for_element(int(spot["row"]) * matrix_dim + int(spot["col"]), matrix_dim, block_size)
        for spot in sampled_block["block_proof"]["spot_B"]
    )
    gemm_state_openings = [
        cap._state_opening_from_tensors(ws.a, ws.b, final_tree, bidx, matrix_dim, block_size)
        for bidx in sorted(gemm_state_blocks)
    ]

    proof = {
        "format": WORKSPACE_TRANSITION_PROOF_FORMAT,
        "workload_version": (
            "fixed_workspace_transition_fanout_v3"
            if transition_fanout > 1
            else "fixed_workspace_transition_mix_v3"
        ),
        "memory_mix_rounds": 0,
        "transition_mix_rounds": transition_mix_rounds,
        "transition_fanout": transition_fanout,
        "proof_challenge_seed": challenge_seed,
        "proof_challenge_public": True,
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "rounds": rounds,
        "passes": passes,
        "spot_checks": spot_checks,
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
    }
    if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
        capacity_params = {
            "matrix_dim": matrix_dim,
            "block_size": block_size,
            "passes": passes,
            "rounds": rounds,
            "transition_mix_rounds": transition_mix_rounds,
            "transition_fanout": transition_fanout,
        }
        context = capacity_gemm_v2_context_from_workspace_proof(
            proof=proof,
            capacity_params=capacity_params,
            lease_id=lease_id,
            gpu_index=gpu_index,
            proof_seed_hex=proof_seed_hex,
            challenge_seed_hex=challenge_seed,
            transcript_root_hex=combined_transcript_root_hex,
        )
        v2_state_openings = _transition_band_state_openings(
            a=ws.a,
            b=ws.b,
            tree=final_tree,
            block_indices=required_transition_state_blocks(context),
            matrix_dim=matrix_dim,
            block_size=block_size,
            row_block=context.row_block,
            column_block=context.column_block,
        )
        proof = _attach_capacity_gemm_v2(
            proof=proof,
            capacity_params=capacity_params,
            lease_id=lease_id,
            gpu_index=gpu_index,
            proof_seed_hex=proof_seed_hex,
            challenge_seed=challenge_seed,
            combined_transcript_root_hex=combined_transcript_root_hex,
            state_openings=v2_state_openings,
        )

    before_root = sampled_state_roots[sampled_mix]
    after_root = sampled_state_roots[sampled_mix + 1]
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
    before_state_root, before_tree = cap._state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
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
    after_state_root, after_tree = cap._state_tree_from_resident_tensors(ext, ws.a, ws.b, block_size)
    if after_state_root != after_root:
        raise RuntimeError(f"sampled after state root mismatch: {after_state_root} != {after_root}")

    element_index = derive_transition_element_index(
        before_root_hex=before_root,
        after_root_hex=after_root,
        lease_id=lease_id,
        gpu_index=gpu_index,
        pass_index=sampled_pass,
        round_index=sampled_round,
        mix_index=sampled_mix,
        matrix_dim=matrix_dim,
        challenge_seed_hex=challenge_seed,
    )
    source_pairs = transition_source_indices(
        workspace_seed=seed,
        pass_index=sampled_pass,
        round_index=sampled_round,
        mix_index=sampled_mix,
        element_index=element_index,
        matrix_dim=matrix_dim,
        fanout=transition_fanout,
    )
    transition_before_blocks = {cap._state_block_index_for_element(element_index, matrix_dim, block_size)}
    for j, k in source_pairs:
        transition_before_blocks.add(cap._state_block_index_for_element(j, matrix_dim, block_size))
        transition_before_blocks.add(cap._state_block_index_for_element(k, matrix_dim, block_size))
    transition_after_blocks = {cap._state_block_index_for_element(element_index, matrix_dim, block_size)}

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
        cap._state_opening_from_tensors(ws.a, ws.b, before_tree, bidx, matrix_dim, block_size)
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
        cap._state_opening_from_tensors(ws.a, ws.b, after_tree, bidx, matrix_dim, block_size)
        for bidx in sorted(transition_after_blocks)
    ]

    proof["sampled_transition"] = {
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
    }
    return proof


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    proof_protocol_version = int(args.proof_protocol_version)
    combined_proof_format = _combined_proof_format(proof_protocol_version)
    ext = fp64._import_ext()
    torch.cuda.set_device(int(args.device_index))
    device = torch.device(f"cuda:{int(args.device_index)}")

    cap_n = int(args.capacity_matrix_dim)
    fp64_n = int(args.fp64_matrix_dim)
    warmup_cap_seed = cap.seed_for(args.lease_id, int(args.gpu_index), args.seed_hex)
    warmup_fp64_seed = fp64.seed_for(args.lease_id, int(args.gpu_index), args.seed_hex)
    capacity_mix_rounds = int(args.transition_mix_rounds)
    capacity_warmup_passes = max(0, int(args.capacity_warmup_passes))
    capacity_tail_passes = max(0, int(args.capacity_tail_passes))
    capacity_tail_rounds = max(1, int(args.capacity_tail_rounds))
    capacity_tail_warmup_passes = max(0, int(args.capacity_tail_warmup_passes)) if capacity_tail_passes else 0
    capacity_tail_mix_rounds = max(1, int(args.capacity_tail_transition_mix_rounds))
    capacity_tail_fanout = max(1, int(args.capacity_tail_transition_fanout))
    fp64_warmup_passes = max(0, int(args.fp64_warmup_passes))

    cap_allocated_passes = max(1, int(args.capacity_passes), capacity_warmup_passes, capacity_tail_passes, capacity_tail_warmup_passes)
    cap_allocated_rounds = max(int(args.capacity_rounds), capacity_tail_rounds)
    cap_ws = cap.allocate_workspace(cap_n, cap_allocated_passes, cap_allocated_rounds, str(device))
    state_roots = torch.empty(
        (
            max(1, cap_allocated_passes * int(args.capacity_rounds) * (max(0, capacity_mix_rounds) + 1)),
            32,
        ),
        dtype=torch.uint8,
        device=device,
    )
    torch.cuda.synchronize()

    capacity_warmup_ms = 0.0
    capacity_warmup_wall_s = 0.0
    if capacity_warmup_passes:
        capacity_warmup_started = time.perf_counter()
        for warmup_pass in range(capacity_warmup_passes):
            if capacity_mix_rounds > 0:
                capacity_warmup_ms += float(ext.run_workspace_pass_transition_committed(
                    cap_ws.a,
                    cap_ws.b,
                    cap_ws.q,
                    cap_ws.partial,
                    cap_ws.root,
                    state_roots,
                    cap_ws.round_roots,
                    int(warmup_cap_seed) ^ 0xA5A55A5A12345678,
                    warmup_pass,
                    int(args.capacity_rounds),
                    int(args.capacity_block_size),
                    capacity_mix_rounds,
                    int(args.transition_fanout),
                    int(args.device_index),
                ))
            else:
                capacity_warmup_ms += float(ext.run_workspace_pass_committed(
                    cap_ws.a,
                    cap_ws.b,
                    cap_ws.q,
                    cap_ws.partial,
                    cap_ws.root,
                    cap_ws.round_roots,
                    int(warmup_cap_seed) ^ 0xA5A55A5A12345678,
                    warmup_pass,
                    int(args.capacity_rounds),
                    int(args.capacity_block_size),
                    int(args.device_index),
                ))
        torch.cuda.synchronize()
        capacity_warmup_wall_s = time.perf_counter() - capacity_warmup_started

    capacity_tail_warmup_ms = 0.0
    capacity_tail_warmup_wall_s = 0.0
    tail_state_roots = torch.empty(
        (
            max(1, capacity_tail_passes, capacity_tail_warmup_passes)
            * capacity_tail_rounds
            * (capacity_tail_mix_rounds + 1),
            32,
        ),
        dtype=torch.uint8,
        device=device,
    )
    if capacity_tail_warmup_passes:
        capacity_tail_warmup_started = time.perf_counter()
        for warmup_pass in range(capacity_tail_warmup_passes):
            capacity_tail_warmup_ms += float(ext.run_workspace_pass_transition_committed(
                cap_ws.a,
                cap_ws.b,
                cap_ws.q,
                cap_ws.partial,
                cap_ws.root,
                tail_state_roots,
                cap_ws.round_roots,
                int(warmup_cap_seed) ^ 0xBADC0FFEE0DDF00D,
                warmup_pass,
                capacity_tail_rounds,
                int(args.capacity_block_size),
                capacity_tail_mix_rounds,
                capacity_tail_fanout,
                int(args.device_index),
            ))
        torch.cuda.synchronize()
        capacity_tail_warmup_wall_s = time.perf_counter() - capacity_tail_warmup_started

    fp64_a = fp64_b = fp64_c = fp64_round_roots = None
    fp64_prepared_before_start = bool(getattr(args, "start_file", None)) and (
        int(args.fp64_passes) > 0 or fp64_warmup_passes > 0
    )
    fp64_warmup_ms = 0.0
    fp64_warmup_wall_s = 0.0
    if fp64_prepared_before_start:
        fp64_a = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_b = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_c = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_round_roots = torch.empty(
            (max(1, int(args.fp64_passes), fp64_warmup_passes) * int(args.fp64_rounds), 32),
            dtype=torch.uint8,
            device=device,
        )
        torch.cuda.synchronize()
        if fp64_warmup_passes:
            fp64_warmup_started = time.perf_counter()
            fp64_warmup_ms = float(ext.run_fp64_identity_chain_committed(
                fp64_a,
                fp64_b,
                fp64_c,
                fp64_round_roots,
                int(warmup_fp64_seed) ^ 0x5A5AA5A5DDBCA9D6,
                0,
                fp64_warmup_passes,
                int(args.fp64_rounds),
                int(args.fp64_block_size),
                int(args.device_index),
            ))
            torch.cuda.synchronize()
            fp64_warmup_wall_s = time.perf_counter() - fp64_warmup_started

    start_payload = _write_ready_and_wait_for_start(args, {
        "lease_id": args.lease_id,
        "gpu_index": int(args.gpu_index),
        "workload_version": WORKLOAD_VERSION,
        "t": time.time(),
        "fp64_prepared": bool(fp64_prepared_before_start),
        "capacity_warmup_passes": capacity_warmup_passes,
        "capacity_tail_warmup_passes": capacity_tail_warmup_passes,
        "fp64_warmup_passes": fp64_warmup_passes,
        "pre_timing_used_bytes": int(torch.cuda.max_memory_allocated(device)),
    })
    cap_seed = cap.seed_for(args.lease_id, int(args.gpu_index), args.seed_hex)
    fp64_seed = fp64.seed_for(args.lease_id, int(args.gpu_index), args.seed_hex)

    end_to_end_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    capacity_started = time.perf_counter()
    cap_ms_total = 0.0
    capacity_main_started = time.perf_counter()
    for pass_index in range(int(args.capacity_passes)):
        if capacity_mix_rounds > 0:
            cap_ms_total += float(ext.run_workspace_pass_transition_committed(
                cap_ws.a,
                cap_ws.b,
                cap_ws.q,
                cap_ws.partial,
                cap_ws.root,
                state_roots,
                cap_ws.round_roots,
                int(cap_seed),
                pass_index,
                int(args.capacity_rounds),
                int(args.capacity_block_size),
                capacity_mix_rounds,
                int(args.transition_fanout),
                int(args.device_index),
            ))
        else:
            cap_ms_total += float(ext.run_workspace_pass_committed(
                cap_ws.a,
                cap_ws.b,
                cap_ws.q,
                cap_ws.partial,
                cap_ws.root,
                cap_ws.round_roots,
                int(cap_seed),
                pass_index,
                int(args.capacity_rounds),
                int(args.capacity_block_size),
                int(args.device_index),
            ))
    torch.cuda.synchronize()
    capacity_main_wall_s = time.perf_counter() - capacity_main_started

    if capacity_mix_rounds > 0:
        cap_root_chain, cap_state_root_chain = _capacity_root_chains(
            round_roots=cap_ws.round_roots,
            state_roots=state_roots,
            passes=int(args.capacity_passes),
            rounds=int(args.capacity_rounds),
            transition_mix_rounds=capacity_mix_rounds,
        )
    else:
        cap_root_chain = _capacity_round_root_chain(
            round_roots=cap_ws.round_roots,
            passes=int(args.capacity_passes),
            rounds=int(args.capacity_rounds),
        )
        cap_state_root_chain = []
    cap_transcript = (
        workspace_transcript_root_v2(cap_root_chain, cap_state_root_chain)
        if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
        else cap.workspace_transcript_root(cap_root_chain)
    )

    capacity_tail_ms = 0.0
    capacity_tail_wall_s = 0.0
    cap_tail_root_chain: list[dict[str, Any]] = []
    cap_tail_state_root_chain: list[dict[str, Any]] = []
    cap_tail_transcript = ""
    if capacity_tail_passes:
        capacity_tail_started = time.perf_counter()
        for pass_index in range(capacity_tail_passes):
            capacity_tail_ms += float(ext.run_workspace_pass_transition_committed(
                cap_ws.a,
                cap_ws.b,
                cap_ws.q,
                cap_ws.partial,
                cap_ws.root,
                tail_state_roots,
                cap_ws.round_roots,
                int(cap_seed),
                pass_index,
                capacity_tail_rounds,
                int(args.capacity_block_size),
                capacity_tail_mix_rounds,
                capacity_tail_fanout,
                int(args.device_index),
            ))
        torch.cuda.synchronize()
        capacity_tail_wall_s = time.perf_counter() - capacity_tail_started
        cap_tail_root_chain, cap_tail_state_root_chain = _capacity_root_chains(
            round_roots=cap_ws.round_roots,
            state_roots=tail_state_roots,
            passes=capacity_tail_passes,
            rounds=capacity_tail_rounds,
            transition_mix_rounds=capacity_tail_mix_rounds,
        )
        cap_tail_transcript = (
            workspace_transcript_root_v2(
                cap_tail_root_chain,
                cap_tail_state_root_chain,
            )
            if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
            else cap.workspace_transcript_root(cap_tail_root_chain)
        )

    capacity_wall_s = time.perf_counter() - capacity_started
    capacity_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    capacity_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    if not args.simultaneous_workspaces:
        del state_roots
        del tail_state_roots
        del cap_ws
        if args.empty_cache_between_lanes:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if fp64_a is None or fp64_b is None or fp64_c is None or fp64_round_roots is None:
        fp64_a = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_b = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_c = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
        fp64_round_roots = torch.empty(
            (max(1, int(args.fp64_passes), fp64_warmup_passes) * int(args.fp64_rounds), 32),
            dtype=torch.uint8,
            device=device,
        )
        torch.cuda.synchronize()

    if fp64_warmup_passes and not fp64_prepared_before_start:
        fp64_warmup_started = time.perf_counter()
        fp64_warmup_ms = float(ext.run_fp64_identity_chain_committed(
            fp64_a,
            fp64_b,
            fp64_c,
            fp64_round_roots,
            int(fp64_seed) ^ 0x5A5AA5A5DDBCA9D6,
            0,
            fp64_warmup_passes,
            int(args.fp64_rounds),
            int(args.fp64_block_size),
            int(args.device_index),
        ))
        torch.cuda.synchronize()
        fp64_warmup_wall_s = time.perf_counter() - fp64_warmup_started

    torch.cuda.reset_peak_memory_stats(device)
    fp64_started = time.perf_counter()
    fp64_ms = float(ext.run_fp64_identity_chain_committed(
        fp64_a,
        fp64_b,
        fp64_c,
        fp64_round_roots,
        int(fp64_seed),
        0,
        int(args.fp64_passes),
        int(args.fp64_rounds),
        int(args.fp64_block_size),
        int(args.device_index),
    ))
    torch.cuda.synchronize()
    fp64_wall_s = time.perf_counter() - fp64_started
    timed_wall_s = capacity_wall_s + fp64_wall_s
    end_to_end_timed_phase_s = time.perf_counter() - end_to_end_started
    fp64_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    fp64_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved_mib = max(capacity_peak_reserved_mib, fp64_peak_reserved_mib)
    peak_allocated_mib = max(capacity_peak_allocated_mib, fp64_peak_allocated_mib)

    fp64_root_chain = fp64.root_chain_from_tensor(fp64_round_roots, int(args.fp64_passes), int(args.fp64_rounds))
    fp64_transcript = fp64.transcript_root(
        seed=fp64_seed,
        matrix_dim=fp64_n,
        block_size=int(args.fp64_block_size),
        passes=int(args.fp64_passes),
        rounds=int(args.fp64_rounds),
        root_chain=fp64_root_chain,
    )
    if not args.simultaneous_workspaces:
        del fp64_a
        del fp64_b
        del fp64_c
        del fp64_round_roots
        if args.empty_cache_between_lanes:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    capacity_params = {
        "matrix_dim": cap_n,
        "block_size": int(args.capacity_block_size),
        "passes": int(args.capacity_passes),
        "rounds": int(args.capacity_rounds),
        "transition_mix_rounds": int(args.transition_mix_rounds),
        "transition_fanout": int(args.transition_fanout),
    }
    capacity_tail_params = None
    if capacity_tail_passes:
        capacity_tail_params = {
            "matrix_dim": cap_n,
            "block_size": int(args.capacity_block_size),
            "passes": capacity_tail_passes,
            "rounds": capacity_tail_rounds,
            "transition_mix_rounds": capacity_tail_mix_rounds,
            "transition_fanout": capacity_tail_fanout,
        }
    fp64_params = {
        "matrix_dim": fp64_n,
        "block_size": int(args.fp64_block_size),
        "passes": int(args.fp64_passes),
        "rounds": int(args.fp64_rounds),
    }
    combined_transcript = combined_transcript_root(
        capacity_transcript=cap_transcript,
        capacity_tail_transcript=cap_tail_transcript,
        fp64_transcript=fp64_transcript,
        capacity_params=capacity_params,
        capacity_tail_params=capacity_tail_params,
        fp64_params=fp64_params,
        proof_protocol_version=proof_protocol_version,
    )
    pass0_root = cap_root_chain[0]["pass_root"] if cap_root_chain else combined_transcript
    final_root = combined_transcript
    final_timing = {
        "lease_id": args.lease_id,
        "root": final_root,
        "pass0_root": pass0_root,
        "transcript_root": combined_transcript,
        "proof_format": combined_proof_format,
        "proof_protocol_version": proof_protocol_version,
        "workload_version": WORKLOAD_VERSION,
        "pass_count": _combined_pass_count(args),
        "capacity_transcript_root": cap_transcript,
        "capacity_tail_transcript_root": cap_tail_transcript,
        "fp64_transcript_root": fp64_transcript,
        "combined_transcript_root": combined_transcript,
        "capacity_params": capacity_params,
        "capacity_tail_params": capacity_tail_params,
        "fp64_params": fp64_params,
        "workspace_mode": "simultaneous" if args.simultaneous_workspaces else "sequential_reuse",
        "hot_start": bool(getattr(args, "start_file", None)),
        "hot_start_payload": start_payload,
        "timed_wall_s": timed_wall_s,
        "end_to_end_timed_phase_s": end_to_end_timed_phase_s,
        "timed_cuda_component_s": (cap_ms_total + capacity_tail_ms + fp64_ms) / 1000.0,
        "capacity_cuda_s": cap_ms_total / 1000.0,
        "capacity_tail_cuda_s": capacity_tail_ms / 1000.0,
        "fp64_cuda_s": fp64_ms / 1000.0,
        "t": time.time(),
    }
    if getattr(args, "child", False):
        _write_child_timing_artifacts(
            args=args,
            pass0_root=pass0_root,
            final_timing=final_timing,
        )

    b_proof_seed, b_proof_source = _resolve_b_proof_seed(args)
    sample_seed = sample_seed_from_b_proof(
        b_proof_seed=b_proof_seed,
        combined_transcript=combined_transcript,
    ) if b_proof_seed else ""

    capacity_proof: dict[str, Any] = {}
    capacity_tail_proof: dict[str, Any] | None = None
    fp64_proof: dict[str, Any] = {}
    combined_proof: dict[str, Any] = {}
    proof_payload_s = 0.0
    proof_peak_reserved_mib = 0.0
    proof_peak_allocated_mib = 0.0
    verify_s = 0.0
    cap_ok = False
    cap_reason = "missing_b_proof_seed"
    cap_tail_ok = True
    cap_tail_reason = ""
    fp64_required = int(args.fp64_passes) > 0
    fp64_ok = not fp64_required
    fp64_reason = "skipped" if not fp64_required else "missing_b_proof_seed"

    if b_proof_seed and not args.simultaneous_workspaces:
        cap_ws = cap.allocate_workspace(cap_n, max(1, int(args.capacity_passes)), int(args.capacity_rounds), str(device))
        torch.cuda.synchronize()

    if b_proof_seed:
        torch.cuda.reset_peak_memory_stats(device)
        proof_start = time.perf_counter()
        capacity_proof = build_capacity_proof(
            ext=ext,
            ws=cap_ws,
            seed=cap_seed,
            root_chain=cap_root_chain,
            state_root_chain=cap_state_root_chain,
            challenge_seed=sample_seed,
            lease_id=args.lease_id,
            gpu_index=int(args.gpu_index),
            matrix_dim=cap_n,
            block_size=int(args.capacity_block_size),
            passes=int(args.capacity_passes),
            rounds=int(args.capacity_rounds),
            spot_checks=int(args.spot_checks),
            transition_mix_rounds=int(args.transition_mix_rounds),
            transition_fanout=int(args.transition_fanout),
            proof_protocol_version=proof_protocol_version,
            proof_seed_hex=str(args.seed_hex or ""),
            combined_transcript_root_hex=combined_transcript,
        )
        if capacity_tail_passes:
            if not args.simultaneous_workspaces:
                del cap_ws
                if args.empty_cache_between_lanes:
                    torch.cuda.empty_cache()
                cap_ws = cap.allocate_workspace(cap_n, capacity_tail_passes, capacity_tail_rounds, str(device))
                torch.cuda.synchronize()
            capacity_tail_proof = build_capacity_proof(
                ext=ext,
                ws=cap_ws,
                seed=cap_seed,
                root_chain=cap_tail_root_chain,
                state_root_chain=cap_tail_state_root_chain,
                challenge_seed=sample_seed,
                lease_id=args.lease_id,
                gpu_index=int(args.gpu_index),
                matrix_dim=cap_n,
                block_size=int(args.capacity_block_size),
                passes=capacity_tail_passes,
                rounds=capacity_tail_rounds,
                spot_checks=int(args.spot_checks),
                transition_mix_rounds=capacity_tail_mix_rounds,
                transition_fanout=capacity_tail_fanout,
                proof_protocol_version=proof_protocol_version,
                proof_seed_hex=str(args.seed_hex or ""),
                combined_transcript_root_hex=combined_transcript,
            )
        if fp64_required and not args.simultaneous_workspaces:
            del cap_ws
            if args.empty_cache_between_lanes:
                torch.cuda.empty_cache()
            fp64_a = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
            fp64_b = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
            fp64_c = torch.empty((fp64_n, fp64_n), dtype=torch.float64, device=device)
            fp64_round_roots = torch.empty(
                (max(1, int(args.fp64_passes) * int(args.fp64_rounds)), 32),
                dtype=torch.uint8,
                device=device,
            )
            torch.cuda.synchronize()
        if fp64_required:
            fp64_proof = fp64.build_fp64_identity_proof(
                ext=ext,
                a=fp64_a,
                b=fp64_b,
                c=fp64_c,
                seed=fp64_seed,
                matrix_dim=fp64_n,
                block_size=int(args.fp64_block_size),
                passes=int(args.fp64_passes),
                rounds=int(args.fp64_rounds),
                round_roots=fp64_round_roots,
                root_chain=fp64_root_chain,
                challenge_hex=sample_seed,
                transcript_hex=fp64_transcript,
                spot_checks=int(args.spot_checks),
                device_index=int(args.device_index),
            )
        proof_payload_s = time.perf_counter() - proof_start
        proof_peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        proof_peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        combined_proof = {
            "format": combined_proof_format,
            "proof_protocol_version": proof_protocol_version,
            "workload_version": WORKLOAD_VERSION,
            "lease_id": args.lease_id,
            "gpu_index": int(args.gpu_index),
            "pass_count": _combined_pass_count(args),
            "capacity_params": capacity_params,
            "capacity_tail_params": capacity_tail_params,
            "fp64_params": fp64_params,
            "capacity_transcript_root": cap_transcript,
            "capacity_tail_transcript_root": cap_tail_transcript,
            "fp64_transcript_root": fp64_transcript,
            "combined_transcript_root": combined_transcript,
            "b_proof_seed": b_proof_seed,
            "b_proof_source": b_proof_source,
            "sample_seed": sample_seed,
            "capacity_proof": capacity_proof,
            "capacity_tail_proof": capacity_tail_proof,
            "fp64_proof": fp64_proof,
        }

        verify_start = time.perf_counter()
        cap_ok, cap_reason = verify_workspace_proof_payload(
            proof=capacity_proof,
            expected_transcript_root=cap_transcript,
            expected_pass0_root=cap_root_chain[0]["pass_root"],
            expected_final_root=cap_root_chain[-1]["pass_root"],
            lease_id=args.lease_id,
            gpu_index=int(args.gpu_index),
            proof_seed_hex=str(args.seed_hex or ""),
            pass_count=int(args.capacity_passes),
            proof_challenge_seed_hex=sample_seed,
            expected_spot_checks=int(args.spot_checks),
            proof_protocol_version=proof_protocol_version,
        )
        if capacity_tail_passes and capacity_tail_proof is not None:
            cap_tail_ok, cap_tail_reason = verify_workspace_proof_payload(
                proof=capacity_tail_proof,
                expected_transcript_root=cap_tail_transcript,
                expected_pass0_root=cap_tail_root_chain[0]["pass_root"],
                expected_final_root=cap_tail_root_chain[-1]["pass_root"],
                lease_id=args.lease_id,
                gpu_index=int(args.gpu_index),
                proof_seed_hex=str(args.seed_hex or ""),
                pass_count=capacity_tail_passes,
                proof_challenge_seed_hex=sample_seed,
                expected_spot_checks=int(args.spot_checks),
                proof_protocol_version=proof_protocol_version,
            )
        if fp64_required:
            fp64_ok, fp64_reason = fp64.verify_fp64_identity_proof(fp64_proof)
        if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
            combined_ok, combined_reason = verify_combined_proof_payload(
                proof=combined_proof,
                final_artifact={
                    "combined": combined_commitment_from_final_timing(final_timing)
                },
                expected_combined_transcript_root=combined_transcript,
                lease_id=args.lease_id,
                gpu_index=int(args.gpu_index),
                proof_seed_hex=str(args.seed_hex or ""),
                proof_challenge_seed_hex=b_proof_seed,
                expected_workload_spec={
                    "workload_version": WORKLOAD_VERSION,
                    "capacity_matrix_dim": cap_n,
                    "capacity_block_size": int(args.capacity_block_size),
                    "capacity_passes": int(args.capacity_passes),
                    "capacity_rounds": int(args.capacity_rounds),
                    "transition_mix_rounds": int(args.transition_mix_rounds),
                    "transition_fanout": int(args.transition_fanout),
                    "capacity_tail_passes": capacity_tail_passes,
                    "capacity_tail_rounds": capacity_tail_rounds,
                    "capacity_tail_transition_mix_rounds": capacity_tail_mix_rounds,
                    "capacity_tail_transition_fanout": capacity_tail_fanout,
                    "fp64_matrix_dim": fp64_n,
                    "fp64_block_size": int(args.fp64_block_size),
                    "fp64_passes": int(args.fp64_passes),
                    "fp64_rounds": int(args.fp64_rounds),
                    "spot_checks": int(args.spot_checks),
                    "pass_count": _combined_pass_count(args),
                },
            )
            if not combined_ok:
                cap_ok = False
                cap_reason = combined_reason
        verify_s = time.perf_counter() - verify_start

    summary = {
        "workload_version": WORKLOAD_VERSION,
        "proof_format": combined_proof_format,
        "proof_protocol_version": proof_protocol_version,
        "lease_id": args.lease_id,
        "gpu_index": int(args.gpu_index),
        "pass_count": _combined_pass_count(args),
        "capacity_params": capacity_params,
        "capacity_tail_params": capacity_tail_params,
        "fp64_params": fp64_params,
        "warmup": {
            "capacity_passes": capacity_warmup_passes,
            "capacity_timed_s": capacity_warmup_ms / 1000.0,
            "capacity_wall_s": capacity_warmup_wall_s,
            "capacity_tail_passes": capacity_tail_warmup_passes,
            "capacity_tail_timed_s": capacity_tail_warmup_ms / 1000.0,
            "capacity_tail_wall_s": capacity_tail_warmup_wall_s,
            "fp64_passes": fp64_warmup_passes,
            "fp64_timed_s": fp64_warmup_ms / 1000.0,
            "fp64_wall_s": fp64_warmup_wall_s,
        },
        "capacity_transcript_root": cap_transcript,
        "capacity_tail_transcript_root": cap_tail_transcript,
        "fp64_transcript_root": fp64_transcript,
        "combined_transcript_root": combined_transcript,
        "b_proof_seed": b_proof_seed,
        "b_proof_source": b_proof_source,
        "sample_seed": sample_seed,
        "workspace_mode": "simultaneous" if args.simultaneous_workspaces else "sequential_reuse",
        "empty_cache_between_lanes": bool(args.empty_cache_between_lanes),
        "hot_start": bool(getattr(args, "start_file", None)),
        "hot_start_payload": start_payload,
        "timed_wall_s": timed_wall_s,
        "end_to_end_timed_phase_s": end_to_end_timed_phase_s,
        "timed_cuda_component_s": (cap_ms_total + capacity_tail_ms + fp64_ms) / 1000.0,
        "capacity_cuda_s": cap_ms_total / 1000.0,
        "capacity_tail_cuda_s": capacity_tail_ms / 1000.0,
        "capacity_main_wall_s": capacity_main_wall_s,
        "capacity_tail_wall_s": capacity_tail_wall_s,
        "capacity_wall_s": capacity_wall_s,
        "fp64_cuda_s": fp64_ms / 1000.0,
        "fp64_wall_s": fp64_wall_s,
        "proof_payload_s": proof_payload_s,
        "proof_payload_bytes": (
            len(
                json.dumps(
                    combined_proof,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
            if combined_proof
            else 0
        ),
        "verify_s": verify_s,
        "verify_ok": bool(cap_ok and cap_tail_ok and fp64_ok),
        "capacity_verify_ok": bool(cap_ok),
        "capacity_verify_reason": cap_reason,
        "capacity_tail_verify_ok": bool(cap_tail_ok),
        "capacity_tail_verify_reason": cap_tail_reason,
        "fp64_verify_ok": bool(fp64_ok),
        "fp64_verify_reason": fp64_reason,
        "peak_reserved_mib": peak_reserved_mib,
        "peak_allocated_mib": peak_allocated_mib,
        "capacity_peak_reserved_mib": capacity_peak_reserved_mib,
        "capacity_peak_allocated_mib": capacity_peak_allocated_mib,
        "fp64_peak_reserved_mib": fp64_peak_reserved_mib,
        "fp64_peak_allocated_mib": fp64_peak_allocated_mib,
        "proof_peak_reserved_mib": proof_peak_reserved_mib,
        "proof_peak_allocated_mib": proof_peak_allocated_mib,
        "proof_payload": combined_proof,
        "capacity_proof": capacity_proof,
        "capacity_tail_proof": capacity_tail_proof,
        "fp64_proof": fp64_proof,
    }
    if getattr(args, "child", False):
        # The combined payload already contains all three lane proofs.  Avoid
        # duplicating multi-megabyte transition openings in the handoff file.
        summary.pop("capacity_proof", None)
        summary.pop("capacity_tail_proof", None)
        summary.pop("fp64_proof", None)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lease-id", default="lease-000")
    p.add_argument("--seed-hex", default="")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--capacity-matrix-dim", type=int, default=512)
    p.add_argument("--capacity-passes", type=int, default=1)
    p.add_argument("--capacity-rounds", type=int, default=1)
    p.add_argument("--capacity-warmup-passes", type=int, default=1)
    p.add_argument("--capacity-block-size", type=int, default=64)
    p.add_argument("--transition-mix-rounds", type=int, default=2)
    p.add_argument("--transition-fanout", type=int, default=8)
    p.add_argument("--capacity-tail-passes", type=int, default=0)
    p.add_argument("--capacity-tail-rounds", type=int, default=1)
    p.add_argument("--capacity-tail-warmup-passes", type=int, default=1)
    p.add_argument("--capacity-tail-transition-mix-rounds", type=int, default=1)
    p.add_argument("--capacity-tail-transition-fanout", type=int, default=8)
    p.add_argument("--fp64-matrix-dim", type=int, default=512)
    p.add_argument("--fp64-passes", type=int, default=1)
    p.add_argument("--fp64-rounds", type=int, default=1)
    p.add_argument("--fp64-warmup-passes", type=int, default=1)
    p.add_argument("--fp64-block-size", type=int, default=64)
    p.add_argument("--spot-checks", type=int, default=1)
    p.add_argument(
        "--proof-protocol-version",
        type=int,
        choices=(
            LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
            CURRENT_COMBINED_PROOF_PROTOCOL_VERSION,
        ),
        default=LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
    )
    p.add_argument("--b-proof-hex", default="")
    p.add_argument("--child", action="store_true", help="write miner runtime pass0/final/proof JSON artifacts")
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    p.add_argument("--challenge-file", type=pathlib.Path, default=None)
    p.add_argument("--challenge-timeout-s", type=float, default=0.0)
    p.add_argument("--ready-file", type=pathlib.Path, default=None)
    p.add_argument("--start-file", type=pathlib.Path, default=None)
    p.add_argument("--start-timeout-s", type=float, default=60.0)
    p.add_argument(
        "--simultaneous-workspaces",
        action="store_true",
        help="diagnostic: keep capacity and FP64 workspaces resident together during timed work",
    )
    p.add_argument(
        "--empty-cache-between-lanes",
        action="store_true",
        help="diagnostic PyTorch mode: release cached CUDA blocks between timed lanes",
    )
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    if args.child and args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        child_text = json.dumps(summary, sort_keys=True, separators=(",", ":"))
        (args.out_dir / f"{args.lease_id}_final.json").write_text(
            child_text + "\n"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    stdout_summary = summary
    if args.child:
        stdout_summary = dict(summary)
        stdout_summary.pop("proof_payload", None)
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
