"""Capacity audit bench: run one scheduled workload on this GPU and build its combined proof.

The miner's audit worker launches one process per drafted endpoint, pinned to that endpoint's GPU through
``CUDA_VISIBLE_DEVICES``:

1. allocate and warm up, write the ready file, and wait for the start file carrying the proof seed (hot start);
2. run the timed lanes (capacity, optional capacity tail, optional FP64), each committing a root chain;
3. write ``<lease>_pass0.json`` and ``<lease>_final_timing.json`` with the combined transcript root;
4. wait for the B_proof challenge seed, build and self-verify the sampled proofs, write ``<lease>_final.json``.

For a local round trip pass ``--seed-hex`` and ``--b-proof-hex`` instead of the start and challenge files.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch

from verathos.capacity.bench import fp64 as fp64_bench
from verathos.capacity.bench.workspace import (
    Workspace,
    allocate_workspace,
    challenged_block_proof,
    gemm_state_blocks,
    load_extension,
    replay_round,
    state_opening,
    state_tree,
    tensor_hex,
)
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
from verathos.capacity.proof.gemm import (
    capacity_gemm_context_from_workspace_proof,
    prove_capacity_gemm,
    required_transition_state_blocks,
)
from verathos.capacity.proof.workspace import (
    WORKSPACE_PROOF_FORMAT,
    WORKSPACE_TRANSITION_PROOF_FORMAT,
    derive_sampled_pass_index,
    derive_sampled_round_index,
    derive_sampled_transition_mix_index,
    derive_transition_element_index,
    pass_root_from_round_roots,
    serialize_merkle_path,
    state_block_index_for_element,
    transition_source_indices,
    workspace_seed_for,
    workspace_transcript_root,
)

GPU_INDEX = 0
WARMUP_SEED_MIX = 0xA5A55A5A12345678
TAIL_WARMUP_SEED_MIX = 0xBADC0FFEE0DDF00D
FP64_WARMUP_SEED_MIX = 0x5A5AA5A5DDBCA9D6
POLL_S = 0.01


def _hex32(text: str) -> str:
    raw = text.strip().removeprefix("0x")
    if len(raw) != 64:
        raise ValueError("expected a 32-byte hex value")
    bytes.fromhex(raw)
    return raw


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _wait_for(path: pathlib.Path, timeout_s: float, parse: Any) -> Any:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return parse(path.read_text())
            except (OSError, ValueError):
                pass
        time.sleep(POLL_S)
    raise TimeoutError(f"timed out waiting for {path}")


# ---- timed lanes ----


def _run_capacity(ext: Any, ws: Workspace, state_roots: torch.Tensor, seed: int, params: dict[str, int],
                  passes: int) -> float:
    """Run ``passes`` committed capacity passes; returns CUDA milliseconds."""
    ms = 0.0
    for pass_index in range(passes):
        if params["transition_mix_rounds"]:
            ms += ext.run_workspace_pass_transition_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, state_roots, ws.round_roots, seed, pass_index,
                params["rounds"], params["block_size"], params["transition_mix_rounds"], params["transition_fanout"], 0)
        else:
            ms += ext.run_workspace_pass_committed(
                ws.a, ws.b, ws.q, ws.partial, ws.root, ws.round_roots, seed, pass_index, params["rounds"],
                params["block_size"], 0)
    torch.cuda.synchronize()
    return float(ms)


def _state_roots_tensor(params: dict[str, int], passes: int, device: torch.device) -> torch.Tensor:
    rows = max(1, passes * params["rounds"] * (params["transition_mix_rounds"] + 1))
    return torch.empty((rows, 32), dtype=torch.uint8, device=device)


def _root_chains(ws: Workspace, state_roots: torch.Tensor, params: dict[str, int]) -> tuple[list, list]:
    """Per-pass round roots and, for transition workspaces, per-round state roots, as committed by the timed run."""
    passes, rounds, mix = params["passes"], params["rounds"], params["transition_mix_rounds"]
    round_rows = ws.round_roots[:passes * rounds].detach().cpu().numpy()
    state_rows = state_roots[:passes * rounds * (mix + 1)].detach().cpu().numpy() if mix else None
    root_chain, state_chain = [], []
    for p in range(passes):
        round_roots = [round_rows[p * rounds + r].tobytes().hex() for r in range(rounds)]
        root_chain.append({"pass_index": p, "pass_root": pass_root_from_round_roots(p, round_roots),
                           "round_roots": round_roots})
        if state_rows is not None:
            state_chain.append({"pass_index": p, "round_state_roots": [
                [state_rows[(p * rounds + r) * (mix + 1) + m].tobytes().hex() for m in range(mix + 1)]
                for r in range(rounds)]})
    return root_chain, state_chain


# ---- capacity lane proof ----


def _band_state_openings(ws: Workspace, tree: Any, block_indices: tuple[int, ...], matrix_dim: int, block_size: int,
                         row_block: int, column_block: int) -> list[dict[str, Any]]:
    """Packed int8 A/B openings for the state blocks on the challenged row and column bands."""
    rs, cs = row_block * block_size, column_block * block_size
    a_row = ws.a[rs:rs + block_size, :].cpu().numpy()
    b_row = ws.b[rs:rs + block_size, :].cpu().numpy()
    a_col = ws.a[:, cs:cs + block_size].cpu().numpy()
    b_col = ws.b[:, cs:cs + block_size].cpu().numpy()
    blocks_per_row = matrix_dim // block_size
    openings = []
    for block_index in block_indices:
        br, bc = divmod(block_index, blocks_per_row)
        if br == row_block:
            a_block = a_row[:, bc * block_size:(bc + 1) * block_size]
            b_block = b_row[:, bc * block_size:(bc + 1) * block_size]
        elif bc == column_block:
            a_block = a_col[br * block_size:(br + 1) * block_size, :]
            b_block = b_col[br * block_size:(br + 1) * block_size, :]
        else:
            raise ValueError("state opening is outside the challenged bands")
        packed = np.empty((block_size * 2, block_size), dtype=np.uint8)
        packed[0::2, :] = a_block.view(np.uint8)
        packed[1::2, :] = b_block.view(np.uint8)
        openings.append({"block_index": block_index,
                         "state_block_i8": "base64:" + base64.b64encode(packed.tobytes()).decode("ascii"),
                         "merkle_path": serialize_merkle_path(tree.get_path(block_index))})
    return openings


def _attach_gemm_proof(proof: dict[str, Any], *, params: dict[str, int], lease_id: str, proof_seed_hex: str,
                       sample_seed_hex: str, combined_root: str, ws: Workspace | None = None, tree: Any = None) -> None:
    context = capacity_gemm_context_from_workspace_proof(
        proof=proof, capacity_params=params, lease_id=lease_id, gpu_index=GPU_INDEX, proof_seed_hex=proof_seed_hex,
        challenge_seed_hex=sample_seed_hex, transcript_root_hex=combined_root)
    state_openings = []
    if ws is not None:
        state_openings = _band_state_openings(ws, tree, required_transition_state_blocks(context), params["matrix_dim"],
                                              params["block_size"], context.row_block, context.column_block)
    block_proof = proof["block_proof"]
    payload = prove_capacity_gemm(context=context, q_block_i32=block_proof["q_block_i32"],
                                     q_merkle_path=block_proof["merkle_path"], state_openings=state_openings)
    proof["gemm_proof"] = payload.to_mapping()
    proof["gemm_band_openings"] = state_openings


def build_capacity_proof(*, ext: Any, ws: Workspace, seed: int, params: dict[str, int], root_chain: list,
                         state_chain: list, lease_id: str, spot_checks: int, proof_seed_hex: str, sample_seed_hex: str,
                         combined_root: str) -> dict[str, Any]:
    """Recompute the challenged round of one capacity lane and open it."""
    n, bs = params["matrix_dim"], params["block_size"]
    passes, rounds, mix, fanout = params["passes"], params["rounds"], params["transition_mix_rounds"], \
        params["transition_fanout"]
    sampled_pass = derive_sampled_pass_index(sample_seed_hex, lease_id, GPU_INDEX, passes)
    sampled_round = derive_sampled_round_index(sample_seed_hex, lease_id, GPU_INDEX, sampled_pass, rounds)
    round_root = root_chain[sampled_pass]["round_roots"][sampled_round]
    common = {"ext": ext, "ws": ws, "round_root": round_root, "pass_index": sampled_pass, "round_index": sampled_round,
              "lease_id": lease_id, "gpu_index": GPU_INDEX, "matrix_dim": n, "block_size": bs,
              "spot_checks": spot_checks, "challenge_seed_hex": sample_seed_hex}
    proof: dict[str, Any] = {
        "transition_mix_rounds": mix,
        "transition_fanout": fanout,
        "proof_challenge_seed": sample_seed_hex,
        "proof_challenge_public": True,
        "matrix_dim": n,
        "block_size": bs,
        "rounds": rounds,
        "passes": passes,
        "spot_checks": spot_checks,
        "workspace_seed": seed,
        "root_chain": root_chain,
    }
    gemm_args = {"params": params, "lease_id": lease_id, "proof_seed_hex": proof_seed_hex,
                 "sample_seed_hex": sample_seed_hex, "combined_root": combined_root}

    if not mix:
        replay_round(ext=ext, ws=ws, seed=seed, pass_index=sampled_pass, round_index=sampled_round)
        block = challenged_block_proof(**common)
        proof.update(format=WORKSPACE_PROOF_FORMAT, workload_version="fixed_workspace_merkle_gemm_v1",
                     sampled={"pass_index": sampled_pass, "round_index": sampled_round, "round_root": round_root,
                              "block_index": block["block_index"]},
                     block_proof=block["block_proof"])
        _attach_gemm_proof(proof, **gemm_args)
        return proof

    sampled_mix = derive_sampled_transition_mix_index(sample_seed_hex, lease_id, GPU_INDEX, sampled_pass,
                                                      sampled_round, mix)
    state_roots = state_chain[sampled_pass]["round_state_roots"][sampled_round]
    state_tensor = torch.empty((mix + 1, 32), dtype=torch.uint8, device=ws.a.device)
    round_tensor = torch.empty((1, 32), dtype=torch.uint8, device=ws.a.device)
    ext.run_workspace_round_transition_committed(ws.a, ws.b, ws.q, ws.partial, ws.root, state_tensor, round_tensor,
                                                 seed, sampled_pass, sampled_round, bs, mix, fanout, 0)
    torch.cuda.synchronize()
    if tensor_hex(round_tensor[0]) != round_root:
        raise RuntimeError("challenged capacity round does not reproduce its committed root")
    if [tensor_hex(state_tensor[i]) for i in range(mix + 1)] != state_roots:
        raise RuntimeError("challenged capacity round does not reproduce its committed state roots")

    block = challenged_block_proof(**common)
    final_root, final_tree = state_tree(ext, ws.a, ws.b, bs)
    if final_root != state_roots[-1]:
        raise RuntimeError("final transition state does not match its committed root")
    proof.update(
        format=WORKSPACE_TRANSITION_PROOF_FORMAT,
        workload_version="fixed_workspace_transition_fanout_v1" if fanout > 1 else "fixed_workspace_transition_mix_v1",
        state_root_chain=state_chain,
        sampled={"pass_index": sampled_pass, "round_index": sampled_round, "round_root": round_root,
                 "block_index": block["block_index"]},
        block_proof=block["block_proof"],
        gemm_state_openings=[state_opening(ws.a, ws.b, final_tree, b, n, bs)
                             for b in gemm_state_blocks(block["block_proof"], n, bs)],
    )
    _attach_gemm_proof(proof, ws=ws, tree=final_tree, **gemm_args)

    def load_state(mix_index: int) -> None:
        ext.run_workspace_round_transition_state(ws.a, ws.b, ws.q, seed, sampled_pass, sampled_round, mix, fanout,
                                                 mix_index, 0)
        torch.cuda.synchronize()

    before_root, after_root = state_roots[sampled_mix], state_roots[sampled_mix + 1]
    load_state(sampled_mix)
    root, before_tree = state_tree(ext, ws.a, ws.b, bs)
    if root != before_root:
        raise RuntimeError("transition state before the challenged mix does not match its committed root")
    load_state(sampled_mix + 1)
    root, after_tree = state_tree(ext, ws.a, ws.b, bs)
    if root != after_root:
        raise RuntimeError("transition state after the challenged mix does not match its committed root")

    element = derive_transition_element_index(
        before_root_hex=before_root, after_root_hex=after_root, lease_id=lease_id, gpu_index=GPU_INDEX,
        pass_index=sampled_pass, round_index=sampled_round, mix_index=sampled_mix, matrix_dim=n,
        challenge_seed_hex=sample_seed_hex)
    sources = transition_source_indices(workspace_seed=seed, pass_index=sampled_pass, round_index=sampled_round,
                                        mix_index=sampled_mix, element_index=element, matrix_dim=n, fanout=fanout)
    after_blocks = {state_block_index_for_element(element, n, bs)}
    before_blocks = after_blocks | {state_block_index_for_element(i, n, bs) for pair in sources for i in pair}
    load_state(sampled_mix)
    before_openings = [state_opening(ws.a, ws.b, before_tree, b, n, bs) for b in sorted(before_blocks)]
    load_state(sampled_mix + 1)
    after_openings = [state_opening(ws.a, ws.b, after_tree, b, n, bs) for b in sorted(after_blocks)]
    proof["sampled_transition"] = {
        "pass_index": sampled_pass, "round_index": sampled_round, "mix_index": sampled_mix,
        "before_root": before_root, "after_root": after_root,
        "opening": {"element_index": element, "before_blocks": before_openings, "after_blocks": after_openings},
    }
    return proof


# ---- one audit ----


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = json.loads(args.workload)
    capacity, tail, fp64_lane, spot_checks = lane_params(spec)
    ext = load_extension()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    lease = args.lease_id
    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocation and warm-up happen before B_start so the timed lanes start hot.
    cap_warmup = int(spec["capacity_warmup_passes"])
    tail_warmup = int(spec["capacity_tail_warmup_passes"]) if tail else 0
    fp64_warmup = int(spec["fp64_warmup_passes"]) if fp64_lane["passes"] else 0
    tail_passes = tail["passes"] if tail else 0
    ws = allocate_workspace(capacity["matrix_dim"], max(capacity["passes"], cap_warmup, tail_passes, tail_warmup),
                            max(capacity["rounds"], tail["rounds"] if tail else 1), device)
    state_roots = _state_roots_tensor(capacity, max(capacity["passes"], cap_warmup), device)
    tail_state_roots = _state_roots_tensor(tail, max(tail_passes, tail_warmup), device) if tail else None
    warmup_seed = workspace_seed_for(lease, GPU_INDEX)
    _run_capacity(ext, ws, state_roots, warmup_seed ^ WARMUP_SEED_MIX, capacity, cap_warmup)
    if tail:
        _run_capacity(ext, ws, tail_state_roots, warmup_seed ^ TAIL_WARMUP_SEED_MIX, tail, tail_warmup)
    fp64_buffers = None
    if fp64_lane["passes"]:
        n = fp64_lane["matrix_dim"]
        fp64_buffers = [torch.empty((n, n), dtype=torch.float64, device=device) for _ in range(3)]
        fp64_roots = torch.empty((max(fp64_lane["passes"], fp64_warmup) * fp64_lane["rounds"], 32), dtype=torch.uint8,
                                 device=device)
        if fp64_warmup:
            ext.run_fp64_identity_chain_committed(*fp64_buffers, fp64_roots,
                                                  fp64.seed_for(lease, GPU_INDEX) ^ FP64_WARMUP_SEED_MIX, 0,
                                                  fp64_warmup, fp64_lane["rounds"],
                                                  fp64_lane["block_size"], 0)
    torch.cuda.synchronize()

    seed_hex = args.seed_hex
    if args.start_file:
        if args.ready_file:
            _write_json(args.ready_file, {"lease_id": lease, "t": time.time()})
        start = _wait_for(args.start_file, args.start_timeout_s, json.loads)
        seed_hex = str(start["seed_hex"])
    seed_hex = _hex32(seed_hex)
    cap_seed = workspace_seed_for(lease, GPU_INDEX, seed_hex)
    fp64_seed = fp64.seed_for(lease, GPU_INDEX, seed_hex)

    # Timed lanes.
    started = time.perf_counter()
    cap_ms = _run_capacity(ext, ws, state_roots, cap_seed, capacity, capacity["passes"])
    cap_chain, cap_state_chain = _root_chains(ws, state_roots, capacity)
    cap_transcript = workspace_transcript_root(cap_chain, cap_state_chain)
    tail_ms, tail_chain, tail_state_chain, tail_transcript = 0.0, [], [], ""
    if tail:
        tail_ms = _run_capacity(ext, ws, tail_state_roots, cap_seed, tail, tail_passes)
        tail_chain, tail_state_chain = _root_chains(ws, tail_state_roots, tail)
        tail_transcript = workspace_transcript_root(tail_chain, tail_state_chain)
    capacity_wall_s = time.perf_counter() - started
    fp64_ms, fp64_chain, fp64_transcript, fp64_wall_s = 0.0, [], "", 0.0
    if fp64_buffers is not None:
        fp64_started = time.perf_counter()
        fp64_ms = float(ext.run_fp64_identity_chain_committed(*fp64_buffers, fp64_roots, fp64_seed, 0,
                                                              fp64_lane["passes"], fp64_lane["rounds"],
                                                              fp64_lane["block_size"], 0))
        torch.cuda.synchronize()
        fp64_wall_s = time.perf_counter() - fp64_started
        fp64_chain = fp64_bench.root_chain_from_tensor(fp64_roots, fp64_lane["passes"], fp64_lane["rounds"])
        fp64_transcript = fp64.transcript_root(seed=fp64_seed, matrix_dim=fp64_lane["matrix_dim"],
                                               block_size=fp64_lane["block_size"], passes=fp64_lane["passes"],
                                               rounds=fp64_lane["rounds"], root_chain=fp64_chain)

    combined_root = combined_transcript_root(
        capacity_transcript=cap_transcript, capacity_tail_transcript=tail_transcript, fp64_transcript=fp64_transcript,
        capacity_params=capacity, capacity_tail_params=tail, fp64_params=fp64_lane)
    pass0_root = cap_chain[0]["pass_root"]
    final_timing = {
        "lease_id": lease,
        "root": combined_root,
        "pass0_root": pass0_root,
        "transcript_root": combined_root,
        "workload_version": WORKLOAD_VERSION,
        "pass_count": spec["pass_count"],
        "combined_transcript_root": combined_root,
        "capacity_transcript_root": cap_transcript,
        "capacity_tail_transcript_root": tail_transcript,
        "fp64_transcript_root": fp64_transcript,
        "capacity_params": capacity,
        "capacity_tail_params": tail,
        "fp64_params": fp64_lane,
        "timed_wall_s": capacity_wall_s + fp64_wall_s,
        "timed_cuda_component_s": (cap_ms + tail_ms + fp64_ms) / 1000.0,
        "capacity_cuda_s": cap_ms / 1000.0,
        "capacity_tail_cuda_s": tail_ms / 1000.0,
        "fp64_cuda_s": fp64_ms / 1000.0,
        "t": time.time(),
    }
    _write_json(out_dir / f"{lease}_pass0.json", {"lease_id": lease, "root": pass0_root, "t": time.time()})
    _write_json(out_dir / f"{lease}_final_timing.json", final_timing)

    # Post-timing proof.
    if args.b_proof_hex:
        b_proof_seed = _hex32(args.b_proof_hex)
    else:
        b_proof_seed = _wait_for(args.challenge_file, args.challenge_timeout_s, _hex32)
    seed = sample_seed(b_proof_seed=b_proof_seed, combined_transcript=combined_root)
    proof_started = time.perf_counter()
    lane_args = {"ext": ext, "ws": ws, "seed": cap_seed, "lease_id": lease, "spot_checks": spot_checks,
                 "proof_seed_hex": seed_hex, "sample_seed_hex": seed, "combined_root": combined_root}
    capacity_proof = build_capacity_proof(params=capacity, root_chain=cap_chain, state_chain=cap_state_chain,
                                          **lane_args)
    tail_proof = build_capacity_proof(params=tail, root_chain=tail_chain, state_chain=tail_state_chain,
                                      **lane_args) if tail else None
    fp64_proof = fp64_bench.build_fp64_proof(
        ext=ext, a=fp64_buffers[0], b=fp64_buffers[1], c=fp64_buffers[2], seed=fp64_seed,
        matrix_dim=fp64_lane["matrix_dim"], block_size=fp64_lane["block_size"],
        passes=fp64_lane["passes"], rounds=fp64_lane["rounds"],
        root_chain=fp64_chain, challenge_hex=seed, transcript_hex=fp64_transcript,
        spot_checks=spot_checks) if fp64_buffers is not None else {}
    proof_s = time.perf_counter() - proof_started
    proof = {
        "format": COMBINED_PROOF_FORMAT,
        "workload_version": WORKLOAD_VERSION,
        "lease_id": lease,
        "gpu_index": GPU_INDEX,
        "pass_count": spec["pass_count"],
        "capacity_params": capacity,
        "capacity_tail_params": tail,
        "fp64_params": fp64_lane,
        "capacity_transcript_root": cap_transcript,
        "capacity_tail_transcript_root": tail_transcript,
        "fp64_transcript_root": fp64_transcript,
        "combined_transcript_root": combined_root,
        "b_proof_seed": b_proof_seed,
        "sample_seed": seed,
        "capacity_proof": capacity_proof,
        "capacity_tail_proof": tail_proof,
        "fp64_proof": fp64_proof,
    }
    verify_started = time.perf_counter()
    ok, reason = verify_combined_proof(proof=proof, final_commit=final_commitment(final_timing), workload_spec=spec,
                                       lease_id=lease, gpu_index=GPU_INDEX, proof_seed_hex=seed_hex,
                                       b_proof_seed_hex=b_proof_seed)
    summary = {
        **{k: v for k, v in final_timing.items() if k.endswith("_s")},
        "proof_s": proof_s,
        "verify_s": time.perf_counter() - verify_started,
        "verify_ok": ok,
        "verify_reason": reason,
        "proof_bytes": len(json.dumps(proof, separators=(",", ":"))),
        "proof_payload": proof,
    }
    _write_json(out_dir / f"{lease}_final.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lease-id", required=True)
    p.add_argument("--workload", required=True, help="workload spec JSON (core.gpu_workload_spec)")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--start-file", type=pathlib.Path, help="hot start: wait for this file to carry seed_hex")
    p.add_argument("--ready-file", type=pathlib.Path, help="written once warm-up is done")
    p.add_argument("--start-timeout-s", type=float, default=60.0)
    p.add_argument("--seed-hex", default="", help="proof seed when no start file is used")
    p.add_argument("--challenge-file", type=pathlib.Path, help="wait for this file to carry the B_proof seed")
    p.add_argument("--challenge-timeout-s", type=float, default=60.0)
    p.add_argument("--b-proof-hex", default="", help="B_proof seed when no challenge file is used")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.start_file and not args.seed_hex:
        raise SystemExit("either --start-file or --seed-hex is required")
    if not args.challenge_file and not args.b_proof_hex:
        raise SystemExit("either --challenge-file or --b-proof-hex is required")
    summary = run(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "proof_payload"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
