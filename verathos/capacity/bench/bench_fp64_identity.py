#!/usr/bin/env python3
"""Local FP64 server-identity lane prototype for hot-capacity audits.

This is a protocol benchmark, not subnet integration code.  It mirrors the
int8 v3 workspace harness shape: timed GPU work commits roots first; only after
that does a Fiat-Shamir/B_proof-like challenge select one round and one output
block for a cheap verifier opening.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import secrets
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


WORKLOAD_VERSION = "fp64_identity_c_merkle_v1"
PROOF_FORMAT = "fp64_identity_merkle_dot_v1"
MASK64 = (1 << 64) - 1
PASS_MUL = 0xD1B54A32D192ED03
ROUND_MUL = 0xABC98388FB8FAC03
FP64_A_DOMAIN = 0xDDBCA9D6B4A7A1D9
FP64_B_DOMAIN = 0xA1F0E27C5C314DD1


def _import_ext():
    # The compiled extension is built into this directory by build.py.
    while str(SCRIPT_DIR) in sys.path:
        sys.path.remove(str(SCRIPT_DIR))
    sys.path.insert(0, str(SCRIPT_DIR))
    import hot_capacity_workspace_cuda as ext  # type: ignore

    return ext


def _tag32(text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) > 32:
        raise ValueError("tag too long")
    return raw + bytes(32 - len(raw))


FP64_BLOCK_TAG = _tag32("VERATHOS_FP64_C_BLOCK_V1")
FP64_LEAF_TAG = _tag32("VERATHOS_FP64_C_LEAF_V1")
FP64_NODE_TAG = _tag32("VERATHOS_FP64_C_NODE_V1")


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def fp64_value(*, seed: int, pass_index: int, round_index: int, matrix_id: int, row: int, col: int, matrix_dim: int) -> int:
    idx = int(row) * int(matrix_dim) + int(col)
    x = int(seed) & MASK64
    x ^= (int(pass_index) * PASS_MUL) & MASK64
    x ^= (int(round_index) * ROUND_MUL) & MASK64
    x ^= idx & MASK64
    domain = FP64_A_DOMAIN if int(matrix_id) == 0 else FP64_B_DOMAIN
    r = _splitmix64((x ^ domain) & MASK64)
    return 1 if (r & 1) else -1


def fp64_dot_value(*, seed: int, pass_index: int, round_index: int, row: int, col: int, matrix_dim: int) -> int:
    n = int(matrix_dim)
    acc = 0
    for k in range(n):
        acc += fp64_value(
            seed=seed,
            pass_index=pass_index,
            round_index=round_index,
            matrix_id=0,
            row=row,
            col=k,
            matrix_dim=n,
        ) * fp64_value(
            seed=seed,
            pass_index=pass_index,
            round_index=round_index,
            matrix_id=1,
            row=k,
            col=col,
            matrix_dim=n,
        )
    return acc


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def fp64_c_block_hash(block_i64: list[list[int]]) -> bytes:
    h = hashlib.sha256()
    h.update(FP64_BLOCK_TAG)
    for row in block_i64:
        for value in row:
            h.update(struct.pack("<d", float(int(value))))
    return h.digest()


def fp64_leaf_hash(block_hash: bytes) -> bytes:
    return _sha256(FP64_LEAF_TAG, block_hash)


def fp64_parent_hash(left: bytes, right: bytes) -> bytes:
    return _sha256(FP64_NODE_TAG, left, right)


def _level_sizes(leaves: int) -> list[int]:
    sizes: list[int] = []
    n = int(leaves)
    while True:
        sizes.append(n)
        if n == 1:
            return sizes
        n = (n + 1) >> 1


def _level_offsets(leaves: int) -> list[int]:
    offsets = [0]
    total = 0
    for size in _level_sizes(leaves)[:-1]:
        total += size
        offsets.append(total)
    return offsets


@dataclass
class MerklePath:
    leaf_index: int
    siblings: list[tuple[bytes, bool]]


class FlatMerkleTree:
    def __init__(self, flat_tree: bytes, num_leaves: int):
        if len(flat_tree) % 32 != 0:
            raise ValueError("flat tree is not 32-byte aligned")
        expected = sum(_level_sizes(int(num_leaves)))
        nodes = len(flat_tree) // 32
        if nodes != expected:
            raise ValueError(f"flat tree has {nodes} nodes, expected {expected}")
        self.num_leaves = int(num_leaves)
        self.nodes = [flat_tree[i * 32:(i + 1) * 32] for i in range(nodes)]
        self.offsets = _level_offsets(self.num_leaves)

    @property
    def root(self) -> bytes:
        return self.nodes[-1]

    def get_path(self, leaf_index: int) -> MerklePath:
        idx = int(leaf_index)
        if idx < 0 or idx >= self.num_leaves:
            raise ValueError("leaf_index out of range")
        siblings: list[tuple[bytes, bool]] = []
        sizes = _level_sizes(self.num_leaves)
        for level, level_size in enumerate(sizes[:-1]):
            if idx % 2 == 0:
                sibling_idx = idx + 1
                is_left = False
                if sibling_idx >= level_size:
                    sibling_idx = idx
            else:
                sibling_idx = idx - 1
                is_left = True
            siblings.append((self.nodes[self.offsets[level] + sibling_idx], is_left))
            idx >>= 1
        return MerklePath(leaf_index=int(leaf_index), siblings=siblings)


def serialize_path(path: MerklePath) -> dict[str, Any]:
    return {
        "leaf_index": int(path.leaf_index),
        "siblings": [[sibling.hex(), bool(is_left)] for sibling, is_left in path.siblings],
    }


def verify_path(root: bytes, leaf_hash: bytes, path_data: dict[str, Any]) -> bool:
    current = bytes(leaf_hash)
    for sibling_hex, is_left in path_data.get("siblings", []):
        sibling = bytes.fromhex(str(sibling_hex))
        if bool(is_left):
            current = fp64_parent_hash(sibling, current)
        else:
            current = fp64_parent_hash(current, sibling)
    return current == root


def seed_for(lease_id: str, gpu_index: int, seed_hex: str = "") -> int:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_SEED_V1")
    if seed_hex:
        raw = seed_hex[2:] if seed_hex.startswith("0x") else seed_hex
        h.update(bytes.fromhex(raw))
    h.update(lease_id.encode())
    h.update(int(gpu_index).to_bytes(4, "little"))
    return int.from_bytes(h.digest()[:8], "little")


def _tensor_u8_hex(tensor: torch.Tensor) -> str:
    return tensor.detach().cpu().contiguous().numpy().tobytes().hex()


def _tensor_u8_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def transcript_root(*, seed: int, matrix_dim: int, block_size: int, passes: int, rounds: int, root_chain: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_TRANSCRIPT_V1")
    for value in (seed, matrix_dim, block_size, passes, rounds):
        h.update(int(value).to_bytes(8, "little", signed=False))
    for pass_entry in root_chain:
        h.update(int(pass_entry["pass_index"]).to_bytes(8, "little", signed=False))
        for root_hex in pass_entry["round_roots"]:
            h.update(bytes.fromhex(str(root_hex)))
    return h.hexdigest()


def _challenge_bytes(challenge_hex: str) -> bytes:
    raw = str(challenge_hex).removeprefix("0x")
    return bytes.fromhex(raw)


def _derive_u64(label: bytes, challenge_hex: str, transcript_hex: str, counter: int = 0) -> int:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_FS_V1")
    h.update(label)
    h.update(_challenge_bytes(challenge_hex))
    h.update(bytes.fromhex(transcript_hex))
    h.update(int(counter).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little")


def derive_sample(*, challenge_hex: str, transcript_hex: str, passes: int, rounds: int, total_blocks: int, block_size: int, spot_checks: int) -> dict[str, Any]:
    pass_index = _derive_u64(b"pass", challenge_hex, transcript_hex) % int(passes)
    round_index = _derive_u64(b"round", challenge_hex, transcript_hex) % int(rounds)
    block_index = _derive_u64(b"block", challenge_hex, transcript_hex) % int(total_blocks)
    offsets = [
        _derive_u64(b"spot", challenge_hex, transcript_hex, i) % (int(block_size) * int(block_size))
        for i in range(int(spot_checks))
    ]
    return {
        "pass_index": int(pass_index),
        "round_index": int(round_index),
        "block_index": int(block_index),
        "spot_offsets": [int(v) for v in offsets],
    }


def root_chain_from_tensor(round_roots: torch.Tensor, passes: int, rounds: int) -> list[dict[str, Any]]:
    roots = [
        _tensor_u8_hex(round_roots[i])
        for i in range(int(passes) * int(rounds))
    ]
    return [
        {
            "pass_index": p,
            "round_roots": roots[p * int(rounds):(p + 1) * int(rounds)],
        }
        for p in range(int(passes))
    ]


def _sample_block(c: torch.Tensor, block_index: int, block_size: int) -> tuple[int, int, list[list[int]]]:
    n = int(c.size(0))
    blocks_per_row = n // int(block_size)
    bi = int(block_index) // blocks_per_row
    bj = int(block_index) % blocks_per_row
    block = c[
        bi * block_size:(bi + 1) * block_size,
        bj * block_size:(bj + 1) * block_size,
    ].detach().cpu().contiguous()
    values = [[int(round(float(v))) for v in row] for row in block.tolist()]
    return bi, bj, values


def build_fp64_identity_proof(
    *,
    ext: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    seed: int,
    matrix_dim: int,
    block_size: int,
    passes: int,
    rounds: int,
    round_roots: torch.Tensor,
    root_chain: list[dict[str, Any]],
    challenge_hex: str,
    transcript_hex: str,
    spot_checks: int,
    device_index: int,
) -> dict[str, Any]:
    blocks_per_row = int(matrix_dim) // int(block_size)
    total_blocks = blocks_per_row * blocks_per_row
    sample = derive_sample(
        challenge_hex=challenge_hex,
        transcript_hex=transcript_hex,
        passes=passes,
        rounds=rounds,
        total_blocks=total_blocks,
        block_size=block_size,
        spot_checks=spot_checks,
    )
    sampled_pass = int(sample["pass_index"])
    sampled_round = int(sample["round_index"])
    sampled_block = int(sample["block_index"])
    expected_round_root = root_chain[sampled_pass]["round_roots"][sampled_round]

    tmp_root = torch.empty((1, 32), dtype=torch.uint8, device=c.device)
    ext.run_fp64_identity_round_committed(
        a,
        b,
        c,
        tmp_root,
        int(seed),
        sampled_pass,
        sampled_round,
        int(block_size),
        int(device_index),
    )
    recomputed_root = _tensor_u8_hex(tmp_root[0])
    if recomputed_root != expected_round_root:
        raise RuntimeError(f"sampled FP64 round root mismatch: {recomputed_root} != {expected_round_root}")

    root_tensor, flat_tensor = ext.fp64_c_merkle_commit_f64(c, int(block_size), int(device_index))
    commit_root = _tensor_u8_hex(root_tensor)
    if commit_root != expected_round_root:
        raise RuntimeError(f"sampled FP64 commit root mismatch: {commit_root} != {expected_round_root}")

    bi, bj, c_block = _sample_block(c, sampled_block, int(block_size))
    flat_tree = _tensor_u8_bytes(flat_tensor)
    tree = FlatMerkleTree(flat_tree, total_blocks)
    if tree.root.hex() != expected_round_root:
        raise RuntimeError("flat tree root does not match sampled root")
    path = tree.get_path(sampled_block)
    block_hash = fp64_c_block_hash(c_block)
    leaf_hash = fp64_leaf_hash(block_hash)
    if tree.nodes[sampled_block] != leaf_hash:
        raise RuntimeError("sampled FP64 leaf hash does not match flat tree")

    spots: list[dict[str, int]] = []
    for offset in sample["spot_offsets"]:
        local_r = int(offset) // int(block_size)
        local_c = int(offset) % int(block_size)
        row = bi * int(block_size) + local_r
        col = bj * int(block_size) + local_c
        value = int(c_block[local_r][local_c])
        spots.append({
            "offset": int(offset),
            "row": int(row),
            "col": int(col),
            "value": value,
        })

    return {
        "format": PROOF_FORMAT,
        "workload_version": WORKLOAD_VERSION,
        "seed": int(seed),
        "matrix_dim": int(matrix_dim),
        "block_size": int(block_size),
        "passes": int(passes),
        "rounds": int(rounds),
        "root_chain": root_chain,
        "proof_challenge_seed": str(challenge_hex),
        "transcript_root": str(transcript_hex),
        "sampled": {
            "pass_index": sampled_pass,
            "round_index": sampled_round,
            "block_index": sampled_block,
            "bi": int(bi),
            "bj": int(bj),
        },
        "c_block_i64": c_block,
        "leaf_hash": leaf_hash.hex(),
        "merkle_path": serialize_path(path),
        "spots": spots,
    }


def verify_fp64_identity_proof(
    proof: dict[str, Any],
    *,
    expected_seed: int | None = None,
    expected_challenge_hex: str | None = None,
    expected_transcript_hex: str | None = None,
    expected_matrix_dim: int | None = None,
    expected_block_size: int | None = None,
    expected_passes: int | None = None,
    expected_rounds: int | None = None,
    expected_spot_checks: int | None = None,
) -> tuple[bool, str]:
    try:
        if proof.get("format") != PROOF_FORMAT:
            return False, "invalid_format"
        if proof.get("workload_version") != WORKLOAD_VERSION:
            return False, "invalid_workload_version"
        expected_fields = (
            ("seed", expected_seed),
            ("matrix_dim", expected_matrix_dim),
            ("block_size", expected_block_size),
            ("passes", expected_passes),
            ("rounds", expected_rounds),
        )
        for field_name, expected_value in expected_fields:
            if expected_value is None:
                continue
            if type(expected_value) is not int or expected_value < 0:
                return False, f"invalid_expected_{field_name}"
            actual_value = proof.get(field_name)
            if type(actual_value) is not int or actual_value != expected_value:
                return False, f"{field_name}_does_not_match_scheduled_workload"
        if expected_challenge_hex is not None:
            if (
                not isinstance(expected_challenge_hex, str)
                or proof.get("proof_challenge_seed") != expected_challenge_hex
            ):
                return False, "proof_challenge_seed_mismatch"
        if expected_transcript_hex is not None:
            if (
                not isinstance(expected_transcript_hex, str)
                or proof.get("transcript_root") != expected_transcript_hex
            ):
                return False, "transcript_root_parent_mismatch"
        if expected_spot_checks is not None:
            if (
                type(expected_spot_checks) is not int
                or expected_spot_checks <= 0
                or expected_spot_checks > 4096
            ):
                return False, "invalid_expected_spot_check_count"
            proof_spots = proof.get("spots")
            if not isinstance(proof_spots, list) or len(proof_spots) != expected_spot_checks:
                return False, "spot_check_count_does_not_match_scheduled_workload"
        seed = int(proof["seed"])
        matrix_dim = int(proof["matrix_dim"])
        block_size = int(proof["block_size"])
        passes = int(proof["passes"])
        rounds = int(proof["rounds"])
        if (
            matrix_dim <= 1
            or block_size <= 1
            or passes <= 0
            or rounds <= 0
            or matrix_dim % block_size
            or block_size & (block_size - 1)
        ):
            return False, "invalid_proof_parameters"
        root_chain = proof["root_chain"]
        if not isinstance(root_chain, list) or len(root_chain) != passes:
            return False, "root_chain_length_mismatch"
        for pass_index_expected, pass_entry in enumerate(root_chain):
            if not isinstance(pass_entry, dict):
                return False, "invalid_root_chain_entry"
            if type(pass_entry.get("pass_index")) is not int or pass_entry.get(
                "pass_index"
            ) != pass_index_expected:
                return False, "root_chain_pass_index_mismatch"
            round_roots = pass_entry.get("round_roots")
            if not isinstance(round_roots, list) or len(round_roots) != rounds:
                return False, "round_root_chain_length_mismatch"
        challenge_hex = str(proof["proof_challenge_seed"])
        transcript_hex = str(proof["transcript_root"])
        if transcript_root(
            seed=seed,
            matrix_dim=matrix_dim,
            block_size=block_size,
            passes=passes,
            rounds=rounds,
            root_chain=root_chain,
        ) != transcript_hex:
            return False, "transcript_root_mismatch"

        blocks_per_row = matrix_dim // block_size
        total_blocks = blocks_per_row * blocks_per_row
        sample = derive_sample(
            challenge_hex=challenge_hex,
            transcript_hex=transcript_hex,
            passes=passes,
            rounds=rounds,
            total_blocks=total_blocks,
            block_size=block_size,
            spot_checks=len(proof.get("spots") or []),
        )
        sampled = proof.get("sampled") or {}
        if int(sampled.get("pass_index", -1)) != int(sample["pass_index"]):
            return False, "sampled_pass_mismatch"
        if int(sampled.get("round_index", -1)) != int(sample["round_index"]):
            return False, "sampled_round_mismatch"
        if int(sampled.get("block_index", -1)) != int(sample["block_index"]):
            return False, "sampled_block_mismatch"
        pass_index = int(sample["pass_index"])
        round_index = int(sample["round_index"])
        block_index = int(sample["block_index"])
        expected_root_hex = str(root_chain[pass_index]["round_roots"][round_index])
        expected_root = bytes.fromhex(expected_root_hex)

        c_block = proof.get("c_block_i64")
        if not isinstance(c_block, list) or len(c_block) != block_size:
            return False, "invalid_c_block_shape"
        if any(not isinstance(row, list) or len(row) != block_size for row in c_block):
            return False, "invalid_c_block_shape"
        c_block_i64 = [[int(v) for v in row] for row in c_block]
        leaf_hash = fp64_leaf_hash(fp64_c_block_hash(c_block_i64))
        if leaf_hash.hex() != str(proof.get("leaf_hash") or ""):
            return False, "leaf_hash_mismatch"
        merkle_path = proof.get("merkle_path") or {}
        expected_leaf_index = block_index
        if (
            not isinstance(merkle_path, dict)
            or type(merkle_path.get("leaf_index")) is not int
            or merkle_path.get("leaf_index") != expected_leaf_index
        ):
            return False, "merkle_leaf_index_mismatch"
        siblings = merkle_path.get("siblings")
        expected_path_length = len(_level_sizes(total_blocks)) - 1
        if not isinstance(siblings, list) or len(siblings) != expected_path_length:
            return False, "merkle_path_length_mismatch"
        node_index = expected_leaf_index
        for sibling in siblings:
            if not isinstance(sibling, list) or len(sibling) != 2:
                return False, "invalid_merkle_sibling"
            sibling_hex, is_left = sibling
            if (
                not isinstance(sibling_hex, str)
                or len(sibling_hex) != 64
                or not isinstance(is_left, bool)
            ):
                return False, "invalid_merkle_sibling"
            try:
                sibling_bytes = bytes.fromhex(sibling_hex)
            except ValueError:
                return False, "invalid_merkle_sibling"
            if sibling_hex != sibling_bytes.hex():
                return False, "invalid_merkle_sibling"
            if is_left != bool(node_index & 1):
                return False, "merkle_path_direction_mismatch"
            node_index //= 2
        if not verify_path(expected_root, leaf_hash, merkle_path):
            return False, "merkle_path_invalid"

        bi = block_index // blocks_per_row
        bj = block_index % blocks_per_row
        if int(sampled.get("bi", -1)) != bi or int(sampled.get("bj", -1)) != bj:
            return False, "sampled_block_coordinates_mismatch"
        spots = proof.get("spots") or []
        if not spots:
            return False, "missing_spot_checks"
        if len(spots) != len(sample["spot_offsets"]):
            return False, "spot_count_mismatch"
        for spot, expected_offset in zip(spots, sample["spot_offsets"]):
            offset = int(spot.get("offset", -1))
            if offset != int(expected_offset):
                return False, "spot_offset_mismatch"
            local_r = offset // block_size
            local_c = offset % block_size
            row = bi * block_size + local_r
            col = bj * block_size + local_c
            if int(spot.get("row", -1)) != row or int(spot.get("col", -1)) != col:
                return False, "spot_coordinates_mismatch"
            block_value = int(c_block_i64[local_r][local_c])
            if int(spot.get("value", 0)) != block_value:
                return False, "spot_value_block_mismatch"
            expected = fp64_dot_value(
                seed=seed,
                pass_index=pass_index,
                round_index=round_index,
                row=row,
                col=col,
                matrix_dim=matrix_dim,
            )
            if block_value != expected:
                return False, "spot_dot_mismatch"
        return True, "ok"
    except Exception as exc:
        return False, f"exception:{type(exc).__name__}:{exc}"


def _tamper_tests(proof: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    mutations: dict[str, Any] = {}

    c_value = copy.deepcopy(proof)
    c_value["c_block_i64"][0][0] = int(c_value["c_block_i64"][0][0]) + 1
    mutations["c_value"] = c_value

    leaf = copy.deepcopy(proof)
    leaf["leaf_hash"] = ("00" if str(leaf["leaf_hash"])[:2] != "00" else "ff") + str(leaf["leaf_hash"])[2:]
    mutations["leaf_hash"] = leaf

    root = copy.deepcopy(proof)
    sampled = root["sampled"]
    p = int(sampled["pass_index"])
    r = int(sampled["round_index"])
    old_root = str(root["root_chain"][p]["round_roots"][r])
    root["root_chain"][p]["round_roots"][r] = ("00" if old_root[:2] != "00" else "ff") + old_root[2:]
    mutations["round_root"] = root

    for name, mutated in mutations.items():
        ok, reason = verify_fp64_identity_proof(mutated)
        results[name] = {
            "rejected": not ok,
            "verify_ok": bool(ok),
            "reason": reason,
        }
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FP64 identity benchmark")
    ext = _import_ext()
    device_index = int(args.device_index)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    n = int(args.matrix_dim)
    if n % int(args.block_size) != 0:
        raise ValueError("--matrix-dim must be divisible by --block-size")
    seed = seed_for(args.lease_id, int(args.gpu_index), args.seed_hex)

    a = torch.empty((n, n), dtype=torch.float64, device=device)
    b = torch.empty((n, n), dtype=torch.float64, device=device)
    c = torch.empty((n, n), dtype=torch.float64, device=device)
    rounds = int(args.rounds)
    requested_passes = int(args.passes)
    effective_passes = requested_passes
    calibration: dict[str, Any] = {
        "enabled": bool(args.calibrate_passes),
        "requested_passes": requested_passes,
    }

    if args.calibrate_passes:
        warmup_passes = max(1, int(args.calibration_warmup_passes))
        warmup_roots = torch.empty((max(1, warmup_passes * rounds), 32), dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        warmup_wall_start = time.perf_counter()
        warmup_ms = ext.run_fp64_identity_chain_committed(
            a,
            b,
            c,
            warmup_roots,
            int(seed),
            0,
            warmup_passes,
            rounds,
            int(args.block_size),
            device_index,
        )
        torch.cuda.synchronize(device)
        warmup_wall_s = time.perf_counter() - warmup_wall_start
        mean_round_s = (float(warmup_ms) / 1000.0) / max(1, warmup_passes * rounds)
        if float(args.target_timed_s) > 0:
            target_timed_s = float(args.target_timed_s)
            target_source = "explicit"
        else:
            target_timed_s = float(args.deadline_s) * (1.0 - float(args.deadline_margin_pct) / 100.0)
            target_source = "deadline_margin"
        target_rounds = max(1, int(target_timed_s / max(mean_round_s, 1e-9)))
        effective_passes = max(1, target_rounds // max(1, rounds))
        calibration = {
            "enabled": True,
            "requested_passes": requested_passes,
            "warmup_passes": warmup_passes,
            "warmup_rounds": rounds,
            "warmup_timed_s": float(warmup_ms) / 1000.0,
            "warmup_wall_s": warmup_wall_s,
            "warmup_mean_round_s": mean_round_s,
            "deadline_s": float(args.deadline_s),
            "deadline_margin_pct": float(args.deadline_margin_pct),
            "target_timed_s": target_timed_s,
            "target_source": target_source,
            "calibrated_passes": effective_passes,
            "estimated_timed_s": effective_passes * rounds * mean_round_s,
        }

    round_roots = torch.empty((max(1, effective_passes * rounds), 32), dtype=torch.uint8, device=device)
    hot_warmup_passes = 0 if args.calibrate_passes else max(0, int(args.warmup_passes))
    hot_warmup_timed_s = 0.0
    hot_warmup_wall_s = 0.0
    if hot_warmup_passes:
        warmup_roots = torch.empty((max(1, hot_warmup_passes * rounds), 32), dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        hot_warmup_started = time.perf_counter()
        hot_warmup_ms = ext.run_fp64_identity_chain_committed(
            a,
            b,
            c,
            warmup_roots,
            int(seed) ^ 0x5A5AA5A5DDBCA9D6,
            0,
            hot_warmup_passes,
            rounds,
            int(args.block_size),
            device_index,
        )
        torch.cuda.synchronize(device)
        hot_warmup_wall_s = time.perf_counter() - hot_warmup_started
        hot_warmup_timed_s = float(hot_warmup_ms) / 1000.0

    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    timed_ms = ext.run_fp64_identity_chain_committed(
        a,
        b,
        c,
        round_roots,
        int(seed),
        0,
        effective_passes,
        rounds,
        int(args.block_size),
        device_index,
    )
    torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - wall_start
    root_chain = root_chain_from_tensor(round_roots, effective_passes, rounds)
    transcript_hex = transcript_root(
        seed=seed,
        matrix_dim=n,
        block_size=int(args.block_size),
        passes=effective_passes,
        rounds=rounds,
        root_chain=root_chain,
    )
    if args.challenge_hex:
        challenge_hex = args.challenge_hex.removeprefix("0x")
        challenge_source = "provided"
    else:
        # Local harness only. Production must use a post-timing unpredictable
        # value such as B_proof/block randomness. A transcript-only fallback
        # would be grindable by the prover, so do not simulate it that way.
        challenge_hex = secrets.token_hex(32)
        challenge_source = "local_random_simulated_b_proof"

    proof_payload: dict[str, Any] = {}
    proof_payload_s = 0.0
    verify_ok = None
    verify_reason = ""
    verify_s = 0.0
    tamper: dict[str, Any] = {}
    if not args.skip_proof_payload:
        proof_start = time.perf_counter()
        proof_payload = build_fp64_identity_proof(
            ext=ext,
            a=a,
            b=b,
            c=c,
            seed=seed,
            matrix_dim=n,
            block_size=int(args.block_size),
            passes=effective_passes,
            rounds=rounds,
            round_roots=round_roots,
            root_chain=root_chain,
            challenge_hex=challenge_hex,
            transcript_hex=transcript_hex,
            spot_checks=int(args.spot_checks),
            device_index=device_index,
        )
        proof_payload_s = time.perf_counter() - proof_start
        if args.verify_proof:
            verify_start = time.perf_counter()
            verify_ok, verify_reason = verify_fp64_identity_proof(proof_payload)
            verify_s = time.perf_counter() - verify_start
        if args.tamper_test:
            tamper = _tamper_tests(proof_payload)

    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    summary = {
        "workload_version": WORKLOAD_VERSION,
        "proof_format": PROOF_FORMAT,
        "seed": int(seed),
        "matrix_dim": n,
        "block_size": int(args.block_size),
        "passes": effective_passes,
        "requested_passes": requested_passes,
        "rounds": rounds,
        "spot_checks": int(args.spot_checks),
        "calibration": calibration,
        "warmup_passes": hot_warmup_passes,
        "warmup_timed_s": hot_warmup_timed_s,
        "warmup_wall_s": hot_warmup_wall_s,
        "timed_elapsed_s": float(timed_ms) / 1000.0,
        "timed_cuda_component_s": float(timed_ms) / 1000.0,
        "timed_cuda_ms": float(timed_ms),
        "wall_elapsed_s": wall_s,
        "mean_round_ms": float(timed_ms) / max(1, effective_passes * rounds),
        "peak_reserved_mib": peak_reserved_mib,
        "peak_allocated_mib": peak_allocated_mib,
        "root_chain": root_chain,
        "transcript_root": transcript_hex,
        "challenge_seed": challenge_hex,
        "challenge_source": challenge_source,
        "challenge_security_note": (
            "Production must derive this from post-timing unpredictable B_proof/"
            "block randomness plus the committed transcript. Local random is "
            "only a harness simulation."
        ),
        "proof_payload_s": proof_payload_s,
        "verify_s": verify_s,
        "verify_ok": verify_ok,
        "verify_reason": verify_reason,
        "tamper_tests": tamper,
        "proof_payload": proof_payload,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-dim", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument(
        "--warmup-passes",
        type=int,
        default=1,
        help="hot calibration: run this many untimed passes before measured work",
    )
    p.add_argument(
        "--calibrate-passes",
        action="store_true",
        help="measure a short warmup and choose passes for the target timed window",
    )
    p.add_argument("--calibration-warmup-passes", type=int, default=3)
    p.add_argument("--deadline-s", type=float, default=30.0)
    p.add_argument("--deadline-margin-pct", type=float, default=25.0)
    p.add_argument(
        "--target-timed-s",
        type=float,
        default=0.0,
        help="override deadline/margin and target this timed CUDA duration",
    )
    p.add_argument("--spot-checks", type=int, default=1)
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--lease-id", default="local-fp64-identity")
    p.add_argument("--seed-hex", default="")
    p.add_argument("--challenge-hex", default="")
    p.add_argument("--skip-proof-payload", action="store_true")
    p.add_argument("--verify-proof", action="store_true")
    p.add_argument("--tamper-test", action="store_true")
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
