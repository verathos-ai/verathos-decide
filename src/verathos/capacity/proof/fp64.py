"""FP64 identity lane: deterministic +-1 FP64 GEMM rounds with a committed C-block Merkle root per round.

Server-class GPUs (A100/H100 and descendants) run FP64 at a rate consumer cards cannot match, so a timed FP64
lane separates the two. Timed work commits every round root before the challenge is known; afterwards one
round and one C block are opened and a few entries are recomputed on the CPU as exact dot products.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any

WORKLOAD_VERSION = "fp64_identity_c_merkle_v1"
PROOF_FORMAT = "fp64_identity_merkle_dot_v1"
MASK64 = (1 << 64) - 1
PASS_MUL = 0xD1B54A32D192ED03
ROUND_MUL = 0xABC98388FB8FAC03
FP64_A_DOMAIN = 0xDDBCA9D6B4A7A1D9
FP64_B_DOMAIN = 0xA1F0E27C5C314DD1


def _tag32(text: str) -> bytes:
    raw = text.encode("ascii")
    return raw + bytes(32 - len(raw))


FP64_BLOCK_TAG = _tag32("VERATHOS_FP64_C_BLOCK_V1")
FP64_LEAF_TAG = _tag32("VERATHOS_FP64_C_LEAF_V1")
FP64_NODE_TAG = _tag32("VERATHOS_FP64_C_NODE_V1")

_DATA_ERRORS = (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def _sha256(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def seed_for(lease_id: str, gpu_index: int, seed_hex: str = "") -> int:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_SEED_V1")
    if seed_hex:
        h.update(bytes.fromhex(seed_hex.removeprefix("0x")))
    h.update(lease_id.encode())
    h.update(int(gpu_index).to_bytes(4, "little"))
    return int.from_bytes(h.digest()[:8], "little")


def fp64_value(*, seed: int, pass_index: int, round_index: int, matrix_id: int, row: int, col: int,
               matrix_dim: int) -> int:
    x = int(seed) & MASK64
    x ^= (int(pass_index) * PASS_MUL) & MASK64
    x ^= (int(round_index) * ROUND_MUL) & MASK64
    x ^= (int(row) * int(matrix_dim) + int(col)) & MASK64
    domain = FP64_A_DOMAIN if int(matrix_id) == 0 else FP64_B_DOMAIN
    return 1 if _splitmix64(x ^ domain) & 1 else -1


def fp64_dot_value(*, seed: int, pass_index: int, round_index: int, row: int, col: int, matrix_dim: int) -> int:
    common = {"seed": seed, "pass_index": pass_index, "round_index": round_index, "matrix_dim": matrix_dim}
    return sum(fp64_value(**common, matrix_id=0, row=row, col=k) * fp64_value(**common, matrix_id=1, row=k, col=col)
               for k in range(int(matrix_dim)))


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


def level_sizes(leaves: int) -> list[int]:
    sizes, n = [], int(leaves)
    while True:
        sizes.append(n)
        if n == 1:
            return sizes
        n = (n + 1) >> 1


def transcript_root(*, seed: int, matrix_dim: int, block_size: int, passes: int, rounds: int,
                    root_chain: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_TRANSCRIPT_V1")
    for value in (seed, matrix_dim, block_size, passes, rounds):
        h.update(int(value).to_bytes(8, "little", signed=False))
    for pass_entry in root_chain:
        h.update(int(pass_entry["pass_index"]).to_bytes(8, "little", signed=False))
        for root_hex in pass_entry["round_roots"]:
            h.update(bytes.fromhex(str(root_hex)))
    return h.hexdigest()


def _derive_u64(label: bytes, challenge_hex: str, transcript_hex: str, counter: int = 0) -> int:
    h = hashlib.sha256()
    h.update(b"VERATHOS_FP64_IDENTITY_FS_V1")
    h.update(label)
    h.update(bytes.fromhex(str(challenge_hex).removeprefix("0x")))
    h.update(bytes.fromhex(transcript_hex))
    h.update(int(counter).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little")


def derive_sample(*, challenge_hex: str, transcript_hex: str, passes: int, rounds: int, total_blocks: int,
                  block_size: int, spot_checks: int) -> dict[str, Any]:
    """Round, block and in-block spot offsets opened for a given post-timing challenge."""
    return {
        "pass_index": _derive_u64(b"pass", challenge_hex, transcript_hex) % int(passes),
        "round_index": _derive_u64(b"round", challenge_hex, transcript_hex) % int(rounds),
        "block_index": _derive_u64(b"block", challenge_hex, transcript_hex) % int(total_blocks),
        "spot_offsets": [_derive_u64(b"spot", challenge_hex, transcript_hex, i) % (int(block_size) ** 2)
                         for i in range(int(spot_checks))],
    }


def _verify_path(root: bytes, leaf_hash: bytes, leaf_index: int, siblings: Any, expected_length: int) -> str:
    if not isinstance(siblings, list) or len(siblings) != expected_length:
        return "merkle_path_length_mismatch"
    current, node_index = leaf_hash, leaf_index
    for sibling in siblings:
        if not isinstance(sibling, list) or len(sibling) != 2:
            return "invalid_merkle_sibling"
        sibling_hex, is_left = sibling
        if not isinstance(sibling_hex, str) or len(sibling_hex) != 64 or not isinstance(is_left, bool):
            return "invalid_merkle_sibling"
        sibling_bytes = bytes.fromhex(sibling_hex)
        if sibling_hex != sibling_bytes.hex():
            return "invalid_merkle_sibling"
        if is_left != bool(node_index & 1):
            return "merkle_path_direction_mismatch"
        current = fp64_parent_hash(sibling_bytes, current) if is_left else fp64_parent_hash(current, sibling_bytes)
        node_index //= 2
    return "" if current == root else "merkle_path_invalid"


def verify_fp64_proof(proof: Mapping[str, Any], *, expected_seed: int, expected_challenge_hex: str,
                      expected_transcript_hex: str, expected_params: Mapping[str, Any],
                      expected_spot_checks: int) -> tuple[bool, str]:
    """Verify an FP64 lane opening against the scheduled workload and the combined challenge."""
    try:
        return _verify(proof, expected_seed, expected_challenge_hex, expected_transcript_hex, expected_params,
                       expected_spot_checks)
    except _DATA_ERRORS:
        return False, "malformed_fp64_proof"


def _verify(proof: Mapping[str, Any], expected_seed: int, expected_challenge_hex: str, expected_transcript_hex: str,
            expected_params: Mapping[str, Any], expected_spot_checks: int) -> tuple[bool, str]:
    if proof.get("format") != PROOF_FORMAT or proof.get("workload_version") != WORKLOAD_VERSION:
        return False, "invalid_format"
    expected_fields = {"seed": expected_seed, **{k: expected_params.get(k)
                                                 for k in ("matrix_dim", "block_size", "passes", "rounds")}}
    for name, expected in expected_fields.items():
        actual = proof.get(name)
        if type(expected) is not int or type(actual) is not int or actual != expected:
            return False, f"{name}_does_not_match_scheduled_workload"
    if proof.get("proof_challenge_seed") != expected_challenge_hex:
        return False, "proof_challenge_seed_mismatch"
    if proof.get("transcript_root") != expected_transcript_hex:
        return False, "transcript_root_parent_mismatch"
    spots = proof.get("spots")
    if not isinstance(spots, list) or len(spots) != expected_spot_checks or not spots:
        return False, "spot_check_count_does_not_match_scheduled_workload"

    seed, matrix_dim, block_size = expected_seed, proof["matrix_dim"], proof["block_size"]
    passes, rounds = proof["passes"], proof["rounds"]
    if matrix_dim <= 1 or block_size <= 1 or passes <= 0 or rounds <= 0 or matrix_dim % block_size \
            or block_size & (block_size - 1):
        return False, "invalid_proof_parameters"
    root_chain = proof["root_chain"]
    if not isinstance(root_chain, list) or len(root_chain) != passes:
        return False, "root_chain_length_mismatch"
    for index, entry in enumerate(root_chain):
        if not isinstance(entry, dict) or type(entry.get("pass_index")) is not int or entry["pass_index"] != index:
            return False, "root_chain_pass_index_mismatch"
        if not isinstance(entry.get("round_roots"), list) or len(entry["round_roots"]) != rounds:
            return False, "round_root_chain_length_mismatch"
    if transcript_root(seed=seed, matrix_dim=matrix_dim, block_size=block_size, passes=passes, rounds=rounds,
                       root_chain=root_chain) != expected_transcript_hex:
        return False, "transcript_root_mismatch"

    blocks_per_row = matrix_dim // block_size
    total_blocks = blocks_per_row * blocks_per_row
    sample = derive_sample(challenge_hex=expected_challenge_hex, transcript_hex=expected_transcript_hex,
                           passes=passes, rounds=rounds, total_blocks=total_blocks, block_size=block_size,
                           spot_checks=expected_spot_checks)
    pass_index, round_index, block_index = sample["pass_index"], sample["round_index"], sample["block_index"]
    bi, bj = divmod(block_index, blocks_per_row)
    sampled = proof.get("sampled") or {}
    if (sampled.get("pass_index"), sampled.get("round_index"), sampled.get("block_index"), sampled.get("bi"),
            sampled.get("bj")) != (pass_index, round_index, block_index, bi, bj):
        return False, "sampled_index_mismatch"

    c_block = proof.get("c_block_i64")
    if not isinstance(c_block, list) or len(c_block) != block_size \
            or any(not isinstance(row, list) or len(row) != block_size for row in c_block):
        return False, "invalid_c_block_shape"
    c_block = [[int(v) for v in row] for row in c_block]
    leaf_hash = fp64_leaf_hash(fp64_c_block_hash(c_block))
    if leaf_hash.hex() != proof.get("leaf_hash"):
        return False, "leaf_hash_mismatch"
    path = proof.get("merkle_path") or {}
    if not isinstance(path, dict) or type(path.get("leaf_index")) is not int or path["leaf_index"] != block_index:
        return False, "merkle_leaf_index_mismatch"
    root = bytes.fromhex(str(root_chain[pass_index]["round_roots"][round_index]))
    reason = _verify_path(root, leaf_hash, block_index, path.get("siblings"), len(level_sizes(total_blocks)) - 1)
    if reason:
        return False, reason

    for spot, offset in zip(spots, sample["spot_offsets"]):
        local_r, local_c = divmod(offset, block_size)
        row, col = bi * block_size + local_r, bj * block_size + local_c
        if (spot.get("offset"), spot.get("row"), spot.get("col")) != (offset, row, col):
            return False, "spot_coordinates_mismatch"
        value = c_block[local_r][local_c]
        if spot.get("value") != value:
            return False, "spot_value_block_mismatch"
        if value != fp64_dot_value(seed=seed, pass_index=pass_index, round_index=round_index, row=row, col=col,
                                   matrix_dim=matrix_dim):
            return False, "spot_dot_mismatch"
    return True, "ok"
