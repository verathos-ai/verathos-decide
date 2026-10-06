"""Combined capacity proof: the capacity lane, the optional capacity tail and the optional FP64 identity lane bound
into one transcript and opened with one post-timing sample seed.

The timed work commits every lane's root chain into ``combined_transcript_root`` before B_proof exists. The
validator derives ``b_proof_seed`` from finalized chain data, the sample seed is ``H(b_proof_seed, transcript)``, and
the same seed selects the opening in every lane. Each capacity lane carries a sumcheck block proof over its sampled
Q block and a polynomial-commitment GEMM proof (``proof.gemm_v2``) over the same block.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any, Mapping

from verathos.capacity.proof import fp64
from verathos.capacity.proof.gemm_v2 import (
    CapacityGemmV2FormatError,
    capacity_gemm_v2_context_from_workspace_proof,
    verify_capacity_gemm_v2,
)
from verathos.capacity.proof.workspace import verify_workspace_proof_payload

COMBINED_PROOF_FORMAT = "capacity_combined_proof_v1"
WORKLOAD_VERSION = "hot_capacity_combined"
_TRANSCRIPT_DOMAIN = b"VERATHOS_CAPACITY_COMBINED_TRANSCRIPT_V1"
_SAMPLE_DOMAIN = b"VERATHOS_CAPACITY_COMBINED_SAMPLE_V1"
_DATA_ERRORS = (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError, struct.error)
_COMMITTED_ROOTS = ("capacity_transcript_root", "capacity_tail_transcript_root", "fp64_transcript_root")
_COMMITTED_PARAMS = ("capacity_params", "capacity_tail_params", "fp64_params")


def combined_transcript_root(*, capacity_transcript: str, capacity_tail_transcript: str, fp64_transcript: str,
                             capacity_params: Mapping[str, Any], capacity_tail_params: Mapping[str, Any] | None,
                             fp64_params: Mapping[str, Any]) -> str:
    payload = json.dumps({
        "version": WORKLOAD_VERSION,
        "capacity_transcript": capacity_transcript,
        "capacity_tail_transcript": capacity_tail_transcript,
        "fp64_transcript": fp64_transcript,
        "capacity_params": capacity_params,
        "capacity_tail_params": capacity_tail_params,
        "fp64_params": fp64_params,
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(_TRANSCRIPT_DOMAIN + payload).hexdigest()


def sample_seed(*, b_proof_seed: str, combined_transcript: str) -> str:
    return hashlib.sha256(_SAMPLE_DOMAIN + bytes.fromhex(b_proof_seed.removeprefix("0x"))
                          + bytes.fromhex(combined_transcript)).hexdigest()


def lane_params(workload_spec: Mapping[str, Any]) -> tuple[dict[str, int], dict[str, int] | None, dict[str, int], int]:
    """Per-lane parameters and the spot-check count of a scheduled workload spec (``core.gpu_workload_spec``)."""
    if workload_spec.get("workload_version") != WORKLOAD_VERSION:
        raise ValueError("unsupported workload version")

    def field(name: str, minimum: int = 0) -> int:
        value = workload_spec.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"workload field {name} is invalid")
        return value

    capacity = {
        "matrix_dim": field("capacity_matrix_dim", 2),
        "block_size": field("capacity_block_size", 2),
        "passes": field("capacity_passes", 1),
        "rounds": field("capacity_rounds", 1),
        "transition_mix_rounds": field("transition_mix_rounds"),
        "transition_fanout": field("transition_fanout", 1),
    }
    tail = None
    if field("capacity_tail_passes"):
        tail = {
            "matrix_dim": capacity["matrix_dim"],
            "block_size": capacity["block_size"],
            "passes": field("capacity_tail_passes"),
            "rounds": field("capacity_tail_rounds", 1),
            "transition_mix_rounds": field("capacity_tail_transition_mix_rounds", 1),
            "transition_fanout": field("capacity_tail_transition_fanout", 1),
        }
    fp64_lane = {
        "matrix_dim": field("fp64_matrix_dim", 2),
        "block_size": field("fp64_block_size", 2),
        "passes": field("fp64_passes"),
        "rounds": field("fp64_rounds", 1),
    }
    pass_count = capacity["passes"] + (tail["passes"] if tail else 0) + fp64_lane["passes"]
    if field("pass_count", 1) != pass_count:
        raise ValueError("workload pass_count is inconsistent")
    spot_checks = field("spot_checks", 1)
    if spot_checks > 4096:
        raise ValueError("workload spot_checks is invalid")
    return capacity, tail, fp64_lane, spot_checks


def final_commitment(final_timing: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the bench's final timing that the signed final receipt commits to."""
    keys = ("workload_version", "pass_count", "combined_transcript_root", *_COMMITTED_ROOTS, *_COMMITTED_PARAMS,
            "timed_cuda_component_s", "timed_wall_s")
    return {"format": COMBINED_PROOF_FORMAT, **{k: final_timing.get(k) for k in keys}}


def _verify_capacity_lane(*, proof: Mapping[str, Any], params: Mapping[str, Any], transcript_root: str,
                          combined_root: str, lease_id: str, gpu_index: int, proof_seed_hex: str, sample_seed_hex: str,
                          spot_checks: int) -> tuple[bool, str]:
    chain = proof.get("root_chain") or [{}]
    ok, reason = verify_workspace_proof_payload(
        proof=proof,
        expected_transcript_root=transcript_root,
        expected_pass0_root=str(chain[0].get("pass_root") or ""),
        expected_final_root=str(chain[-1].get("pass_root") or ""),
        lease_id=lease_id,
        gpu_index=gpu_index,
        proof_seed_hex=proof_seed_hex,
        pass_count=int(params["passes"]),
        proof_challenge_seed_hex=sample_seed_hex,
        expected_spot_checks=spot_checks,
    )
    if not ok:
        return False, reason
    state_openings = proof.get("gemm_v2_state_openings")
    block_proof = proof.get("block_proof")
    if "gemm_v2" not in proof or not isinstance(state_openings, list) or not isinstance(block_proof, Mapping) \
            or not isinstance(block_proof.get("merkle_path"), Mapping):
        return False, "missing_gemm_proof"
    try:
        context = capacity_gemm_v2_context_from_workspace_proof(
            proof=proof, capacity_params=params, lease_id=lease_id, gpu_index=gpu_index,
            proof_seed_hex=proof_seed_hex, challenge_seed_hex=sample_seed_hex, transcript_root_hex=combined_root)
    except CapacityGemmV2FormatError as exc:
        return False, f"invalid_gemm_context:{exc}"
    ok, reason = verify_capacity_gemm_v2(context=context, payload=proof.get("gemm_v2"),
                                         q_block_i32=block_proof.get("q_block_i32"),
                                         q_merkle_path=block_proof.get("merkle_path"), state_openings=state_openings)
    return (True, "ok") if ok else (False, f"gemm_invalid:{reason}")


def verify_combined_proof(*, proof: Mapping[str, Any], final_commit: Mapping[str, Any], workload_spec: Mapping[str, Any],
                          lease_id: str, gpu_index: int, proof_seed_hex: str, b_proof_seed_hex: str) -> tuple[bool, str]:
    """Verify a combined proof against the signed final commitment and the validator's own workload spec.

    ``b_proof_seed_hex`` is the challenge seed the validator derives from the finalized B_proof block.
    """
    try:
        return _verify(proof, final_commit, workload_spec, lease_id, gpu_index, proof_seed_hex, b_proof_seed_hex)
    except _DATA_ERRORS:
        return False, "malformed_combined_proof"


def _verify(proof: Mapping[str, Any], final_commit: Mapping[str, Any], workload_spec: Mapping[str, Any], lease_id: str,
            gpu_index: int, proof_seed_hex: str, b_proof_seed_hex: str) -> tuple[bool, str]:
    if proof.get("format") != COMBINED_PROOF_FORMAT or proof.get("workload_version") != WORKLOAD_VERSION:
        return False, "unsupported_combined_format"
    if final_commit.get("format") != COMBINED_PROOF_FORMAT:
        return False, "missing_final_combined_commitment"
    combined_root = str(proof.get("combined_transcript_root") or "")
    if not combined_root or final_commit.get("combined_transcript_root") != combined_root:
        return False, "final_combined_transcript_root_mismatch"
    for key in (*_COMMITTED_ROOTS, *_COMMITTED_PARAMS):
        if final_commit.get(key) != proof.get(key):
            return False, f"final_{key}_mismatch"

    capacity, tail, fp64_lane, spot_checks = lane_params(workload_spec)
    if proof.get("capacity_params") != capacity:
        return False, "capacity_params_do_not_match_scheduled_workload"
    if proof.get("capacity_tail_params") != tail:
        return False, "capacity_tail_params_do_not_match_scheduled_workload"
    if proof.get("fp64_params") != fp64_lane:
        return False, "fp64_params_do_not_match_scheduled_workload"
    if proof.get("pass_count") != workload_spec["pass_count"]:
        return False, "pass_count_does_not_match_scheduled_workload"
    recomputed = combined_transcript_root(
        capacity_transcript=str(proof.get("capacity_transcript_root") or ""),
        capacity_tail_transcript=str(proof.get("capacity_tail_transcript_root") or ""),
        fp64_transcript=str(proof.get("fp64_transcript_root") or ""),
        capacity_params=capacity, capacity_tail_params=tail, fp64_params=fp64_lane)
    if recomputed != combined_root:
        return False, "recomputed_combined_transcript_mismatch"

    b_proof_seed = b_proof_seed_hex.removeprefix("0x")
    if str(proof.get("b_proof_seed") or "").removeprefix("0x") != b_proof_seed:
        return False, "b_proof_seed_mismatch"
    seed = sample_seed(b_proof_seed=b_proof_seed, combined_transcript=combined_root)
    if proof.get("sample_seed") != seed:
        return False, "sample_seed_mismatch"

    common = {"combined_root": combined_root, "lease_id": lease_id, "gpu_index": gpu_index,
              "proof_seed_hex": proof_seed_hex, "sample_seed_hex": seed, "spot_checks": spot_checks}
    ok, reason = _verify_capacity_lane(proof=proof.get("capacity_proof") or {}, params=capacity,
                                       transcript_root=str(proof.get("capacity_transcript_root") or ""), **common)
    if not ok:
        return False, f"capacity_{reason}"
    if tail is not None:
        ok, reason = _verify_capacity_lane(proof=proof.get("capacity_tail_proof") or {}, params=tail,
                                           transcript_root=str(proof.get("capacity_tail_transcript_root") or ""),
                                           **common)
        if not ok:
            return False, f"capacity_tail_{reason}"
    elif proof.get("capacity_tail_proof"):
        return False, "unexpected_capacity_tail_proof"

    if fp64_lane["passes"]:
        ok, reason = fp64.verify_fp64_proof(
            proof.get("fp64_proof") or {},
            expected_seed=fp64.seed_for(lease_id, gpu_index, proof_seed_hex),
            expected_challenge_hex=seed,
            expected_transcript_hex=str(proof.get("fp64_transcript_root") or ""),
            expected_params=fp64_lane,
            expected_spot_checks=spot_checks,
        )
        if not ok:
            return False, f"fp64_{reason}"
    elif proof.get("fp64_proof"):
        return False, "unexpected_fp64_proof"
    return True, "ok"


__all__ = [
    "COMBINED_PROOF_FORMAT",
    "WORKLOAD_VERSION",
    "combined_transcript_root",
    "final_commitment",
    "lane_params",
    "sample_seed",
    "verify_combined_proof",
]
