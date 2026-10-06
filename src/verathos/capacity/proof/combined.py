"""Combined hot-capacity audit proof verification."""

from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys
from typing import Any, Mapping

from verathos.capacity.proof.gemm_v2 import (
    CAPACITY_GEMM_V2_PROTOCOL_VERSION,
    CapacityGemmV2FormatError,
    capacity_gemm_v2_context_from_workspace_proof,
    verify_capacity_gemm_v2,
)
from verathos.capacity.proof.workspace import verify_workspace_proof_payload


COMBINED_PROOF_FORMAT_V1 = "hot_capacity_combined_proof_v1"
COMBINED_PROOF_FORMAT_V2 = "hot_capacity_combined_proof_v2"
# Backward-compatible name retained for callers and legacy fixtures.
COMBINED_PROOF_FORMAT = COMBINED_PROOF_FORMAT_V1
COMBINED_WORKLOAD_VERSION = "hot_capacity_combined"
LEGACY_COMBINED_PROOF_PROTOCOL_VERSION = 1
CURRENT_COMBINED_PROOF_PROTOCOL_VERSION = CAPACITY_GEMM_V2_PROTOCOL_VERSION
SUPPORTED_COMBINED_PROOF_FORMATS = frozenset(
    {COMBINED_PROOF_FORMAT_V1, COMBINED_PROOF_FORMAT_V2}
)

_PROOF_DATA_EXCEPTIONS = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
    struct.error,
)


def _sha256_hex(domain: bytes, *parts: bytes) -> str:
    h = hashlib.sha256()
    h.update(domain)
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _combined_transcript_root(
    *,
    capacity_transcript: str,
    capacity_tail_transcript: str,
    fp64_transcript: str,
    capacity_params: dict[str, Any],
    capacity_tail_params: dict[str, Any] | None,
    fp64_params: dict[str, Any],
    proof_protocol_version: int = LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
) -> str:
    if type(proof_protocol_version) is not int or proof_protocol_version not in {
        LEGACY_COMBINED_PROOF_PROTOCOL_VERSION,
        CURRENT_COMBINED_PROOF_PROTOCOL_VERSION,
    }:
        raise ValueError("unsupported combined proof protocol version")
    version_fields: dict[str, Any] = {}
    transcript_domain = b"VERATHOS_HOT_CAPACITY_COMBINED_TRANSCRIPT_V1"
    if proof_protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
        version_fields["proof_protocol_version"] = proof_protocol_version
        transcript_domain = b"VERATHOS_HOT_CAPACITY_COMBINED_TRANSCRIPT_V2"
    payload = json.dumps(
        {
            "version": COMBINED_WORKLOAD_VERSION,
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


def _sample_seed_from_b_proof(*, b_proof_seed: str, combined_transcript: str) -> str:
    return _sha256_hex(
        b"VERATHOS_HOT_CAPACITY_COMBINED_SAMPLE_V1",
        bytes.fromhex(str(b_proof_seed).removeprefix("0x")),
        bytes.fromhex(str(combined_transcript)),
    )


def _verify_fp64_identity_proof(
    proof: Mapping[str, Any],
    *,
    fp64_params: Mapping[str, Any],
    expected_transcript_root: str,
    expected_challenge_seed: str,
    expected_spot_checks: int | None,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
) -> tuple[bool, str]:
    from verathos.capacity.bench import bench_fp64_identity as fp64_identity  # noqa: PLC0415

    expected_seed = fp64_identity.seed_for(
        lease_id,
        int(gpu_index),
        str(proof_seed_hex or ""),
    )
    exact_fields = {
        "seed": expected_seed,
        "matrix_dim": fp64_params.get("matrix_dim"),
        "block_size": fp64_params.get("block_size"),
        "passes": fp64_params.get("passes"),
        "rounds": fp64_params.get("rounds"),
    }
    for field_name, expected_value in exact_fields.items():
        actual_value = proof.get(field_name)
        if type(expected_value) is not int or type(actual_value) is not int:
            return False, f"{field_name}_does_not_match_scheduled_workload"
        if actual_value != expected_value:
            return False, f"{field_name}_does_not_match_scheduled_workload"
    if proof.get("transcript_root") != expected_transcript_root:
        return False, "transcript_root_parent_mismatch"
    if proof.get("proof_challenge_seed") != expected_challenge_seed:
        return False, "proof_challenge_seed_mismatch"
    if expected_spot_checks is not None:
        spots = proof.get("spots")
        if not isinstance(spots, list) or len(spots) != expected_spot_checks:
            return False, "spot_check_count_does_not_match_scheduled_workload"
    return fp64_identity.verify_fp64_identity_proof(proof)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def combined_proof_protocol_version(proof: Mapping[str, Any] | Any) -> int | None:
    """Return only an unambiguous combined proof protocol version."""

    if not isinstance(proof, Mapping):
        return None
    proof_format = proof.get("format")
    version = proof.get("proof_protocol_version")
    if proof_format == COMBINED_PROOF_FORMAT_V1:
        if "proof_protocol_version" not in proof:
            return LEGACY_COMBINED_PROOF_PROTOCOL_VERSION
        if type(version) is int and version == LEGACY_COMBINED_PROOF_PROTOCOL_VERSION:
            return version
        return None
    if (
        proof_format == COMBINED_PROOF_FORMAT_V2
        and type(version) is int
        and version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
    ):
        return version
    return None


def is_combined_proof_payload(proof: Mapping[str, Any] | Any) -> bool:
    return (
        isinstance(proof, Mapping)
        and proof.get("format") in SUPPORTED_COMBINED_PROOF_FORMATS
    )


def combined_commitment_from_final_timing(
    final_timing: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Normalize the timed child result into the signed final commitment."""

    if not isinstance(final_timing, Mapping):
        return {}
    proof_format = str(final_timing.get("proof_format") or "")
    if proof_format not in SUPPORTED_COMBINED_PROOF_FORMATS:
        return {}
    commitment = {
        "format": proof_format,
        "workload_version": final_timing.get("workload_version"),
        "pass_count": final_timing.get("pass_count"),
        "capacity_transcript_root": final_timing.get("capacity_transcript_root"),
        "capacity_tail_transcript_root": final_timing.get(
            "capacity_tail_transcript_root"
        ),
        "fp64_transcript_root": final_timing.get("fp64_transcript_root"),
        "combined_transcript_root": final_timing.get("combined_transcript_root"),
        "capacity_params": final_timing.get("capacity_params"),
        "capacity_tail_params": final_timing.get("capacity_tail_params"),
        "fp64_params": final_timing.get("fp64_params"),
        "workspace_mode": final_timing.get("workspace_mode"),
        "timed_cuda_component_s": final_timing.get("timed_cuda_component_s"),
        "timed_wall_s": final_timing.get("timed_wall_s"),
    }
    protocol_version = final_timing.get("proof_protocol_version")
    if type(protocol_version) is int:
        commitment["proof_protocol_version"] = protocol_version
    return commitment


def _passes(params: Mapping[str, Any] | None) -> int:
    if not isinstance(params, Mapping):
        return 0
    try:
        return max(0, int(params.get("passes") or 0))
    except Exception:
        return 0


def _combined_params_from_workload_spec(
    workload_spec: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int] | None, dict[str, int], int, int]:
    if not isinstance(workload_spec, Mapping):
        raise ValueError("expected workload spec is not a mapping")
    if workload_spec.get("workload_version") != COMBINED_WORKLOAD_VERSION:
        raise ValueError("expected workload version is not supported")

    def exact_int(name: str, *, minimum: int = 0) -> int:
        value = workload_spec.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"expected workload field {name} is invalid")
        return value

    capacity_params = {
        "matrix_dim": exact_int("capacity_matrix_dim", minimum=2),
        "block_size": exact_int("capacity_block_size", minimum=2),
        "passes": exact_int("capacity_passes", minimum=1),
        "rounds": exact_int("capacity_rounds", minimum=1),
        "transition_mix_rounds": exact_int("transition_mix_rounds"),
        "transition_fanout": exact_int("transition_fanout", minimum=1),
    }
    tail_passes = exact_int("capacity_tail_passes")
    capacity_tail_params = None
    if tail_passes:
        capacity_tail_params = {
            "matrix_dim": capacity_params["matrix_dim"],
            "block_size": capacity_params["block_size"],
            "passes": tail_passes,
            "rounds": exact_int("capacity_tail_rounds", minimum=1),
            "transition_mix_rounds": exact_int(
                "capacity_tail_transition_mix_rounds", minimum=1
            ),
            "transition_fanout": exact_int(
                "capacity_tail_transition_fanout", minimum=1
            ),
        }
    fp64_params = {
        "matrix_dim": exact_int("fp64_matrix_dim", minimum=2),
        "block_size": exact_int("fp64_block_size", minimum=2),
        "passes": exact_int("fp64_passes"),
        "rounds": exact_int("fp64_rounds", minimum=1),
    }
    expected_pass_count = (
        capacity_params["passes"] + tail_passes + fp64_params["passes"]
    )
    declared_pass_count = exact_int("pass_count", minimum=1)
    if declared_pass_count != expected_pass_count:
        raise ValueError("expected workload pass_count is inconsistent")
    spot_checks = exact_int("spot_checks", minimum=1)
    if spot_checks > 4096:
        raise ValueError("expected workload spot_checks is invalid")
    return (
        capacity_params,
        capacity_tail_params,
        fp64_params,
        expected_pass_count,
        spot_checks,
    )


def _verify_capacity_lane_v2(
    *,
    capacity_proof: Mapping[str, Any],
    capacity_params: Mapping[str, Any],
    combined_transcript_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    sample_seed_hex: str,
) -> tuple[bool, str]:
    if "gemm_v2" not in capacity_proof:
        return False, "missing_gemm_v2_payload"
    if "gemm_v2_state_openings" not in capacity_proof:
        return False, "missing_gemm_v2_state_openings"
    state_openings = capacity_proof.get("gemm_v2_state_openings")
    if not isinstance(state_openings, list):
        return False, "malformed_gemm_v2_state_openings"
    block_proof = capacity_proof.get("block_proof")
    if not isinstance(block_proof, Mapping):
        return False, "missing_gemm_v2_block_opening"
    q_block = block_proof.get("q_block_i32")
    q_path = block_proof.get("merkle_path")
    if not isinstance(q_path, Mapping):
        return False, "missing_gemm_v2_q_merkle_path"
    try:
        context = capacity_gemm_v2_context_from_workspace_proof(
            proof=capacity_proof,
            capacity_params=capacity_params,
            lease_id=lease_id,
            gpu_index=gpu_index,
            proof_seed_hex=proof_seed_hex,
            challenge_seed_hex=sample_seed_hex,
            transcript_root_hex=combined_transcript_root,
        )
    except CapacityGemmV2FormatError as exc:
        return False, f"invalid_gemm_v2_context:{exc}"
    ok, reason = verify_capacity_gemm_v2(
        context=context,
        payload=capacity_proof.get("gemm_v2"),
        q_block_i32=q_block,
        q_merkle_path=q_path,
        state_openings=state_openings,
    )
    return (True, "ok") if ok else (False, f"gemm_v2_invalid:{reason}")


def _verify_combined_proof_payload_strict(
    *,
    proof: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    expected_combined_transcript_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    proof_challenge_seed_hex: str,
    expected_workload_spec: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Verify a combined capacity audit proof payload.

    ``proof_challenge_seed_hex`` is the B_proof-derived seed committed by the
    validator from finalized chain data. The sampled lane openings are derived
    from that seed plus the pre-B_proof combined transcript.
    """

    proof = _as_dict(proof)
    protocol_version = combined_proof_protocol_version(proof)
    if protocol_version is None:
        return False, "unsupported_combined_format"
    if str(proof.get("workload_version") or "") != COMBINED_WORKLOAD_VERSION:
        return False, "unsupported_combined_workload"

    combined_root = str(proof.get("combined_transcript_root") or "")
    expected_root = str(expected_combined_transcript_root or "")
    if not combined_root or combined_root != expected_root:
        return False, "combined_transcript_root_mismatch"

    final_commit = _as_dict(final_artifact.get("combined"))
    if (
        protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
        and not final_commit
    ):
        return False, "missing_final_combined_commitment"
    if final_commit:
        if str(final_commit.get("format") or "") != str(proof.get("format") or ""):
            return False, "final_combined_format_mismatch"
        final_version = final_commit.get("proof_protocol_version")
        if protocol_version == LEGACY_COMBINED_PROOF_PROTOCOL_VERSION:
            if final_version not in (None, LEGACY_COMBINED_PROOF_PROTOCOL_VERSION):
                return False, "final_proof_protocol_version_mismatch"
        elif (
            type(final_version) is not int
            or final_version != CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
        ):
            return False, "final_proof_protocol_version_mismatch"
        if str(final_commit.get("combined_transcript_root") or "") != combined_root:
            return False, "final_combined_transcript_root_mismatch"
        for key in (
            "capacity_transcript_root",
            "capacity_tail_transcript_root",
            "fp64_transcript_root",
        ):
            if str(final_commit.get(key) or "") != str(proof.get(key) or ""):
                return False, f"final_{key}_mismatch"
        for key in ("capacity_params", "capacity_tail_params", "fp64_params"):
            final_value = final_commit.get(key)
            proof_value = proof.get(key)
            if final_value != proof_value:
                return False, f"final_{key}_mismatch"

    capacity_params = _as_dict(proof.get("capacity_params"))
    capacity_tail_params_raw = proof.get("capacity_tail_params")
    capacity_tail_params = _as_dict(capacity_tail_params_raw)
    fp64_params = _as_dict(proof.get("fp64_params"))
    expected_spot_checks: int | None = None
    if (
        protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
        and expected_workload_spec is None
    ):
        return False, "missing_validator_owned_workload_spec"
    if expected_workload_spec is not None:
        (
            expected_capacity_params,
            expected_capacity_tail_params,
            expected_fp64_params,
            expected_pass_count,
            expected_spot_checks,
        ) = _combined_params_from_workload_spec(expected_workload_spec)
        if capacity_params != expected_capacity_params:
            return False, "capacity_params_do_not_match_scheduled_workload"
        if capacity_tail_params_raw != expected_capacity_tail_params:
            return False, "capacity_tail_params_do_not_match_scheduled_workload"
        if fp64_params != expected_fp64_params:
            return False, "fp64_params_do_not_match_scheduled_workload"
        if (
            type(proof.get("pass_count")) is not int
            or proof.get("pass_count") != expected_pass_count
        ):
            return False, "pass_count_does_not_match_scheduled_workload"
    recomputed_combined = _combined_transcript_root(
        capacity_transcript=str(proof.get("capacity_transcript_root") or ""),
        capacity_tail_transcript=str(proof.get("capacity_tail_transcript_root") or ""),
        fp64_transcript=str(proof.get("fp64_transcript_root") or ""),
        capacity_params=capacity_params,
        capacity_tail_params=capacity_tail_params_raw
        if isinstance(capacity_tail_params_raw, dict)
        else None,
        fp64_params=fp64_params,
        proof_protocol_version=protocol_version,
    )
    if recomputed_combined != combined_root:
        return False, "recomputed_combined_transcript_mismatch"

    b_proof_seed = str(proof.get("b_proof_seed") or "").removeprefix("0x")
    expected_b_proof_seed = str(proof_challenge_seed_hex or "").removeprefix("0x")
    if b_proof_seed != expected_b_proof_seed:
        return False, "b_proof_seed_mismatch"
    expected_sample_seed = _sample_seed_from_b_proof(
        b_proof_seed=b_proof_seed,
        combined_transcript=combined_root,
    )
    sample_seed = str(proof.get("sample_seed") or "").removeprefix("0x")
    if sample_seed != expected_sample_seed:
        return False, "sample_seed_mismatch"

    capacity_proof = _as_dict(proof.get("capacity_proof"))
    cap_passes = _passes(capacity_params)
    if cap_passes <= 0:
        return False, "invalid_capacity_pass_count"
    ok, reason = verify_workspace_proof_payload(
        proof=capacity_proof,
        expected_transcript_root=str(proof.get("capacity_transcript_root") or ""),
        expected_pass0_root=str(
            (capacity_proof.get("root_chain") or [{}])[0].get("pass_root") or ""
        ),
        expected_final_root=str(
            (capacity_proof.get("root_chain") or [{}])[-1].get("pass_root") or ""
        ),
        lease_id=lease_id,
        gpu_index=int(gpu_index),
        proof_seed_hex=str(proof_seed_hex or ""),
        pass_count=cap_passes,
        proof_challenge_seed_hex=sample_seed,
        expected_spot_checks=(
            expected_spot_checks if expected_workload_spec is not None else None
        ),
        proof_protocol_version=protocol_version,
    )
    if not ok:
        return False, f"capacity_{reason}"
    if protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
        ok, reason = _verify_capacity_lane_v2(
            capacity_proof=capacity_proof,
            capacity_params=capacity_params,
            combined_transcript_root=combined_root,
            lease_id=lease_id,
            gpu_index=int(gpu_index),
            proof_seed_hex=proof_seed_hex,
            sample_seed_hex=sample_seed,
        )
        if not ok:
            return False, f"capacity_{reason}"

    tail_passes = _passes(capacity_tail_params)
    if tail_passes > 0:
        tail_proof = _as_dict(proof.get("capacity_tail_proof"))
        ok, reason = verify_workspace_proof_payload(
            proof=tail_proof,
            expected_transcript_root=str(
                proof.get("capacity_tail_transcript_root") or ""
            ),
            expected_pass0_root=str(
                (tail_proof.get("root_chain") or [{}])[0].get("pass_root") or ""
            ),
            expected_final_root=str(
                (tail_proof.get("root_chain") or [{}])[-1].get("pass_root") or ""
            ),
            lease_id=lease_id,
            gpu_index=int(gpu_index),
            proof_seed_hex=str(proof_seed_hex or ""),
            pass_count=tail_passes,
            proof_challenge_seed_hex=sample_seed,
            expected_spot_checks=(
                expected_spot_checks if expected_workload_spec is not None else None
            ),
            proof_protocol_version=protocol_version,
        )
        if not ok:
            return False, f"capacity_tail_{reason}"
        if protocol_version == CURRENT_COMBINED_PROOF_PROTOCOL_VERSION:
            ok, reason = _verify_capacity_lane_v2(
                capacity_proof=tail_proof,
                capacity_params=capacity_tail_params,
                combined_transcript_root=combined_root,
                lease_id=lease_id,
                gpu_index=int(gpu_index),
                proof_seed_hex=proof_seed_hex,
                sample_seed_hex=sample_seed,
            )
            if not ok:
                return False, f"capacity_tail_{reason}"

    fp64_proof = _as_dict(proof.get("fp64_proof"))
    fp64_passes = _passes(fp64_params)
    if fp64_passes > 0:
        ok, reason = _verify_fp64_identity_proof(
            fp64_proof,
            fp64_params=fp64_params,
            expected_transcript_root=str(proof.get("fp64_transcript_root") or ""),
            expected_challenge_seed=sample_seed,
            expected_spot_checks=expected_spot_checks,
            lease_id=lease_id,
            gpu_index=int(gpu_index),
            proof_seed_hex=proof_seed_hex,
        )
        if not ok:
            return False, f"fp64_{reason}"
    elif fp64_proof:
        return False, "unexpected_fp64_proof"

    return True, "ok"


def verify_combined_proof_payload(
    *,
    proof: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
    expected_combined_transcript_root: str,
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    proof_challenge_seed_hex: str,
    expected_workload_spec: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """Verify v1/v2 combined proofs and classify miner data errors as invalid."""

    try:
        return _verify_combined_proof_payload_strict(
            proof=proof,
            final_artifact=final_artifact,
            expected_combined_transcript_root=expected_combined_transcript_root,
            lease_id=lease_id,
            gpu_index=gpu_index,
            proof_seed_hex=proof_seed_hex,
            proof_challenge_seed_hex=proof_challenge_seed_hex,
            expected_workload_spec=expected_workload_spec,
        )
    except _PROOF_DATA_EXCEPTIONS:
        return False, "malformed_combined_proof"


__all__ = [
    "COMBINED_PROOF_FORMAT",
    "COMBINED_PROOF_FORMAT_V1",
    "COMBINED_PROOF_FORMAT_V2",
    "COMBINED_WORKLOAD_VERSION",
    "CURRENT_COMBINED_PROOF_PROTOCOL_VERSION",
    "LEGACY_COMBINED_PROOF_PROTOCOL_VERSION",
    "SUPPORTED_COMBINED_PROOF_FORMATS",
    "combined_proof_protocol_version",
    "combined_commitment_from_final_timing",
    "is_combined_proof_payload",
    "verify_combined_proof_payload",
]
