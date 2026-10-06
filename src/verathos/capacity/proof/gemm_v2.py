"""Strict arithmetic adapter for protocol-v2 hot-capacity proofs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import struct
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np

from verathos.capacity.proof.workspace import (
    WORKSPACE_MIXED_PROOF_FORMAT,
    WORKSPACE_PROOF_FORMAT,
    WORKSPACE_TRANSITION_PROOF_FORMAT,
    derive_challenge_block_index,
    derive_sampled_round_index,
    workspace_i8_values_after_mix,
    workspace_q_block_hash_i32,
    workspace_seed_for,
    verify_workspace_merkle_path,
    verify_workspace_state_merkle_path,
    workspace_state_block_hash_i8,
)
from verathos.capacity.zk.crypto.gemm_v2_batch import (
    GemmV2BatchProof,
    GemmV2BatchStatement,
    verify_gemm_v2_batch_sumcheck,
)
from verathos.capacity.zk.crypto.gemm_v2_reference import (
    GemmV2FormatError,
    GemmV2Statement,
    GemmV2VerificationError,
    PALLAS_SCALAR_MODULUS,
)
from verathos.capacity.zk.crypto.pcs_v2 import (
    PCSUnavailableError,
    evaluate_i8_matrix,
    prove_gemm_batch_sumcheck_i8,
    scalar_to_int as pcs_scalar_to_int,
)
from verathos.capacity.zk.types import MerklePath


CAPACITY_GEMM_V2_FORMAT = "hot_capacity_gemm_pallas_v2"
CAPACITY_GEMM_V2_PROTOCOL_VERSION = 2
CAPACITY_GEMM_V2_CHALLENGE_COUNT = 1

_CONTEXT_DOMAIN = b"VERATHOS/HOT_CAPACITY/GEMM_V2/CONTEXT/SHA256"
_SOURCE_DOMAIN = b"VERATHOS/HOT_CAPACITY/GEMM_V2/SOURCE/SHA256"
_PROOF_MAGIC = b"HCG2"
_PROOF_HEADER = struct.Struct("<4sHH32sI")
_MAX_ARITHMETIC_PROOF_BYTES = 16 * 1024
_MAX_MATRIX_DIM = 1 << 17
_MAX_TEXT_BYTES = 512
_IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")


class CapacityGemmV2FormatError(ValueError):
    """A protocol-v2 capacity statement or proof is malformed."""


class CapacityGemmV2VerificationError(ValueError):
    """A well-formed protocol-v2 capacity proof did not verify."""


def _record(fields: Sequence[tuple[bytes, bytes]]) -> bytes:
    encoded = bytearray()
    for label, value in fields:
        if not isinstance(label, bytes) or not label:
            raise CapacityGemmV2FormatError("record label must be non-empty bytes")
        if not isinstance(value, bytes):
            raise CapacityGemmV2FormatError("record value must be bytes")
        encoded.extend(struct.pack("<I", len(label)))
        encoded.extend(label)
        encoded.extend(struct.pack("<Q", len(value)))
        encoded.extend(value)
    return bytes(encoded)


def _fixed_bytes(value: bytes, size: int, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != size:
        raise CapacityGemmV2FormatError(f"{field_name} must be exactly {size} bytes")
    return value


def _integer(value: int, field_name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise CapacityGemmV2FormatError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < minimum or integer > maximum:
        raise CapacityGemmV2FormatError(f"{field_name} is outside the allowed range")
    return integer


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapacityGemmV2FormatError(f"{field_name} must be non-empty text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CapacityGemmV2FormatError(f"{field_name} is not valid UTF-8") from exc
    if len(encoded) > _MAX_TEXT_BYTES:
        raise CapacityGemmV2FormatError(f"{field_name} is too long")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _hex32(value: Any, field_name: str) -> bytes:
    if not isinstance(value, str):
        raise CapacityGemmV2FormatError(f"{field_name} must be hex text")
    raw = value.removeprefix("0x")
    if len(raw) != 64:
        raise CapacityGemmV2FormatError(f"{field_name} must be exactly 32 bytes")
    try:
        decoded = bytes.fromhex(raw)
    except ValueError as exc:
        raise CapacityGemmV2FormatError(f"{field_name} is not valid hex") from exc
    if raw.lower() != decoded.hex():
        raise CapacityGemmV2FormatError(f"{field_name} hex is not canonical")
    return decoded


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapacityGemmV2FormatError(f"{field_name} must be a mapping")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        raise CapacityGemmV2FormatError(f"{field_name} must be a sequence")
    return value


@dataclass(frozen=True, slots=True)
class CapacityGemmV2Context:
    """Verifier-owned context for exactly one sampled workspace GEMM block."""

    workload_version: str
    lease_id: str
    gpu_index: int
    proof_seed: bytes
    challenge_seed: bytes
    transcript_root: bytes
    round_root: bytes
    workspace_seed: int
    pass_count: int
    rounds_per_pass: int
    pass_index: int
    round_index: int
    matrix_dim: int
    block_size: int
    row_block: int
    column_block: int
    memory_mix_rounds: int = 0
    transition_mix_rounds: int = 0
    transition_fanout: int = 1
    state_root: bytes = b""
    challenge_count: int = CAPACITY_GEMM_V2_CHALLENGE_COUNT
    challenge_ordinal: int = 0

    def __post_init__(self) -> None:
        workload = _text(self.workload_version, "workload_version")
        if not _IDENTIFIER_RE.fullmatch(workload):
            raise CapacityGemmV2FormatError("workload_version is not canonical")
        lease_id = _text(self.lease_id, "lease_id")
        gpu_index = _integer(
            self.gpu_index, "gpu_index", minimum=0, maximum=(1 << 32) - 1
        )
        proof_seed = _fixed_bytes(self.proof_seed, 32, "proof_seed")
        _fixed_bytes(self.challenge_seed, 32, "challenge_seed")
        _fixed_bytes(self.transcript_root, 32, "transcript_root")
        _fixed_bytes(self.round_root, 32, "round_root")
        workspace_seed = _integer(
            self.workspace_seed,
            "workspace_seed",
            minimum=0,
            maximum=(1 << 64) - 1,
        )
        pass_count = _integer(
            self.pass_count, "pass_count", minimum=1, maximum=(1 << 32) - 1
        )
        rounds = _integer(
            self.rounds_per_pass,
            "rounds_per_pass",
            minimum=1,
            maximum=(1 << 32) - 1,
        )
        pass_index = _integer(
            self.pass_index, "pass_index", minimum=0, maximum=pass_count - 1
        )
        round_index = _integer(
            self.round_index, "round_index", minimum=0, maximum=rounds - 1
        )
        matrix_dim = _integer(
            self.matrix_dim, "matrix_dim", minimum=2, maximum=_MAX_MATRIX_DIM
        )
        block_size = _integer(
            self.block_size, "block_size", minimum=2, maximum=matrix_dim
        )
        if block_size & (block_size - 1):
            raise CapacityGemmV2FormatError("block_size must be a power of two")
        if matrix_dim % block_size:
            raise CapacityGemmV2FormatError(
                "matrix_dim must be divisible by block_size"
            )
        blocks_per_row = matrix_dim // block_size
        row_block = _integer(
            self.row_block, "row_block", minimum=0, maximum=blocks_per_row - 1
        )
        column_block = _integer(
            self.column_block,
            "column_block",
            minimum=0,
            maximum=blocks_per_row - 1,
        )
        memory_mix_rounds = _integer(
            self.memory_mix_rounds,
            "memory_mix_rounds",
            minimum=0,
            maximum=(1 << 16) - 1,
        )
        transition_mix_rounds = _integer(
            self.transition_mix_rounds,
            "transition_mix_rounds",
            minimum=0,
            maximum=(1 << 16) - 1,
        )
        transition_fanout = _integer(
            self.transition_fanout,
            "transition_fanout",
            minimum=1,
            maximum=(1 << 16) - 1,
        )
        if memory_mix_rounds and transition_mix_rounds:
            raise CapacityGemmV2FormatError(
                "workspace mix modes are mutually exclusive"
            )
        if transition_mix_rounds:
            _fixed_bytes(self.state_root, 32, "state_root")
        elif self.state_root:
            raise CapacityGemmV2FormatError(
                "state_root is only valid for transition workspaces"
            )
        if not transition_mix_rounds and transition_fanout != 1:
            raise CapacityGemmV2FormatError(
                "transition_fanout must be one without transition mixing"
            )
        if self.challenge_count != CAPACITY_GEMM_V2_CHALLENGE_COUNT:
            raise CapacityGemmV2FormatError("challenge_count is not the fixed v2 count")
        if self.challenge_ordinal != 0:
            raise CapacityGemmV2FormatError("challenge_ordinal is not canonical")

        expected_seed = workspace_seed_for(lease_id, gpu_index, proof_seed.hex())
        if workspace_seed != expected_seed:
            raise CapacityGemmV2FormatError("workspace_seed does not match the context")

        # These are verifier-derived indices. Keeping the checks in the context
        # constructor makes an incorrectly assembled statement fail closed.
        from verathos.capacity.core import derive_sampled_pass_index

        challenge_hex = self.challenge_seed.hex()
        expected_pass = derive_sampled_pass_index(
            challenge_hex,
            lease_id,
            gpu_index,
            pass_count,
        )
        if pass_index != expected_pass:
            raise CapacityGemmV2FormatError("pass_index is not the derived challenge")
        expected_round = derive_sampled_round_index(
            challenge_hex,
            lease_id,
            gpu_index,
            pass_index,
            rounds,
        )
        if round_index != expected_round:
            raise CapacityGemmV2FormatError("round_index is not the derived challenge")
        expected_block = derive_challenge_block_index(
            round_root_hex=self.round_root.hex(),
            lease_id=lease_id,
            gpu_index=gpu_index,
            pass_index=pass_index,
            round_index=round_index,
            blocks_per_row=blocks_per_row,
            challenge_seed_hex=challenge_hex,
        )
        if row_block * blocks_per_row + column_block != expected_block:
            raise CapacityGemmV2FormatError(
                "block coordinates are not the derived challenge"
            )

    @property
    def padded_inner(self) -> int:
        return _next_power_of_two(self.matrix_dim)

    @property
    def arithmetic_rounds(self) -> int:
        return self.padded_inner.bit_length() - 1

    @property
    def block_index(self) -> int:
        return self.row_block * (self.matrix_dim // self.block_size) + self.column_block

    def canonical_bytes(self) -> bytes:
        state_root = self.state_root if self.state_root else b"\x00" * 32
        return _record(
            (
                (
                    b"protocol_version",
                    struct.pack("<H", CAPACITY_GEMM_V2_PROTOCOL_VERSION),
                ),
                (b"format", CAPACITY_GEMM_V2_FORMAT.encode()),
                (b"workload_version", self.workload_version.encode()),
                (b"lease_id", self.lease_id.encode()),
                (b"gpu_index", struct.pack("<I", self.gpu_index)),
                (b"proof_seed", self.proof_seed),
                (b"challenge_seed", self.challenge_seed),
                (b"transcript_root", self.transcript_root),
                (b"round_root", self.round_root),
                (b"workspace_seed", struct.pack("<Q", self.workspace_seed)),
                (b"pass_count", struct.pack("<I", self.pass_count)),
                (b"rounds_per_pass", struct.pack("<I", self.rounds_per_pass)),
                (b"pass_index", struct.pack("<I", self.pass_index)),
                (b"round_index", struct.pack("<I", self.round_index)),
                (b"matrix_dim", struct.pack("<I", self.matrix_dim)),
                (b"block_size", struct.pack("<I", self.block_size)),
                (b"row_block", struct.pack("<I", self.row_block)),
                (b"column_block", struct.pack("<I", self.column_block)),
                (b"memory_mix_rounds", struct.pack("<I", self.memory_mix_rounds)),
                (
                    b"transition_mix_rounds",
                    struct.pack("<I", self.transition_mix_rounds),
                ),
                (b"transition_fanout", struct.pack("<I", self.transition_fanout)),
                (b"state_root", state_root),
                (b"challenge_count", struct.pack("<I", self.challenge_count)),
                (b"challenge_ordinal", struct.pack("<I", self.challenge_ordinal)),
            )
        )

    def digest(self) -> bytes:
        return hashlib.sha256(_CONTEXT_DOMAIN + self.canonical_bytes()).digest()

    def source_binding(self, matrix_id: int) -> bytes:
        matrix_id = _integer(matrix_id, "matrix_id", minimum=0, maximum=1)
        mode = b"state_merkle" if self.transition_mix_rounds else b"deterministic"
        return hashlib.sha256(
            _SOURCE_DOMAIN + self.digest() + struct.pack("<B", matrix_id) + mode
        ).digest()

    def statement(self, y_leaf_hash: bytes) -> GemmV2Statement:
        y_leaf_hash = _fixed_bytes(y_leaf_hash, 32, "y_leaf_hash")
        operation_identity = _record(
            (
                (b"role", b"hot_capacity"),
                (b"workload_version", self.workload_version.encode()),
                (b"context_digest", self.digest()),
            )
        )
        return GemmV2Statement(
            outer_digest=self.digest(),
            operation_identity=operation_identity,
            row_block=self.row_block,
            column_block=self.column_block,
            rows=self.block_size,
            inner=self.padded_inner,
            valid_inner=self.matrix_dim,
            columns=self.block_size,
            x_commitment=self.source_binding(0),
            w_commitment=self.source_binding(1),
            y_commitment=y_leaf_hash,
        )


def capacity_gemm_v2_context_from_workspace_proof(
    *,
    proof: Mapping[str, Any],
    capacity_params: Mapping[str, Any],
    lease_id: str,
    gpu_index: int,
    proof_seed_hex: str,
    challenge_seed_hex: str,
    transcript_root_hex: str,
) -> CapacityGemmV2Context:
    """Reconstruct the verifier-owned v2 statement from one workspace lane.

    The combined envelope commits ``capacity_params`` before the post-timing
    challenge exists.  The duplicated lane fields must match those parameters
    exactly; no dimensions, counts, coordinates, or mix controls are accepted
    from the prover independently.
    """

    proof = _mapping(proof, "workspace proof")
    capacity_params = _mapping(capacity_params, "capacity_params")
    expected_param_fields = {
        "matrix_dim",
        "block_size",
        "passes",
        "rounds",
        "transition_mix_rounds",
        "transition_fanout",
    }
    if set(capacity_params) != expected_param_fields:
        raise CapacityGemmV2FormatError("capacity_params fields are not canonical")

    matrix_dim = _integer(
        capacity_params.get("matrix_dim"),
        "capacity matrix_dim",
        minimum=2,
        maximum=_MAX_MATRIX_DIM,
    )
    block_size = _integer(
        capacity_params.get("block_size"),
        "capacity block_size",
        minimum=2,
        maximum=matrix_dim,
    )
    pass_count = _integer(
        capacity_params.get("passes"),
        "capacity passes",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    rounds = _integer(
        capacity_params.get("rounds"),
        "capacity rounds",
        minimum=1,
        maximum=(1 << 32) - 1,
    )
    transition_mix_rounds = _integer(
        capacity_params.get("transition_mix_rounds"),
        "capacity transition_mix_rounds",
        minimum=0,
        maximum=(1 << 16) - 1,
    )
    transition_fanout = _integer(
        capacity_params.get("transition_fanout"),
        "capacity transition_fanout",
        minimum=1,
        maximum=(1 << 16) - 1,
    )

    duplicated_fields = {
        "matrix_dim": matrix_dim,
        "block_size": block_size,
        "passes": pass_count,
        "rounds": rounds,
        "transition_mix_rounds": transition_mix_rounds,
        "transition_fanout": transition_fanout,
    }
    for field_name, expected in duplicated_fields.items():
        actual = _integer(
            proof.get(field_name),
            f"workspace {field_name}",
            minimum=0,
            maximum=(1 << 32) - 1,
        )
        if actual != expected:
            raise CapacityGemmV2FormatError(
                f"workspace {field_name} does not match capacity_params"
            )

    proof_format = proof.get("format")
    memory_mix_rounds = _integer(
        proof.get("memory_mix_rounds"),
        "workspace memory_mix_rounds",
        minimum=0,
        maximum=(1 << 16) - 1,
    )
    if proof_format == WORKSPACE_PROOF_FORMAT:
        expected_workload = "fixed_workspace_merkle_gemm_v1"
        if memory_mix_rounds or transition_mix_rounds or transition_fanout != 1:
            raise CapacityGemmV2FormatError(
                "deterministic workspace mode is inconsistent"
            )
    elif proof_format == WORKSPACE_MIXED_PROOF_FORMAT:
        expected_workload = "fixed_workspace_mixed_v2"
        if not memory_mix_rounds or transition_mix_rounds or transition_fanout != 1:
            raise CapacityGemmV2FormatError("mixed workspace mode is inconsistent")
    elif proof_format == WORKSPACE_TRANSITION_PROOF_FORMAT:
        expected_workload = (
            "fixed_workspace_transition_fanout_v3"
            if transition_fanout > 1
            else "fixed_workspace_transition_mix_v3"
        )
        if memory_mix_rounds or not transition_mix_rounds:
            raise CapacityGemmV2FormatError("transition workspace mode is inconsistent")
    else:
        raise CapacityGemmV2FormatError("workspace proof format is not supported by v2")
    if proof.get("workload_version") != expected_workload:
        raise CapacityGemmV2FormatError(
            "workspace workload_version does not match its format"
        )

    workspace_seed = _integer(
        proof.get("workspace_seed"),
        "workspace_seed",
        minimum=0,
        maximum=(1 << 64) - 1,
    )
    sampled = _mapping(proof.get("sampled"), "sampled workspace challenge")
    pass_index = _integer(
        sampled.get("pass_index"),
        "sampled pass_index",
        minimum=0,
        maximum=pass_count - 1,
    )
    round_index = _integer(
        sampled.get("round_index"),
        "sampled round_index",
        minimum=0,
        maximum=rounds - 1,
    )
    block_index = _integer(
        sampled.get("block_index"),
        "sampled block_index",
        minimum=0,
        maximum=(matrix_dim // block_size) ** 2 - 1,
    )

    root_chain = _sequence(proof.get("root_chain"), "workspace root_chain")
    if len(root_chain) != pass_count:
        raise CapacityGemmV2FormatError("workspace root_chain length does not match")
    pass_entry = _mapping(root_chain[pass_index], "sampled pass root entry")
    round_roots = _sequence(pass_entry.get("round_roots"), "sampled pass round roots")
    if len(round_roots) != rounds:
        raise CapacityGemmV2FormatError("round root count does not match")
    round_root = _hex32(round_roots[round_index], "sampled round_root")
    if sampled.get("round_root") != round_root.hex():
        raise CapacityGemmV2FormatError("sampled round_root does not match root_chain")

    block_proof = _mapping(proof.get("block_proof"), "workspace block_proof")
    blocks_per_row = matrix_dim // block_size
    row_block = _integer(
        block_proof.get("bi"),
        "workspace row block",
        minimum=0,
        maximum=blocks_per_row - 1,
    )
    column_block = _integer(
        block_proof.get("bj"),
        "workspace column block",
        minimum=0,
        maximum=blocks_per_row - 1,
    )
    if row_block * blocks_per_row + column_block != block_index:
        raise CapacityGemmV2FormatError("workspace block coordinates do not match")

    state_root = b""
    if proof_format == WORKSPACE_TRANSITION_PROOF_FORMAT:
        state_root_chain = _sequence(
            proof.get("state_root_chain"), "workspace state_root_chain"
        )
        if len(state_root_chain) != pass_count:
            raise CapacityGemmV2FormatError("state_root_chain length does not match")
        state_pass = _mapping(state_root_chain[pass_index], "sampled state pass entry")
        state_rounds = _sequence(
            state_pass.get("round_state_roots"), "sampled state round roots"
        )
        if len(state_rounds) != rounds:
            raise CapacityGemmV2FormatError("state round root count does not match")
        state_mix_roots = _sequence(
            state_rounds[round_index], "sampled state mix roots"
        )
        if len(state_mix_roots) != transition_mix_rounds + 1:
            raise CapacityGemmV2FormatError("state mix root count does not match")
        state_root = _hex32(state_mix_roots[-1], "sampled final state_root")

    return CapacityGemmV2Context(
        workload_version=expected_workload,
        lease_id=_text(lease_id, "lease_id"),
        gpu_index=_integer(gpu_index, "gpu_index", minimum=0, maximum=(1 << 32) - 1),
        proof_seed=_hex32(proof_seed_hex, "proof_seed"),
        challenge_seed=_hex32(challenge_seed_hex, "challenge_seed"),
        transcript_root=_hex32(transcript_root_hex, "transcript_root"),
        round_root=round_root,
        workspace_seed=workspace_seed,
        pass_count=pass_count,
        rounds_per_pass=rounds,
        pass_index=pass_index,
        round_index=round_index,
        matrix_dim=matrix_dim,
        block_size=block_size,
        row_block=row_block,
        column_block=column_block,
        memory_mix_rounds=memory_mix_rounds,
        transition_mix_rounds=transition_mix_rounds,
        transition_fanout=transition_fanout,
        state_root=state_root,
    )


@dataclass(frozen=True, slots=True)
class CapacityGemmV2Payload:
    """Canonical v2 arithmetic payload with no prover-controlled dimensions."""

    statement_digest: bytes
    arithmetic_proof: bytes

    def __post_init__(self) -> None:
        _fixed_bytes(self.statement_digest, 32, "statement_digest")
        if not isinstance(self.arithmetic_proof, bytes) or not self.arithmetic_proof:
            raise CapacityGemmV2FormatError("arithmetic_proof must be non-empty bytes")
        if len(self.arithmetic_proof) > _MAX_ARITHMETIC_PROOF_BYTES:
            raise CapacityGemmV2FormatError("arithmetic_proof is too large")

    def canonical_bytes(self) -> bytes:
        return (
            _PROOF_HEADER.pack(
                _PROOF_MAGIC,
                CAPACITY_GEMM_V2_PROTOCOL_VERSION,
                0,
                self.statement_digest,
                len(self.arithmetic_proof),
            )
            + self.arithmetic_proof
        )

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> "CapacityGemmV2Payload":
        if not isinstance(encoded, bytes):
            raise CapacityGemmV2FormatError("capacity proof encoding must be bytes")
        if len(encoded) < _PROOF_HEADER.size:
            raise CapacityGemmV2FormatError("capacity proof encoding is truncated")
        (
            magic,
            version,
            flags,
            statement_digest,
            proof_length,
        ) = _PROOF_HEADER.unpack_from(encoded)
        if magic != _PROOF_MAGIC:
            raise CapacityGemmV2FormatError("capacity proof magic is not recognized")
        if version != CAPACITY_GEMM_V2_PROTOCOL_VERSION:
            raise CapacityGemmV2FormatError("capacity proof version is not supported")
        if flags != 0:
            raise CapacityGemmV2FormatError("capacity proof flags are not canonical")
        if proof_length == 0 or proof_length > _MAX_ARITHMETIC_PROOF_BYTES:
            raise CapacityGemmV2FormatError(
                "capacity arithmetic proof length is invalid"
            )
        if len(encoded) != _PROOF_HEADER.size + proof_length:
            raise CapacityGemmV2FormatError(
                "capacity proof encoding has an invalid length"
            )
        return cls(statement_digest, encoded[_PROOF_HEADER.size :])

    def to_mapping(self) -> dict[str, Any]:
        return {
            "format": CAPACITY_GEMM_V2_FORMAT,
            "protocol_version": CAPACITY_GEMM_V2_PROTOCOL_VERSION,
            "payload": self.canonical_bytes().hex(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapacityGemmV2Payload":
        if not isinstance(value, Mapping):
            raise CapacityGemmV2FormatError("capacity proof must be a mapping")
        if set(value) != {"format", "protocol_version", "payload"}:
            raise CapacityGemmV2FormatError("capacity proof fields are not canonical")
        if value.get("format") != CAPACITY_GEMM_V2_FORMAT:
            raise CapacityGemmV2FormatError("capacity proof format is not supported")
        version = value.get("protocol_version")
        if isinstance(version, bool) or version != CAPACITY_GEMM_V2_PROTOCOL_VERSION:
            raise CapacityGemmV2FormatError(
                "capacity proof protocol version is not supported"
            )
        payload_hex = value.get("payload")
        if not isinstance(payload_hex, str) or not payload_hex:
            raise CapacityGemmV2FormatError("capacity proof payload must be hex text")
        if payload_hex.startswith("0x") or len(payload_hex) % 2:
            raise CapacityGemmV2FormatError(
                "capacity proof payload hex is not canonical"
            )
        try:
            encoded = bytes.fromhex(payload_hex)
        except ValueError as exc:
            raise CapacityGemmV2FormatError(
                "capacity proof payload is not valid hex"
            ) from exc
        if payload_hex != encoded.hex():
            raise CapacityGemmV2FormatError(
                "capacity proof payload hex is not canonical"
            )
        return cls.from_canonical_bytes(encoded)


def _strict_i32_block(
    q_block_i32: Sequence[Sequence[int]], block_size: int
) -> tuple[tuple[int, ...], ...]:
    if isinstance(q_block_i32, (bytes, bytearray, str)):
        raise CapacityGemmV2FormatError("Q block must be a matrix")
    try:
        rows = tuple(tuple(row) for row in q_block_i32)
    except TypeError as exc:
        raise CapacityGemmV2FormatError("Q block must be a matrix") from exc
    if len(rows) != block_size or any(len(row) != block_size for row in rows):
        raise CapacityGemmV2FormatError("Q block shape does not match block_size")
    normalized = []
    for row in rows:
        normalized_row = []
        for value in row:
            normalized_row.append(
                _integer(
                    value, "Q block value", minimum=-(1 << 31), maximum=(1 << 31) - 1
                )
            )
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _tree_height(num_leaves: int) -> int:
    height = 0
    size = int(num_leaves)
    while size > 1:
        size = (size + 1) // 2
        height += 1
    return height


def _strict_merkle_path(
    path_data: Mapping[str, Any], *, expected_leaf: int, num_leaves: int
) -> MerklePath:
    if not isinstance(path_data, Mapping) or set(path_data) != {
        "leaf_index",
        "siblings",
    }:
        raise CapacityGemmV2FormatError("Merkle path fields are not canonical")
    leaf_index = path_data.get("leaf_index")
    if isinstance(leaf_index, bool) or leaf_index != expected_leaf:
        raise CapacityGemmV2FormatError("Merkle path leaf index does not match")
    siblings_data = path_data.get("siblings")
    if not isinstance(siblings_data, list) or len(siblings_data) != _tree_height(
        num_leaves
    ):
        raise CapacityGemmV2FormatError("Merkle path length does not match the tree")
    siblings = []
    node_index = expected_leaf
    level_size = num_leaves
    for item in siblings_data:
        if not isinstance(item, list) or len(item) != 2:
            raise CapacityGemmV2FormatError("Merkle sibling entry is malformed")
        digest_hex, is_left = item
        if not isinstance(digest_hex, str) or len(digest_hex) != 64:
            raise CapacityGemmV2FormatError("Merkle sibling digest is malformed")
        if not isinstance(is_left, bool):
            raise CapacityGemmV2FormatError("Merkle sibling direction is malformed")
        try:
            digest = bytes.fromhex(digest_hex)
        except ValueError as exc:
            raise CapacityGemmV2FormatError(
                "Merkle sibling digest is malformed"
            ) from exc
        expected_left = bool(node_index & 1)
        if is_left != expected_left:
            raise CapacityGemmV2FormatError("Merkle sibling direction does not match")
        siblings.append((digest, is_left))
        node_index //= 2
        level_size = (level_size + 1) // 2
    return MerklePath(expected_leaf, siblings)


def _verify_q_opening(
    context: CapacityGemmV2Context,
    q_block_i32: Sequence[Sequence[int]],
    q_merkle_path: Mapping[str, Any],
) -> tuple[tuple[tuple[int, ...], ...], bytes]:
    q_block = _strict_i32_block(q_block_i32, context.block_size)
    blocks_per_row = context.matrix_dim // context.block_size
    path = _strict_merkle_path(
        q_merkle_path,
        expected_leaf=context.block_index,
        num_leaves=blocks_per_row * blocks_per_row,
    )
    leaf_hash = workspace_q_block_hash_i32(q_block)
    if not verify_workspace_merkle_path(context.round_root, leaf_hash, path):
        raise CapacityGemmV2VerificationError("Q block Merkle path did not verify")
    return q_block, leaf_hash


def _packed_state_block_i8(
    value: Any,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, str) or not value.startswith("base64:"):
        raise CapacityGemmV2FormatError(
            "packed state block must use canonical base64"
        )
    encoded = value.removeprefix("base64:")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CapacityGemmV2FormatError("packed state block is invalid") from exc
    if base64.b64encode(raw).decode("ascii") != encoded:
        raise CapacityGemmV2FormatError("packed state block is noncanonical")
    expected_length = 2 * block_size * block_size
    if len(raw) != expected_length:
        raise CapacityGemmV2FormatError("packed state block has the wrong length")
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(block_size * 2, block_size)
    a_block = np.ascontiguousarray(packed[0::2, :]).view(np.int8)
    b_block = np.ascontiguousarray(packed[1::2, :]).view(np.int8)
    return a_block, b_block


def required_transition_state_blocks(context: CapacityGemmV2Context) -> tuple[int, ...]:
    """Return the exact state-block set needed for the selected A and B bands."""

    if not isinstance(context, CapacityGemmV2Context):
        raise CapacityGemmV2FormatError("context has an unexpected type")
    if not context.transition_mix_rounds:
        return ()
    blocks_per_row = context.matrix_dim // context.block_size
    row_blocks = {
        context.row_block * blocks_per_row + column for column in range(blocks_per_row)
    }
    column_blocks = {
        row * blocks_per_row + context.column_block for row in range(blocks_per_row)
    }
    return tuple(sorted(row_blocks | column_blocks))


def _transition_bands(
    context: CapacityGemmV2Context,
    state_openings: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    expected_indices = required_transition_state_blocks(context)
    if not isinstance(state_openings, Sequence) or isinstance(
        state_openings, (bytes, bytearray, str)
    ):
        raise CapacityGemmV2FormatError("state openings must be a sequence")
    openings = tuple(state_openings)
    if len(openings) != len(expected_indices):
        raise CapacityGemmV2VerificationError(
            "state openings do not match the exact required block set"
        )
    blocks_per_row = context.matrix_dim // context.block_size
    total_blocks = blocks_per_row * blocks_per_row
    opened: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for opening, expected_index in zip(openings, expected_indices):
        if not isinstance(opening, Mapping) or set(opening) != {
            "block_index",
            "state_block_i8",
            "merkle_path",
        }:
            raise CapacityGemmV2FormatError("state opening fields are not canonical")
        block_index = opening.get("block_index")
        if isinstance(block_index, bool) or not isinstance(block_index, Integral):
            raise CapacityGemmV2FormatError("state block index must be an integer")
        block_index = int(block_index)
        if block_index != expected_index:
            raise CapacityGemmV2VerificationError(
                "state openings do not match the exact required block set"
            )
        a_block, b_block = _packed_state_block_i8(
            opening.get("state_block_i8"),
            context.block_size,
        )
        path = _strict_merkle_path(
            opening.get("merkle_path"),
            expected_leaf=block_index,
            num_leaves=total_blocks,
        )
        leaf_hash = workspace_state_block_hash_i8(a_block, b_block)
        if not verify_workspace_state_merkle_path(
            context.state_root,
            leaf_hash,
            path,
        ):
            raise CapacityGemmV2VerificationError(
                "state Merkle path did not verify"
            )
        opened[block_index] = (a_block, b_block)

    block_size = context.block_size
    x_band = np.empty((block_size, context.matrix_dim), dtype=np.int8)
    for block_column in range(blocks_per_row):
        block_index = context.row_block * blocks_per_row + block_column
        column_start = block_column * block_size
        x_band[:, column_start : column_start + block_size] = opened[block_index][0]

    w_band = np.empty((context.matrix_dim, block_size), dtype=np.int8)
    for block_row in range(blocks_per_row):
        block_index = block_row * blocks_per_row + context.column_block
        row_start = block_row * block_size
        w_band[row_start : row_start + block_size, :] = opened[block_index][1]
    return x_band, w_band


def _deterministic_bands(
    context: CapacityGemmV2Context,
) -> tuple[np.ndarray, np.ndarray]:
    row_start = context.row_block * context.block_size
    column_start = context.column_block * context.block_size
    local_rows = row_start + np.arange(context.block_size, dtype=np.uint64)
    inner_columns = np.arange(context.matrix_dim, dtype=np.uint64)
    x_indices = (
        local_rows[:, np.newaxis] * np.uint64(context.matrix_dim)
        + inner_columns[np.newaxis, :]
    )
    x_band = workspace_i8_values_after_mix(
        workspace_seed=context.workspace_seed,
        pass_index=context.pass_index,
        round_index=context.round_index,
        matrix_id=0,
        element_indices=x_indices,
        memory_mix_rounds=context.memory_mix_rounds,
    )

    inner_rows = np.arange(context.matrix_dim, dtype=np.uint64)
    local_columns = column_start + np.arange(context.block_size, dtype=np.uint64)
    w_indices = (
        inner_rows[:, np.newaxis] * np.uint64(context.matrix_dim)
        + local_columns[np.newaxis, :]
    )
    w_band = workspace_i8_values_after_mix(
        workspace_seed=context.workspace_seed,
        pass_index=context.pass_index,
        round_index=context.round_index,
        matrix_id=1,
        element_indices=w_indices,
        memory_mix_rounds=context.memory_mix_rounds,
    )
    return x_band, w_band


def _workspace_bands(
    context: CapacityGemmV2Context,
    state_openings: Sequence[Mapping[str, Any]],
) -> tuple[list[list[int]], list[list[int]]]:
    if context.transition_mix_rounds:
        return _transition_bands(context, state_openings)
    if state_openings:
        raise CapacityGemmV2FormatError(
            "state openings are not valid for a deterministic workspace"
        )
    return _deterministic_bands(context)


def _pad_inner_bands(
    context: CapacityGemmV2Context,
    x_band: Sequence[Sequence[int]],
    w_band: Sequence[Sequence[int]],
) -> tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]:
    padding = context.padded_inner - context.matrix_dim
    if isinstance(x_band, np.ndarray) and isinstance(w_band, np.ndarray):
        return (
            np.pad(x_band, ((0, 0), (0, padding))),
            np.pad(w_band, ((0, padding), (0, 0))),
        )
    x_padded = [list(row) + [0] * padding for row in x_band]
    w_padded = [list(row) for row in w_band]
    w_padded.extend([[0] * context.block_size for _ in range(padding)])
    return x_padded, w_padded


def _mle_evaluate(values: Sequence[int], point: Sequence[int]) -> int:
    table = [int(value) % PALLAS_SCALAR_MODULUS for value in values]
    if len(table) != 1 << len(point):
        raise CapacityGemmV2FormatError("MLE table size does not match its point")
    modulus = PALLAS_SCALAR_MODULUS
    for coordinate in point:
        coordinate = _integer(
            coordinate,
            "MLE coordinate",
            minimum=0,
            maximum=modulus - 1,
        )
        half = len(table) // 2
        table = [
            (table[index] + coordinate * (table[index + half] - table[index])) % modulus
            for index in range(half)
        ]
    return table[0]


def _matrix_mle_evaluate(matrix: Sequence[Sequence[int]], point: Sequence[int]) -> int:
    return _mle_evaluate([value for row in matrix for value in row], point)


def _matrix_i8_mle_evaluate_native(
    matrix: Sequence[Sequence[int]], point: Sequence[int]
) -> int:
    """Evaluate an int8 matrix with one bounded native matrix fold."""

    values = np.asarray(matrix)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise CapacityGemmV2FormatError("MLE matrix must be non-empty and rectangular")
    rows, columns = (int(values.shape[0]), int(values.shape[1]))
    if rows & (rows - 1) or columns & (columns - 1):
        raise CapacityGemmV2FormatError("MLE matrix dimensions must be powers of two")
    row_bits = rows.bit_length() - 1
    column_bits = columns.bit_length() - 1
    if len(point) != row_bits + column_bits:
        raise CapacityGemmV2FormatError("MLE table size does not match its point")

    normalized_point = tuple(
        _integer(
            coordinate,
            "MLE coordinate",
            minimum=0,
            maximum=PALLAS_SCALAR_MODULUS - 1,
        )
        for coordinate in point
    )
    row_point = normalized_point[:row_bits]
    column_point = normalized_point[row_bits:]
    if rows > columns:
        values = values.T
        row_point, column_point = column_point, row_point

    if np.any(values < -128) or np.any(values > 127):
        raise CapacityGemmV2FormatError("MLE int8 matrix value is outside range")
    values = np.ascontiguousarray(values, dtype=np.int8)
    return pcs_scalar_to_int(
        evaluate_i8_matrix(
            values.tobytes(),
            rows=int(values.shape[0]),
            columns=int(values.shape[1]),
            point=row_point + column_point,
        )
    )


def prove_capacity_gemm_v2(
    *,
    context: CapacityGemmV2Context,
    q_block_i32: Sequence[Sequence[int]],
    q_merkle_path: Mapping[str, Any],
    state_openings: Sequence[Mapping[str, Any]] = (),
) -> CapacityGemmV2Payload:
    """Build one canonical arithmetic proof from authenticated workspace data."""

    if not isinstance(context, CapacityGemmV2Context):
        raise CapacityGemmV2FormatError("context has an unexpected type")
    q_block, leaf_hash = _verify_q_opening(context, q_block_i32, q_merkle_path)
    x_band, w_band = _workspace_bands(context, state_openings)
    x_padded, w_padded = _pad_inner_bands(context, x_band, w_band)
    statement = context.statement(leaf_hash)
    batch_statement = GemmV2BatchStatement((statement,))
    x_values = np.ascontiguousarray(x_padded, dtype=np.int8)
    w_values = np.ascontiguousarray(w_padded, dtype=np.int8)
    y_values = np.ascontiguousarray(q_block, dtype="<i8")
    proof = prove_gemm_batch_sumcheck_i8(
        batch_digest=batch_statement.digest(),
        x_values=x_values.tobytes(),
        w_values=w_values.tobytes(),
        y_values_i64_le=y_values.tobytes(),
        valid_inner=(context.matrix_dim,),
        block_count=1,
        rows=context.block_size,
        inner=context.padded_inner,
        columns=context.block_size,
    )
    return CapacityGemmV2Payload(batch_statement.digest(), proof)


def verify_capacity_gemm_v2_strict(
    *,
    context: CapacityGemmV2Context,
    payload: CapacityGemmV2Payload,
    q_block_i32: Sequence[Sequence[int]],
    q_merkle_path: Mapping[str, Any],
    state_openings: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Verify one v2 proof, raising on every malformed or invalid input."""

    if not isinstance(context, CapacityGemmV2Context):
        raise CapacityGemmV2FormatError("context has an unexpected type")
    if not isinstance(payload, CapacityGemmV2Payload):
        raise CapacityGemmV2FormatError("payload has an unexpected type")
    q_block, leaf_hash = _verify_q_opening(context, q_block_i32, q_merkle_path)
    statement = context.statement(leaf_hash)
    batch_statement = GemmV2BatchStatement((statement,))
    if payload.statement_digest != batch_statement.digest():
        raise CapacityGemmV2VerificationError("statement digest does not match")
    try:
        arithmetic_proof = GemmV2BatchProof.from_canonical_bytes(
            payload.arithmetic_proof,
            expected_blocks=1,
            expected_rounds=context.arithmetic_rounds,
        )
        claims = verify_gemm_v2_batch_sumcheck(batch_statement, arithmetic_proof)
    except GemmV2FormatError as exc:
        raise CapacityGemmV2FormatError(str(exc)) from exc
    except GemmV2VerificationError as exc:
        raise CapacityGemmV2VerificationError(str(exc)) from exc

    x_band, w_band = _workspace_bands(context, state_openings)
    x_padded, w_padded = _pad_inner_bands(context, x_band, w_band)
    x_point = claims.row_challenges + claims.inner_challenges
    w_point = claims.inner_challenges + claims.column_challenges
    try:
        x_evaluation = _matrix_i8_mle_evaluate_native(x_padded, x_point)
        w_evaluation = _matrix_i8_mle_evaluate_native(w_padded, w_point)
    except PCSUnavailableError:
        x_evaluation = _matrix_mle_evaluate(x_padded, x_point)
        w_evaluation = _matrix_mle_evaluate(w_padded, w_point)
    if x_evaluation != claims.x_values[0]:
        raise CapacityGemmV2VerificationError("X terminal evaluation does not match")
    if w_evaluation != claims.w_values[0]:
        raise CapacityGemmV2VerificationError("W terminal evaluation does not match")
    if _matrix_mle_evaluate(q_block, claims.y_point) != claims.y_values[0]:
        raise CapacityGemmV2VerificationError("Y evaluation does not match")


def verify_capacity_gemm_v2(
    *,
    context: CapacityGemmV2Context,
    payload: CapacityGemmV2Payload | Mapping[str, Any],
    q_block_i32: Sequence[Sequence[int]],
    q_merkle_path: Mapping[str, Any],
    state_openings: Sequence[Mapping[str, Any]] = (),
) -> tuple[bool, str]:
    """Fail-closed wrapper suitable for the capacity verification worker."""

    try:
        decoded = (
            payload
            if isinstance(payload, CapacityGemmV2Payload)
            else CapacityGemmV2Payload.from_mapping(payload)
        )
        verify_capacity_gemm_v2_strict(
            context=context,
            payload=decoded,
            q_block_i32=q_block_i32,
            q_merkle_path=q_merkle_path,
            state_openings=state_openings,
        )
    except Exception as exc:
        if isinstance(
            exc, (CapacityGemmV2FormatError, CapacityGemmV2VerificationError)
        ):
            return False, str(exc)
        raise
    return True, "ok"


__all__ = [
    "CAPACITY_GEMM_V2_CHALLENGE_COUNT",
    "CAPACITY_GEMM_V2_FORMAT",
    "CAPACITY_GEMM_V2_PROTOCOL_VERSION",
    "CapacityGemmV2Context",
    "CapacityGemmV2FormatError",
    "CapacityGemmV2Payload",
    "CapacityGemmV2VerificationError",
    "capacity_gemm_v2_context_from_workspace_proof",
    "prove_capacity_gemm_v2",
    "required_transition_state_blocks",
    "verify_capacity_gemm_v2",
    "verify_capacity_gemm_v2_strict",
]
