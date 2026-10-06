"""Reference GEMM sumcheck core for proof protocol v2.

The PCS wrapper is responsible for authenticating the three evaluation claims
returned by :func:`verify_gemm_v2_sumcheck` against the commitments carried in
the statement. MLE points use Nova's MSB-first variable order throughout.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


# Scalar field of the Pallas curve. This is Fq in the Pasta cycle.
PALLAS_SCALAR_MODULUS = int(
    "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    16,
)

PROTOCOL_VERSION = 2
SCALAR_BYTES = 32
PCS_ENCODING_SIGNED_I8 = 1
PCS_ENCODING_SIGNED_I64 = 2
_TRANSCRIPT_DOMAIN = b"VERATHOS/GEMM_V2/SUMCHECK/PALLAS_SCALAR/SHA256"
_STATEMENT_DOMAIN = b"VERATHOS/GEMM_V2/STATEMENT/SHA256"
_PROOF_MAGIC = b"VG2S"
_MAX_COMMITMENT_BYTES = 4096
_MAX_OPERATION_IDENTITY_BYTES = 4096
_MAX_SUMCHECK_ROUNDS = 63
_UINT64_MAX = (1 << 64) - 1
_CHALLENGE_LIMIT = (1 << 256) - ((1 << 256) % PALLAS_SCALAR_MODULUS)
_INV_TWO = pow(2, PALLAS_SCALAR_MODULUS - 2, PALLAS_SCALAR_MODULUS)
_INV_SIX = pow(6, PALLAS_SCALAR_MODULUS - 2, PALLAS_SCALAR_MODULUS)


class GemmV2FormatError(ValueError):
    """A protocol v2 statement or proof is not canonically encoded."""


class GemmV2VerificationError(ValueError):
    """A protocol v2 GEMM sumcheck did not verify."""


def scalar_to_bytes(value: int) -> bytes:
    """Encode one canonical Pallas scalar in 32-byte little-endian form."""

    scalar = _canonical_scalar(value, "scalar")
    return scalar.to_bytes(SCALAR_BYTES, "little")


def scalar_from_bytes(encoded: bytes) -> int:
    """Decode one canonical 32-byte little-endian Pallas scalar."""

    if not isinstance(encoded, bytes) or len(encoded) != SCALAR_BYTES:
        raise GemmV2FormatError("scalar encoding must be exactly 32 bytes")
    value = int.from_bytes(encoded, "little")
    if value >= PALLAS_SCALAR_MODULUS:
        raise GemmV2FormatError("scalar encoding is not canonical")
    return value


def _canonical_scalar(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GemmV2FormatError(f"{field_name} must be an integer")
    scalar = int(value)
    if scalar < 0 or scalar >= PALLAS_SCALAR_MODULUS:
        raise GemmV2FormatError(f"{field_name} is not a canonical scalar")
    return scalar


def _uint64(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GemmV2FormatError(f"{field_name} must be an integer")
    integer = int(value)
    if integer < 0 or integer > _UINT64_MAX:
        raise GemmV2FormatError(f"{field_name} is outside the uint64 range")
    return integer


def _fixed_bytes(value: bytes, size: int, field_name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != size:
        raise GemmV2FormatError(f"{field_name} must be exactly {size} bytes")
    return value


def _bounded_bytes(value: bytes, maximum: int, field_name: str) -> bytes:
    if not isinstance(value, bytes) or not value:
        raise GemmV2FormatError(f"{field_name} must be non-empty bytes")
    if len(value) > maximum:
        raise GemmV2FormatError(f"{field_name} is too long")
    return value


def _power_of_two_dimension(value: int, field_name: str) -> int:
    dimension = _uint64(value, field_name)
    if dimension == 0 or dimension & (dimension - 1):
        raise GemmV2FormatError(f"{field_name} must be a positive power of two")
    return dimension


def _fixed_encoding_id(value: int, expected: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GemmV2FormatError(f"{field_name} must be an integer")
    if int(value) != expected:
        raise GemmV2FormatError(f"{field_name} is not supported")


@dataclass(frozen=True)
class GemmV2Statement:
    """Exact statement context for one padded GEMM block relation."""

    outer_digest: bytes
    operation_identity: bytes
    row_block: int
    column_block: int
    rows: int
    inner: int
    valid_inner: int
    columns: int
    x_commitment: bytes
    w_commitment: bytes
    y_commitment: bytes
    x_encoding_id: int = PCS_ENCODING_SIGNED_I8
    w_encoding_id: int = PCS_ENCODING_SIGNED_I8
    y_encoding_id: int = PCS_ENCODING_SIGNED_I64

    def __post_init__(self) -> None:
        _fixed_bytes(self.outer_digest, 32, "outer_digest")
        _bounded_bytes(
            self.operation_identity,
            _MAX_OPERATION_IDENTITY_BYTES,
            "operation_identity",
        )
        _uint64(self.row_block, "row_block")
        _uint64(self.column_block, "column_block")
        _power_of_two_dimension(self.rows, "rows")
        inner = _power_of_two_dimension(self.inner, "inner")
        valid_inner = _uint64(self.valid_inner, "valid_inner")
        if valid_inner == 0 or valid_inner > inner:
            raise GemmV2FormatError("valid_inner must be in [1, inner]")
        _power_of_two_dimension(self.columns, "columns")
        if inner.bit_length() - 1 > _MAX_SUMCHECK_ROUNDS:
            raise GemmV2FormatError("inner dimension requires too many rounds")
        _bounded_bytes(self.x_commitment, _MAX_COMMITMENT_BYTES, "x_commitment")
        _bounded_bytes(self.w_commitment, _MAX_COMMITMENT_BYTES, "w_commitment")
        _bounded_bytes(self.y_commitment, _MAX_COMMITMENT_BYTES, "y_commitment")
        _fixed_encoding_id(
            self.x_encoding_id,
            PCS_ENCODING_SIGNED_I8,
            "x_encoding_id",
        )
        _fixed_encoding_id(
            self.w_encoding_id,
            PCS_ENCODING_SIGNED_I8,
            "w_encoding_id",
        )
        _fixed_encoding_id(
            self.y_encoding_id,
            PCS_ENCODING_SIGNED_I64,
            "y_encoding_id",
        )

    @property
    def row_bits(self) -> int:
        return self.rows.bit_length() - 1

    @property
    def inner_bits(self) -> int:
        return self.inner.bit_length() - 1

    @property
    def column_bits(self) -> int:
        return self.columns.bit_length() - 1

    def canonical_bytes(self) -> bytes:
        """Encode the exact block statement with length-framed fields."""

        return _record(
            (
                (b"protocol_version", struct.pack("<H", PROTOCOL_VERSION)),
                (b"parent_outer_digest", self.outer_digest),
                (b"operation_identity", self.operation_identity),
                (b"row_block", struct.pack("<Q", self.row_block)),
                (b"column_block", struct.pack("<Q", self.column_block)),
                (b"rows", struct.pack("<Q", self.rows)),
                (b"inner", struct.pack("<Q", self.inner)),
                (b"valid_inner", struct.pack("<Q", self.valid_inner)),
                (b"columns", struct.pack("<Q", self.columns)),
                (b"x_encoding_id", struct.pack("<B", self.x_encoding_id)),
                (b"x_commitment", self.x_commitment),
                (b"w_encoding_id", struct.pack("<B", self.w_encoding_id)),
                (b"w_commitment", self.w_commitment),
                (b"y_encoding_id", struct.pack("<B", self.y_encoding_id)),
                (b"y_commitment", self.y_commitment),
            )
        )

    def digest(self) -> bytes:
        """Return the exact 32-byte outer digest for the PCS openings."""

        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True)
class GemmV2SumcheckRound:
    """Four evaluations of one degree-3 sumcheck round polynomial."""

    e0: int
    e1: int
    e2: int
    e3: int

    def __post_init__(self) -> None:
        _canonical_scalar(self.e0, "round e0")
        _canonical_scalar(self.e1, "round e1")
        _canonical_scalar(self.e2, "round e2")
        _canonical_scalar(self.e3, "round e3")


@dataclass(frozen=True)
class GemmV2Proof:
    """Canonical arithmetic proof for one protocol v2 GEMM block."""

    claimed_y_eval: int
    rounds: tuple[GemmV2SumcheckRound, ...]
    final_x_eval: int
    final_w_eval: int

    def __post_init__(self) -> None:
        _canonical_scalar(self.claimed_y_eval, "claimed_y_eval")
        rounds = tuple(self.rounds)
        if len(rounds) > _MAX_SUMCHECK_ROUNDS:
            raise GemmV2FormatError("proof contains too many sumcheck rounds")
        if not all(isinstance(round_, GemmV2SumcheckRound) for round_ in rounds):
            raise GemmV2FormatError("rounds must contain GemmV2SumcheckRound values")
        object.__setattr__(self, "rounds", rounds)
        _canonical_scalar(self.final_x_eval, "final_x_eval")
        _canonical_scalar(self.final_w_eval, "final_w_eval")

    def canonical_bytes(self) -> bytes:
        """Serialize with fixed-width canonical scalar encodings."""

        encoded = bytearray(
            struct.pack("<4sHH", _PROOF_MAGIC, PROTOCOL_VERSION, len(self.rounds))
        )
        encoded.extend(scalar_to_bytes(self.claimed_y_eval))
        for round_ in self.rounds:
            encoded.extend(scalar_to_bytes(round_.e0))
            encoded.extend(scalar_to_bytes(round_.e1))
            encoded.extend(scalar_to_bytes(round_.e2))
            encoded.extend(scalar_to_bytes(round_.e3))
        encoded.extend(scalar_to_bytes(self.final_x_eval))
        encoded.extend(scalar_to_bytes(self.final_w_eval))
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
        *,
        expected_rounds: int | None = None,
    ) -> "GemmV2Proof":
        """Decode a proof and reject noncanonical or trailing data."""

        if not isinstance(encoded, bytes):
            raise GemmV2FormatError("proof encoding must be bytes")
        header_size = struct.calcsize("<4sHH")
        minimum_size = header_size + 3 * SCALAR_BYTES
        if len(encoded) < minimum_size:
            raise GemmV2FormatError("proof encoding is truncated")
        magic, version, round_count = struct.unpack_from("<4sHH", encoded)
        if magic != _PROOF_MAGIC:
            raise GemmV2FormatError("proof magic is not recognized")
        if version != PROTOCOL_VERSION:
            raise GemmV2FormatError("proof version is not supported")
        if round_count > _MAX_SUMCHECK_ROUNDS:
            raise GemmV2FormatError("proof contains too many sumcheck rounds")
        if expected_rounds is not None and round_count != expected_rounds:
            raise GemmV2FormatError("proof round count does not match the statement")
        expected_size = header_size + (3 + 4 * round_count) * SCALAR_BYTES
        if len(encoded) != expected_size:
            raise GemmV2FormatError("proof encoding has an invalid length")

        offset = header_size

        def read_scalar() -> int:
            nonlocal offset
            value = scalar_from_bytes(encoded[offset : offset + SCALAR_BYTES])
            offset += SCALAR_BYTES
            return value

        claimed_y_eval = read_scalar()
        rounds = tuple(
            GemmV2SumcheckRound(
                read_scalar(),
                read_scalar(),
                read_scalar(),
                read_scalar(),
            )
            for _ in range(round_count)
        )
        final_x_eval = read_scalar()
        final_w_eval = read_scalar()
        return cls(claimed_y_eval, rounds, final_x_eval, final_w_eval)


@dataclass(frozen=True)
class GemmV2OpeningClaims:
    """PCS evaluation claims and K-axis points from successful verification."""

    x_point: tuple[int, ...]
    x_value: int
    w_point: tuple[int, ...]
    w_value: int
    y_point: tuple[int, ...]
    y_value: int
    row_challenges: tuple[int, ...]
    column_challenges: tuple[int, ...]
    inner_challenges: tuple[int, ...]
    x_k_point: tuple[int, ...]
    w_k_point: tuple[int, ...]
    validity_mask_value: int
    pcs_outer_digest: bytes


class _Transcript:
    def __init__(self, outer_digest: bytes):
        self._state = hashlib.sha256(
            _TRANSCRIPT_DOMAIN + _record(((b"outer_digest", outer_digest),))
        ).digest()

    def absorb_bytes(self, label: bytes, value: bytes) -> None:
        self._state = hashlib.sha256(
            self._state + _record(((label, value),))
        ).digest()

    def absorb_scalar(self, label: bytes, value: int) -> None:
        self.absorb_bytes(label, scalar_to_bytes(value))

    def squeeze_scalar(self, label: bytes) -> int:
        for attempt in range(1 << 32):
            digest = hashlib.sha256(
                self._state
                + _record(
                    (
                        (b"challenge_label", label),
                        (b"attempt", struct.pack("<I", attempt)),
                    )
                )
            ).digest()
            candidate = int.from_bytes(digest, "little")
            if candidate < _CHALLENGE_LIMIT:
                scalar = candidate % PALLAS_SCALAR_MODULUS
                self.absorb_bytes(
                    b"challenge",
                    _record(
                        (
                            (b"label", label),
                            (b"digest", digest),
                            (b"scalar", scalar_to_bytes(scalar)),
                        )
                    ),
                )
                return scalar
        raise RuntimeError("challenge rejection sampling did not terminate")


def _record(fields: Sequence[tuple[bytes, bytes]]) -> bytes:
    encoded = bytearray()
    for label, value in fields:
        if not isinstance(label, bytes) or not label:
            raise GemmV2FormatError("transcript labels must be non-empty bytes")
        if not isinstance(value, bytes):
            raise GemmV2FormatError("transcript values must be bytes")
        encoded.extend(struct.pack("<I", len(label)))
        encoded.extend(label)
        encoded.extend(struct.pack("<Q", len(value)))
        encoded.extend(value)
    return bytes(encoded)


def _statement_transcript(statement: GemmV2Statement) -> _Transcript:
    if not isinstance(statement, GemmV2Statement):
        raise GemmV2FormatError("statement has an unexpected type")
    return _Transcript(statement.digest())


def _index_challenges(
    statement: GemmV2Statement,
) -> tuple[_Transcript, tuple[int, ...], tuple[int, ...]]:
    transcript = _statement_transcript(statement)
    row_challenges = tuple(
        transcript.squeeze_scalar(b"r_i" + struct.pack("<I", index))
        for index in range(statement.row_bits)
    )
    column_challenges = tuple(
        transcript.squeeze_scalar(b"r_j" + struct.pack("<I", index))
        for index in range(statement.column_bits)
    )
    return transcript, row_challenges, column_challenges


def derive_gemm_v2_index_challenges(
    statement: GemmV2Statement,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive the row and column points after all statement commitments."""

    _, row_challenges, column_challenges = _index_challenges(statement)
    return row_challenges, column_challenges


def _field_value(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GemmV2FormatError(f"{field_name} entries must be integers")
    integer = int(value)
    if integer <= -PALLAS_SCALAR_MODULUS or integer >= PALLAS_SCALAR_MODULUS:
        raise GemmV2FormatError(f"{field_name} entry is outside the field range")
    return integer % PALLAS_SCALAR_MODULUS


def _matrix_to_field(
    matrix: Sequence[Sequence[int]],
    rows: int,
    columns: int,
    field_name: str,
) -> list[list[int]]:
    try:
        actual_rows = len(matrix)
    except TypeError as exc:
        raise GemmV2FormatError(f"{field_name} must be a matrix") from exc
    if actual_rows != rows:
        raise GemmV2FormatError(f"{field_name} row count does not match the statement")
    result: list[list[int]] = []
    for row in matrix:
        try:
            actual_columns = len(row)
        except TypeError as exc:
            raise GemmV2FormatError(f"{field_name} must be a matrix") from exc
        if actual_columns != columns:
            raise GemmV2FormatError(
                f"{field_name} column count does not match the statement"
            )
        result.append([_field_value(value, field_name) for value in row])
    return result


def _fold(values: Sequence[int], point: Sequence[int]) -> int:
    table = list(values)
    if len(table) != 1 << len(point):
        raise GemmV2FormatError("MLE table size does not match its evaluation point")
    modulus = PALLAS_SCALAR_MODULUS
    for challenge in point:
        r = _canonical_scalar(challenge, "MLE challenge")
        half = len(table) // 2
        table = [
            (table[index] + r * (table[index + half] - table[index])) % modulus
            for index in range(half)
        ]
    return table[0]


def _fold_x_rows(x: Sequence[Sequence[int]], row_point: Sequence[int]) -> list[int]:
    inner = len(x[0])
    return [_fold([row[index] for row in x], row_point) for index in range(inner)]


def _fold_w_columns(
    w: Sequence[Sequence[int]],
    column_point: Sequence[int],
) -> list[int]:
    return [_fold(row, column_point) for row in w]


def _evaluate_y(
    y: Sequence[Sequence[int]],
    row_point: Sequence[int],
    column_point: Sequence[int],
) -> int:
    row_folded_by_column = [
        _fold([row[column] for row in y], row_point)
        for column in range(len(y[0]))
    ]
    return _fold(row_folded_by_column, column_point)


def _round_evaluations(
    x_table: Sequence[int],
    w_table: Sequence[int],
    mask_table: Sequence[int],
) -> GemmV2SumcheckRound:
    modulus = PALLAS_SCALAR_MODULUS
    evaluations = [0, 0, 0, 0]
    half = len(x_table) // 2
    for index in range(half):
        x0 = x_table[index]
        x1 = x_table[index + half]
        w0 = w_table[index]
        w1 = w_table[index + half]
        mask0 = mask_table[index]
        mask1 = mask_table[index + half]
        for point in range(4):
            x_value = (x0 + point * (x1 - x0)) % modulus
            w_value = (w0 + point * (w1 - w0)) % modulus
            mask_value = (mask0 + point * (mask1 - mask0)) % modulus
            evaluations[point] = (
                evaluations[point] + x_value * w_value * mask_value
            ) % modulus
    return GemmV2SumcheckRound(*evaluations)


def _fold_one(values: Sequence[int], challenge: int) -> list[int]:
    modulus = PALLAS_SCALAR_MODULUS
    half = len(values) // 2
    return [
        (values[index] + challenge * (values[index + half] - values[index])) % modulus
        for index in range(half)
    ]


def _interpolate_degree_three(round_: GemmV2SumcheckRound, challenge: int) -> int:
    modulus = PALLAS_SCALAR_MODULUS
    r = challenge
    l0 = (-(r - 1) * (r - 2) * (r - 3) * _INV_SIX) % modulus
    l1 = (r * (r - 2) * (r - 3) * _INV_TWO) % modulus
    l2 = (-r * (r - 1) * (r - 3) * _INV_TWO) % modulus
    l3 = (r * (r - 1) * (r - 2) * _INV_SIX) % modulus
    return (
        round_.e0 * l0
        + round_.e1 * l1
        + round_.e2 * l2
        + round_.e3 * l3
    ) % modulus


def _validity_mask_mle(valid_inner: int, inner: int, point: Sequence[int]) -> int:
    """Evaluate the canonical prefix mask without materializing its table."""

    if inner != 1 << len(point):
        raise GemmV2FormatError("validity mask point does not match padded inner")
    modulus = PALLAS_SCALAR_MODULUS

    def evaluate(valid: int, size: int, offset: int) -> int:
        if valid == 0:
            return 0
        if valid == size:
            return 1
        r = _canonical_scalar(point[offset], "validity mask challenge")
        half = size // 2
        low = evaluate(min(valid, half), half, offset + 1)
        high = evaluate(max(valid - half, 0), half, offset + 1)
        return ((1 - r) * low + r * high) % modulus

    return evaluate(valid_inner, inner, 0)


def _absorb_claim(transcript: _Transcript, claimed_y_eval: int) -> None:
    transcript.absorb_scalar(b"claimed_y_eval", claimed_y_eval)


def _absorb_round(
    transcript: _Transcript,
    round_index: int,
    round_: GemmV2SumcheckRound,
) -> int:
    transcript.absorb_bytes(b"round_index", struct.pack("<I", round_index))
    transcript.absorb_scalar(b"round_e0", round_.e0)
    transcript.absorb_scalar(b"round_e1", round_.e1)
    transcript.absorb_scalar(b"round_e2", round_.e2)
    transcript.absorb_scalar(b"round_e3", round_.e3)
    return transcript.squeeze_scalar(b"r_k" + struct.pack("<I", round_index))


def prove_gemm_v2_reference(
    statement: GemmV2Statement,
    x_band: Sequence[Sequence[int]],
    w_band: Sequence[Sequence[int]],
    y_block: Sequence[Sequence[int]],
) -> GemmV2Proof:
    """Produce the protocol v2 arithmetic proof for padded matrix bands."""

    if not isinstance(statement, GemmV2Statement):
        raise GemmV2FormatError("statement has an unexpected type")
    x = _matrix_to_field(x_band, statement.rows, statement.inner, "X")
    w = _matrix_to_field(w_band, statement.inner, statement.columns, "W")
    y = _matrix_to_field(y_block, statement.rows, statement.columns, "Y")

    transcript, row_challenges, column_challenges = _index_challenges(statement)
    claimed_y_eval = _evaluate_y(y, row_challenges, column_challenges)
    _absorb_claim(transcript, claimed_y_eval)

    x_table = _fold_x_rows(x, row_challenges)
    w_table = _fold_w_columns(w, column_challenges)
    mask_table = [1 if index < statement.valid_inner else 0 for index in range(statement.inner)]
    rounds = []
    for round_index in range(statement.inner_bits):
        round_ = _round_evaluations(x_table, w_table, mask_table)
        rounds.append(round_)
        challenge = _absorb_round(transcript, round_index, round_)
        x_table = _fold_one(x_table, challenge)
        w_table = _fold_one(w_table, challenge)
        mask_table = _fold_one(mask_table, challenge)

    return GemmV2Proof(
        claimed_y_eval=claimed_y_eval,
        rounds=tuple(rounds),
        final_x_eval=x_table[0],
        final_w_eval=w_table[0],
    )


def verify_gemm_v2_sumcheck(
    statement: GemmV2Statement,
    proof: GemmV2Proof,
) -> GemmV2OpeningClaims:
    """Verify the arithmetic core and return the exact PCS opening claims."""

    if not isinstance(statement, GemmV2Statement):
        raise GemmV2FormatError("statement has an unexpected type")
    if not isinstance(proof, GemmV2Proof):
        raise GemmV2FormatError("proof has an unexpected type")
    if len(proof.rounds) != statement.inner_bits:
        raise GemmV2VerificationError("sumcheck round count does not match the statement")

    transcript, row_challenges, column_challenges = _index_challenges(statement)
    _absorb_claim(transcript, proof.claimed_y_eval)
    current_claim = proof.claimed_y_eval
    inner_challenges = []
    modulus = PALLAS_SCALAR_MODULUS

    for round_index, round_ in enumerate(proof.rounds):
        if (round_.e0 + round_.e1) % modulus != current_claim:
            raise GemmV2VerificationError("sumcheck round does not match the current claim")
        challenge = _absorb_round(transcript, round_index, round_)
        inner_challenges.append(challenge)
        current_claim = _interpolate_degree_three(round_, challenge)

    inner_point = tuple(inner_challenges)
    mask_value = _validity_mask_mle(
        statement.valid_inner,
        statement.inner,
        inner_point,
    )
    expected_terminal = proof.final_x_eval * proof.final_w_eval % modulus
    expected_terminal = expected_terminal * mask_value % modulus
    if current_claim != expected_terminal:
        raise GemmV2VerificationError("sumcheck terminal claim does not match")

    return GemmV2OpeningClaims(
        x_point=tuple(row_challenges) + inner_point,
        x_value=proof.final_x_eval,
        w_point=inner_point + tuple(column_challenges),
        w_value=proof.final_w_eval,
        y_point=tuple(row_challenges) + tuple(column_challenges),
        y_value=proof.claimed_y_eval,
        row_challenges=tuple(row_challenges),
        column_challenges=tuple(column_challenges),
        inner_challenges=inner_point,
        x_k_point=inner_point,
        w_k_point=inner_point,
        validity_mask_value=mask_value,
        pcs_outer_digest=statement.digest(),
    )


__all__ = [
    "GemmV2FormatError",
    "GemmV2OpeningClaims",
    "GemmV2Proof",
    "GemmV2Statement",
    "GemmV2SumcheckRound",
    "GemmV2VerificationError",
    "PALLAS_SCALAR_MODULUS",
    "PCS_ENCODING_SIGNED_I8",
    "PCS_ENCODING_SIGNED_I64",
    "PROTOCOL_VERSION",
    "SCALAR_BYTES",
    "derive_gemm_v2_index_challenges",
    "prove_gemm_v2_reference",
    "scalar_from_bytes",
    "scalar_to_bytes",
    "verify_gemm_v2_sumcheck",
]
