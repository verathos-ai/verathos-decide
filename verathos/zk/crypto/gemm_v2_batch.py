"""Transcript-bound batching for proof-v2 GEMM sumchecks.

Every selected block remains an independent matrix relation.  The batch fixes
all block statements before deriving one shared row/column point, folds the
relations with Fiat--Shamir coefficients, and runs one sumcheck at a shared K
point.  The terminal X/W claims are kept per block so one same-point PCS
opening can authenticate all of them.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Sequence

from verathos.zk.crypto.gemm_v2_reference import (
    GemmV2FormatError,
    GemmV2Statement,
    GemmV2SumcheckRound,
    GemmV2VerificationError,
    PALLAS_SCALAR_MODULUS,
    _evaluate_y,
    _fold_one,
    _fold_w_columns,
    _fold_x_rows,
    _interpolate_degree_three,
    _matrix_to_field,
    _round_evaluations,
    _validity_mask_mle,
    scalar_from_bytes,
    scalar_to_bytes,
)


PROTOCOL_VERSION = 2
_BATCH_STATEMENT_DOMAIN = b"VERATHOS/GEMM_V2/BATCH_STATEMENT/SHA256"
_BATCH_TRANSCRIPT_DOMAIN = b"VERATHOS/GEMM_V2/BATCH_SUMCHECK/PALLAS_SCALAR/SHA256"
_PROOF_MAGIC = b"VGB2"
_MAX_BLOCKS = 4096
_MAX_ROUNDS = 63
_CHALLENGE_LIMIT = (1 << 256) - ((1 << 256) % PALLAS_SCALAR_MODULUS)


def _record(fields: Sequence[tuple[bytes, bytes]]) -> bytes:
    encoded = bytearray()
    for label, value in fields:
        if not isinstance(label, bytes) or not label or not isinstance(value, bytes):
            raise GemmV2FormatError("batch transcript field is malformed")
        encoded.extend(struct.pack("<I", len(label)))
        encoded.extend(label)
        encoded.extend(struct.pack("<Q", len(value)))
        encoded.extend(value)
    return bytes(encoded)


class _BatchTranscript:
    def __init__(self, outer_digest: bytes):
        if not isinstance(outer_digest, bytes) or len(outer_digest) != 32:
            raise GemmV2FormatError("batch outer digest must be exactly 32 bytes")
        self._state = hashlib.sha256(
            _BATCH_TRANSCRIPT_DOMAIN + _record(((b"outer_digest", outer_digest),))
        ).digest()

    def absorb_scalar(self, label: bytes, value: int) -> None:
        self._state = hashlib.sha256(
            self._state + _record(((label, scalar_to_bytes(value)),))
        ).digest()

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
                self._state = hashlib.sha256(
                    self._state
                    + _record(
                        (
                            (
                                b"challenge",
                                _record(
                                    (
                                        (b"label", label),
                                        (b"digest", digest),
                                        (b"scalar", scalar_to_bytes(scalar)),
                                    )
                                ),
                            ),
                        )
                    )
                ).digest()
                return scalar
        raise RuntimeError("batch challenge rejection sampling did not terminate")


@dataclass(frozen=True)
class GemmV2BatchStatement:
    statements: tuple[GemmV2Statement, ...]

    def __post_init__(self) -> None:
        statements = tuple(self.statements)
        if not statements or len(statements) > _MAX_BLOCKS:
            raise GemmV2FormatError("batch statement count is out of range")
        if not all(isinstance(item, GemmV2Statement) for item in statements):
            raise GemmV2FormatError("batch contains an unexpected statement type")
        object.__setattr__(self, "statements", statements)

    @property
    def rows(self) -> int:
        return max(item.rows for item in self.statements)

    @property
    def inner(self) -> int:
        return max(item.inner for item in self.statements)

    @property
    def columns(self) -> int:
        return max(item.columns for item in self.statements)

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
        fields: list[tuple[bytes, bytes]] = [
            (b"protocol_version", struct.pack("<H", PROTOCOL_VERSION)),
            (b"statement_count", struct.pack("<I", len(self.statements))),
        ]
        fields.extend(
            (b"statement" + struct.pack("<I", index), item.canonical_bytes())
            for index, item in enumerate(self.statements)
        )
        return _record(tuple(fields))

    def digest(self) -> bytes:
        return hashlib.sha256(_BATCH_STATEMENT_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True)
class GemmV2BatchProof:
    claimed_y_evals: tuple[int, ...]
    rounds: tuple[GemmV2SumcheckRound, ...]
    final_x_evals: tuple[int, ...]
    final_w_evals: tuple[int, ...]

    def __post_init__(self) -> None:
        y_values = tuple(self.claimed_y_evals)
        x_values = tuple(self.final_x_evals)
        w_values = tuple(self.final_w_evals)
        rounds = tuple(self.rounds)
        if (
            not y_values
            or len(y_values) > _MAX_BLOCKS
            or len(x_values) != len(y_values)
            or len(w_values) != len(y_values)
            or len(rounds) > _MAX_ROUNDS
        ):
            raise GemmV2FormatError("batch proof dimensions are invalid")
        for value in y_values + x_values + w_values:
            scalar_to_bytes(value)
        if not all(isinstance(item, GemmV2SumcheckRound) for item in rounds):
            raise GemmV2FormatError("batch proof contains an invalid round")
        object.__setattr__(self, "claimed_y_evals", y_values)
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "final_x_evals", x_values)
        object.__setattr__(self, "final_w_evals", w_values)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(
            struct.pack(
                "<4sHHH",
                _PROOF_MAGIC,
                PROTOCOL_VERSION,
                len(self.claimed_y_evals),
                len(self.rounds),
            )
        )
        for value in self.claimed_y_evals:
            encoded.extend(scalar_to_bytes(value))
        for round_ in self.rounds:
            for value in (round_.e0, round_.e1, round_.e2, round_.e3):
                encoded.extend(scalar_to_bytes(value))
        for value in self.final_x_evals:
            encoded.extend(scalar_to_bytes(value))
        for value in self.final_w_evals:
            encoded.extend(scalar_to_bytes(value))
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(
        cls,
        encoded: bytes,
        *,
        expected_blocks: int | None = None,
        expected_rounds: int | None = None,
    ) -> "GemmV2BatchProof":
        if not isinstance(encoded, bytes) or len(encoded) < struct.calcsize("<4sHHH"):
            raise GemmV2FormatError("batch proof encoding is truncated")
        magic, version, blocks, rounds = struct.unpack_from("<4sHHH", encoded)
        if magic != _PROOF_MAGIC or version != PROTOCOL_VERSION:
            raise GemmV2FormatError("batch proof header is not supported")
        if not 0 < blocks <= _MAX_BLOCKS or rounds > _MAX_ROUNDS:
            raise GemmV2FormatError("batch proof counts are out of range")
        if expected_blocks is not None and blocks != expected_blocks:
            raise GemmV2FormatError("batch proof block count does not match")
        if expected_rounds is not None and rounds != expected_rounds:
            raise GemmV2FormatError("batch proof round count does not match")
        header = struct.calcsize("<4sHHH")
        expected_size = header + (3 * blocks + 4 * rounds) * 32
        if len(encoded) != expected_size:
            raise GemmV2FormatError("batch proof encoding has an invalid length")
        offset = header

        def read_scalar() -> int:
            nonlocal offset
            value = scalar_from_bytes(encoded[offset : offset + 32])
            offset += 32
            return value

        y_values = tuple(read_scalar() for _ in range(blocks))
        round_values = tuple(
            GemmV2SumcheckRound(
                read_scalar(), read_scalar(), read_scalar(), read_scalar()
            )
            for _ in range(rounds)
        )
        x_values = tuple(read_scalar() for _ in range(blocks))
        w_values = tuple(read_scalar() for _ in range(blocks))
        result = cls(y_values, round_values, x_values, w_values)
        if result.canonical_bytes() != encoded:
            raise GemmV2FormatError("batch proof encoding is not canonical")
        return result


@dataclass(frozen=True)
class GemmV2BatchOpeningClaims:
    row_challenges: tuple[int, ...]
    column_challenges: tuple[int, ...]
    inner_challenges: tuple[int, ...]
    aggregation_coefficients: tuple[int, ...]
    y_values: tuple[int, ...]
    x_values: tuple[int, ...]
    w_values: tuple[int, ...]
    batch_statement_digest: bytes

    @property
    def y_point(self) -> tuple[int, ...]:
        return self.row_challenges + self.column_challenges


def _initial_transcript(
    statement: GemmV2BatchStatement,
) -> tuple[_BatchTranscript, tuple[int, ...], tuple[int, ...]]:
    transcript = _BatchTranscript(statement.digest())
    row = tuple(
        transcript.squeeze_scalar(b"r_i" + struct.pack("<I", index))
        for index in range(statement.row_bits)
    )
    column = tuple(
        transcript.squeeze_scalar(b"r_j" + struct.pack("<I", index))
        for index in range(statement.column_bits)
    )
    return transcript, row, column


def derive_gemm_v2_batch_index_challenges(
    statement: GemmV2BatchStatement,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _, row, column = _initial_transcript(statement)
    return row, column


def _absorb_claims_and_coefficients(
    transcript: _BatchTranscript,
    claimed_y_evals: Sequence[int],
) -> tuple[int, ...]:
    for index, value in enumerate(claimed_y_evals):
        transcript.absorb_scalar(b"claimed_y_eval" + struct.pack("<I", index), value)
    coefficients = [1]
    coefficients.extend(
        transcript.squeeze_scalar(b"relation_alpha" + struct.pack("<I", index))
        for index in range(1, len(claimed_y_evals))
    )
    return tuple(coefficients)


def _absorb_round(
    transcript: _BatchTranscript,
    round_index: int,
    round_: GemmV2SumcheckRound,
) -> int:
    transcript.absorb_scalar(b"round_e0" + struct.pack("<I", round_index), round_.e0)
    transcript.absorb_scalar(b"round_e1" + struct.pack("<I", round_index), round_.e1)
    transcript.absorb_scalar(b"round_e2" + struct.pack("<I", round_index), round_.e2)
    transcript.absorb_scalar(b"round_e3" + struct.pack("<I", round_index), round_.e3)
    return transcript.squeeze_scalar(b"r_k" + struct.pack("<I", round_index))


def prove_gemm_v2_batch_reference(
    statement: GemmV2BatchStatement,
    x_bands: Sequence[Sequence[Sequence[int]]],
    w_bands: Sequence[Sequence[Sequence[int]]],
    y_blocks: Sequence[Sequence[Sequence[int]]],
) -> GemmV2BatchProof:
    """Produce the exact reference batch proof."""

    count = len(statement.statements)
    if len(x_bands) != count or len(w_bands) != count or len(y_blocks) != count:
        raise GemmV2FormatError("batch witness count does not match its statement")
    xs = []
    ws = []
    ys = []
    for item, x_band, w_band, y_block in zip(
        statement.statements, x_bands, w_bands, y_blocks
    ):
        x = _matrix_to_field(x_band, item.rows, item.inner, "X")
        w = _matrix_to_field(w_band, item.inner, item.columns, "W")
        y = _matrix_to_field(y_block, item.rows, item.columns, "Y")
        xs.append(
            [row + [0] * (statement.inner - item.inner) for row in x]
            + [[0] * statement.inner for _ in range(statement.rows - item.rows)]
        )
        ws.append(
            [row + [0] * (statement.columns - item.columns) for row in w]
            + [[0] * statement.columns for _ in range(statement.inner - item.inner)]
        )
        ys.append(
            [row + [0] * (statement.columns - item.columns) for row in y]
            + [[0] * statement.columns for _ in range(statement.rows - item.rows)]
        )

    transcript, row, column = _initial_transcript(statement)
    claimed_y = tuple(_evaluate_y(y, row, column) for y in ys)
    coefficients = _absorb_claims_and_coefficients(transcript, claimed_y)
    x_tables = [_fold_x_rows(x, row) for x in xs]
    w_tables = [_fold_w_columns(w, column) for w in ws]
    masks = [
        [1 if index < item.valid_inner else 0 for index in range(statement.inner)]
        for item in statement.statements
    ]
    rounds = []
    modulus = PALLAS_SCALAR_MODULUS
    for round_index in range(statement.inner_bits):
        individual = tuple(
            _round_evaluations(x_table, w_table, mask)
            for x_table, w_table, mask in zip(x_tables, w_tables, masks)
        )
        values = tuple(
            sum(
                coefficient * getattr(round_, f"e{point}")
                for coefficient, round_ in zip(coefficients, individual)
            )
            % modulus
            for point in range(4)
        )
        aggregate = GemmV2SumcheckRound(*values)
        rounds.append(aggregate)
        challenge = _absorb_round(transcript, round_index, aggregate)
        x_tables = [_fold_one(table, challenge) for table in x_tables]
        w_tables = [_fold_one(table, challenge) for table in w_tables]
        masks = [_fold_one(table, challenge) for table in masks]
    return GemmV2BatchProof(
        claimed_y,
        tuple(rounds),
        tuple(table[0] for table in x_tables),
        tuple(table[0] for table in w_tables),
    )


def verify_gemm_v2_batch_sumcheck(
    statement: GemmV2BatchStatement,
    proof: GemmV2BatchProof,
) -> GemmV2BatchOpeningClaims:
    if len(proof.claimed_y_evals) != len(statement.statements):
        raise GemmV2VerificationError("batch proof statement count does not match")
    if len(proof.rounds) != statement.inner_bits:
        raise GemmV2VerificationError("batch sumcheck round count does not match")
    transcript, row, column = _initial_transcript(statement)
    coefficients = _absorb_claims_and_coefficients(transcript, proof.claimed_y_evals)
    modulus = PALLAS_SCALAR_MODULUS
    current_claim = (
        sum(
            coefficient * value
            for coefficient, value in zip(coefficients, proof.claimed_y_evals)
        )
        % modulus
    )
    inner = []
    for round_index, round_ in enumerate(proof.rounds):
        if (round_.e0 + round_.e1) % modulus != current_claim:
            raise GemmV2VerificationError(
                "batch sumcheck round does not match the current claim"
            )
        challenge = _absorb_round(transcript, round_index, round_)
        inner.append(challenge)
        current_claim = _interpolate_degree_three(round_, challenge)
    expected = 0
    for coefficient, item, x_value, w_value in zip(
        coefficients,
        statement.statements,
        proof.final_x_evals,
        proof.final_w_evals,
    ):
        mask = _validity_mask_mle(item.valid_inner, statement.inner, inner)
        expected = (expected + coefficient * x_value * w_value * mask) % modulus
    if current_claim != expected:
        raise GemmV2VerificationError("batch sumcheck terminal claim does not match")
    return GemmV2BatchOpeningClaims(
        row,
        column,
        tuple(inner),
        coefficients,
        proof.claimed_y_evals,
        proof.final_x_evals,
        proof.final_w_evals,
        statement.digest(),
    )


__all__ = [
    "GemmV2BatchOpeningClaims",
    "GemmV2BatchProof",
    "GemmV2BatchStatement",
    "derive_gemm_v2_batch_index_challenges",
    "prove_gemm_v2_batch_reference",
    "verify_gemm_v2_batch_sumcheck",
]
