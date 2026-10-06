"""Verifier for the GEMM sumcheck over the Pallas scalar field, produced by ``libverathos_gemm``.

A batch of block relations ``Y = X @ W`` is fixed by its statements before any challenge. One shared row/column point
is squeezed, the claimed Y evaluations are folded with Fiat-Shamir coefficients, and one degree-3 sumcheck over the
padded inner dimension reduces them to per-block terminal claims on X and W. The caller checks those terminal claims
(and the Y claims) against data it authenticates itself. MLE points use MSB-first variable order.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

PALLAS_SCALAR_MODULUS = int("40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001", 16)
SCALAR_BYTES = 32
PROOF_VERSION = 1
_PROOF_MAGIC = b"VGB1"
_TRANSCRIPT_DOMAIN = b"VERATHOS/GEMM/BATCH_SUMCHECK/PALLAS_SCALAR/SHA256"
_STATEMENT_DOMAIN = b"VERATHOS/GEMM/STATEMENT/SHA256"
_BATCH_STATEMENT_DOMAIN = b"VERATHOS/GEMM/BATCH_STATEMENT/SHA256"
_PROOF_HEADER = struct.Struct("<4sHHH")
_MAX_BLOCKS = 4096
_MAX_ROUNDS = 63
_MAX_FIELD_BYTES = 4096
_UINT64_MAX = (1 << 64) - 1
_CHALLENGE_LIMIT = (1 << 256) - ((1 << 256) % PALLAS_SCALAR_MODULUS)
_INV_TWO = pow(2, PALLAS_SCALAR_MODULUS - 2, PALLAS_SCALAR_MODULUS)
_INV_SIX = pow(6, PALLAS_SCALAR_MODULUS - 2, PALLAS_SCALAR_MODULUS)


class GemmFormatError(ValueError):
    """A statement or proof is not canonically encoded."""


class GemmVerificationError(ValueError):
    """The sumcheck did not verify."""


def _scalar(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) < PALLAS_SCALAR_MODULUS:
        raise GemmFormatError(f"{field_name} is not a canonical scalar")
    return int(value)


def scalar_to_bytes(value: int) -> bytes:
    return _scalar(value, "scalar").to_bytes(SCALAR_BYTES, "little")


def scalar_from_bytes(encoded: bytes) -> int:
    if not isinstance(encoded, bytes) or len(encoded) != SCALAR_BYTES:
        raise GemmFormatError("scalar encoding must be exactly 32 bytes")
    value = int.from_bytes(encoded, "little")
    if value >= PALLAS_SCALAR_MODULUS:
        raise GemmFormatError("scalar encoding is not canonical")
    return value


def _uint64(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) <= _UINT64_MAX:
        raise GemmFormatError(f"{field_name} is outside the uint64 range")
    return int(value)


def _power_of_two(value: int, field_name: str) -> int:
    dimension = _uint64(value, field_name)
    if dimension == 0 or dimension & (dimension - 1):
        raise GemmFormatError(f"{field_name} must be a positive power of two")
    return dimension


def _bytes_field(value: bytes, field_name: str, *, exact: int | None = None) -> bytes:
    if not isinstance(value, bytes) or not value or len(value) > _MAX_FIELD_BYTES \
            or (exact is not None and len(value) != exact):
        raise GemmFormatError(f"{field_name} has an invalid length")
    return value


def _record(fields: Sequence[tuple[bytes, bytes]]) -> bytes:
    """Length-framed (label, value) encoding used by statements and transcripts."""
    encoded = bytearray()
    for label, value in fields:
        if not isinstance(label, bytes) or not label or not isinstance(value, bytes):
            raise GemmFormatError("record field is malformed")
        encoded += struct.pack("<I", len(label)) + label + struct.pack("<Q", len(value)) + value
    return bytes(encoded)


@dataclass(frozen=True)
class GemmStatement:
    """One padded block relation and the bindings that fix its operands and output."""

    outer_digest: bytes
    operation_identity: bytes
    row_block: int
    column_block: int
    rows: int
    inner: int
    valid_inner: int
    columns: int
    x_binding: bytes
    w_binding: bytes
    y_binding: bytes

    def __post_init__(self) -> None:
        _bytes_field(self.outer_digest, "outer_digest", exact=32)
        _bytes_field(self.operation_identity, "operation_identity")
        _uint64(self.row_block, "row_block")
        _uint64(self.column_block, "column_block")
        _power_of_two(self.rows, "rows")
        inner = _power_of_two(self.inner, "inner")
        if not 0 < _uint64(self.valid_inner, "valid_inner") <= inner:
            raise GemmFormatError("valid_inner must be in [1, inner]")
        _power_of_two(self.columns, "columns")
        if inner.bit_length() - 1 > _MAX_ROUNDS:
            raise GemmFormatError("inner dimension requires too many rounds")
        for name in ("x_binding", "w_binding", "y_binding"):
            _bytes_field(getattr(self, name), name)

    def canonical_bytes(self) -> bytes:
        return _record((
            (b"outer_digest", self.outer_digest),
            (b"operation_identity", self.operation_identity),
            (b"row_block", struct.pack("<Q", self.row_block)),
            (b"column_block", struct.pack("<Q", self.column_block)),
            (b"rows", struct.pack("<Q", self.rows)),
            (b"inner", struct.pack("<Q", self.inner)),
            (b"valid_inner", struct.pack("<Q", self.valid_inner)),
            (b"columns", struct.pack("<Q", self.columns)),
            (b"x_binding", self.x_binding),
            (b"w_binding", self.w_binding),
            (b"y_binding", self.y_binding),
        ))

    def digest(self) -> bytes:
        return hashlib.sha256(_STATEMENT_DOMAIN + self.canonical_bytes()).digest()


@dataclass(frozen=True)
class GemmBatchStatement:
    statements: tuple[GemmStatement, ...]

    def __post_init__(self) -> None:
        statements = tuple(self.statements)
        if not 0 < len(statements) <= _MAX_BLOCKS or not all(isinstance(s, GemmStatement) for s in statements):
            raise GemmFormatError("batch statements are invalid")
        object.__setattr__(self, "statements", statements)

    @property
    def rows(self) -> int:
        return max(s.rows for s in self.statements)

    @property
    def inner(self) -> int:
        return max(s.inner for s in self.statements)

    @property
    def columns(self) -> int:
        return max(s.columns for s in self.statements)

    def digest(self) -> bytes:
        fields = [(b"statement_count", struct.pack("<I", len(self.statements)))]
        fields += [(b"statement" + struct.pack("<I", i), s.canonical_bytes()) for i, s in enumerate(self.statements)]
        return hashlib.sha256(_BATCH_STATEMENT_DOMAIN + _record(fields)).digest()


@dataclass(frozen=True)
class GemmSumcheckRound:
    """Degree-3 round polynomial evaluated at 0, 1, 2 and 3."""

    e0: int
    e1: int
    e2: int
    e3: int

    def __post_init__(self) -> None:
        for name in ("e0", "e1", "e2", "e3"):
            _scalar(getattr(self, name), f"round {name}")


@dataclass(frozen=True)
class GemmBatchProof:
    claimed_y_evals: tuple[int, ...]
    rounds: tuple[GemmSumcheckRound, ...]
    final_x_evals: tuple[int, ...]
    final_w_evals: tuple[int, ...]

    def __post_init__(self) -> None:
        y, x, w, rounds = (tuple(v) for v in (self.claimed_y_evals, self.final_x_evals, self.final_w_evals,
                                              self.rounds))
        if not 0 < len(y) <= _MAX_BLOCKS or len(x) != len(y) or len(w) != len(y) or len(rounds) > _MAX_ROUNDS:
            raise GemmFormatError("batch proof dimensions are invalid")
        for value in y + x + w:
            _scalar(value, "evaluation")
        if not all(isinstance(r, GemmSumcheckRound) for r in rounds):
            raise GemmFormatError("batch proof contains an invalid round")
        for name, value in (("claimed_y_evals", y), ("rounds", rounds), ("final_x_evals", x), ("final_w_evals", w)):
            object.__setattr__(self, name, value)

    def canonical_bytes(self) -> bytes:
        encoded = bytearray(_PROOF_HEADER.pack(_PROOF_MAGIC, PROOF_VERSION, len(self.claimed_y_evals),
                                               len(self.rounds)))
        values = [*self.claimed_y_evals, *(v for r in self.rounds for v in (r.e0, r.e1, r.e2, r.e3)),
                  *self.final_x_evals, *self.final_w_evals]
        for value in values:
            encoded += scalar_to_bytes(value)
        return bytes(encoded)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes, *, expected_blocks: int, expected_rounds: int) -> GemmBatchProof:
        if not isinstance(encoded, bytes) or len(encoded) < _PROOF_HEADER.size:
            raise GemmFormatError("batch proof encoding is truncated")
        magic, version, blocks, rounds = _PROOF_HEADER.unpack_from(encoded)
        if magic != _PROOF_MAGIC or version != PROOF_VERSION:
            raise GemmFormatError("batch proof header is not supported")
        if blocks != expected_blocks:
            raise GemmFormatError("batch proof block count does not match")
        if rounds != expected_rounds:
            raise GemmFormatError("batch proof round count does not match")
        if len(encoded) != _PROOF_HEADER.size + (3 * blocks + 4 * rounds) * SCALAR_BYTES:
            raise GemmFormatError("batch proof encoding has an invalid length")
        scalars = iter(scalar_from_bytes(encoded[o:o + SCALAR_BYTES])
                       for o in range(_PROOF_HEADER.size, len(encoded), SCALAR_BYTES))
        y = tuple(next(scalars) for _ in range(blocks))
        round_values = tuple(GemmSumcheckRound(*(next(scalars) for _ in range(4))) for _ in range(rounds))
        x = tuple(next(scalars) for _ in range(blocks))
        w = tuple(next(scalars) for _ in range(blocks))
        return cls(y, round_values, x, w)


@dataclass(frozen=True)
class GemmOpeningClaims:
    """Points and values the caller must check against authenticated X, W and Y data."""

    row_challenges: tuple[int, ...]
    column_challenges: tuple[int, ...]
    inner_challenges: tuple[int, ...]
    y_values: tuple[int, ...]
    x_values: tuple[int, ...]
    w_values: tuple[int, ...]

    @property
    def y_point(self) -> tuple[int, ...]:
        return self.row_challenges + self.column_challenges


class _Transcript:
    def __init__(self, outer_digest: bytes) -> None:
        self._state = hashlib.sha256(_TRANSCRIPT_DOMAIN + _record(((b"outer_digest", outer_digest),))).digest()

    def absorb_scalar(self, label: bytes, value: int) -> None:
        self._state = hashlib.sha256(self._state + _record(((label, scalar_to_bytes(value)),))).digest()

    def squeeze_scalar(self, label: bytes) -> int:
        """Uniform scalar by rejection sampling; the accepted draw is absorbed back into the state."""
        for attempt in range(1 << 32):
            digest = hashlib.sha256(self._state + _record(((b"challenge_label", label),
                                                           (b"attempt", struct.pack("<I", attempt))))).digest()
            candidate = int.from_bytes(digest, "little")
            if candidate < _CHALLENGE_LIMIT:
                scalar = candidate % PALLAS_SCALAR_MODULUS
                challenge = _record(((b"label", label), (b"digest", digest), (b"scalar", scalar_to_bytes(scalar))))
                self._state = hashlib.sha256(self._state + _record(((b"challenge", challenge),))).digest()
                return scalar
        raise RuntimeError("challenge rejection sampling did not terminate")


def _interpolate_degree_three(round_: GemmSumcheckRound, r: int) -> int:
    modulus = PALLAS_SCALAR_MODULUS
    l0 = (-(r - 1) * (r - 2) * (r - 3) * _INV_SIX) % modulus
    l1 = (r * (r - 2) * (r - 3) * _INV_TWO) % modulus
    l2 = (-r * (r - 1) * (r - 3) * _INV_TWO) % modulus
    l3 = (r * (r - 1) * (r - 2) * _INV_SIX) % modulus
    return (round_.e0 * l0 + round_.e1 * l1 + round_.e2 * l2 + round_.e3 * l3) % modulus


def _validity_mask_mle(valid_inner: int, inner: int, point: Sequence[int]) -> int:
    """MLE of the prefix mask [1]*valid_inner + [0]*(inner - valid_inner), without materializing it."""
    if inner != 1 << len(point):
        raise GemmFormatError("validity mask point does not match padded inner")

    def evaluate(valid: int, size: int, offset: int) -> int:
        if valid == 0:
            return 0
        if valid == size:
            return 1
        r = _scalar(point[offset], "validity mask challenge")
        half = size // 2
        low = evaluate(min(valid, half), half, offset + 1)
        high = evaluate(max(valid - half, 0), half, offset + 1)
        return ((1 - r) * low + r * high) % PALLAS_SCALAR_MODULUS

    return evaluate(valid_inner, inner, 0)


def verify_gemm_batch_sumcheck(statement: GemmBatchStatement, proof: GemmBatchProof) -> GemmOpeningClaims:
    if len(proof.claimed_y_evals) != len(statement.statements):
        raise GemmVerificationError("batch proof statement count does not match")
    if len(proof.rounds) != statement.inner.bit_length() - 1:
        raise GemmVerificationError("batch sumcheck round count does not match")
    modulus = PALLAS_SCALAR_MODULUS
    transcript = _Transcript(statement.digest())
    row = tuple(transcript.squeeze_scalar(b"r_i" + struct.pack("<I", i))
                for i in range(statement.rows.bit_length() - 1))
    column = tuple(transcript.squeeze_scalar(b"r_j" + struct.pack("<I", i))
                   for i in range(statement.columns.bit_length() - 1))
    for i, value in enumerate(proof.claimed_y_evals):
        transcript.absorb_scalar(b"claimed_y_eval" + struct.pack("<I", i), value)
    coefficients = [1] + [transcript.squeeze_scalar(b"relation_alpha" + struct.pack("<I", i))
                          for i in range(1, len(proof.claimed_y_evals))]

    claim = sum(c * v for c, v in zip(coefficients, proof.claimed_y_evals)) % modulus
    inner = []
    for i, round_ in enumerate(proof.rounds):
        if (round_.e0 + round_.e1) % modulus != claim:
            raise GemmVerificationError("batch sumcheck round does not match the current claim")
        suffix = struct.pack("<I", i)
        for name in ("e0", "e1", "e2", "e3"):
            transcript.absorb_scalar(b"round_" + name.encode() + suffix, getattr(round_, name))
        challenge = transcript.squeeze_scalar(b"r_k" + suffix)
        inner.append(challenge)
        claim = _interpolate_degree_three(round_, challenge)

    expected = 0
    for c, item, x, w in zip(coefficients, statement.statements, proof.final_x_evals, proof.final_w_evals):
        expected = (expected + c * x * w * _validity_mask_mle(item.valid_inner, statement.inner, inner)) % modulus
    if claim != expected:
        raise GemmVerificationError("batch sumcheck terminal claim does not match")
    return GemmOpeningClaims(row, column, tuple(inner), proof.claimed_y_evals, proof.final_x_evals,
                             proof.final_w_evals)
