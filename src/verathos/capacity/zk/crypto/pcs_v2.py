"""Pallas IPA multilinear polynomial commitments for proof protocol v2.

The native implementation commits to a signed integer or canonical Pallas field
vector after trailing-zero padding to a power of two. MLE coordinates use
Nova's MSB-first order: the first coordinate selects between the lower and upper
halves of the vector.
"""

from __future__ import annotations

import ctypes
import operator
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from threading import Lock, RLock
from typing import Final, Iterator


ABI_VERSION: Final = 10
PROTOCOL_VERSION: Final = 2
ENCODING_SIGNED_I8: Final = 1
ENCODING_SIGNED_I64: Final = 2
ENCODING_PALLAS_SCALAR: Final = 3
MAX_VECTOR_LEN: Final = 1 << 17
MAX_COMBINE_TERMS: Final = 512
MAX_CATALOG_COMBINE_TERMS: Final = 1 << 20
MAX_CATALOG_FOLDS: Final = 32
MAX_VALIDATE_COMMITMENTS: Final = 1 << 20
MAX_BATCH_COMMITMENTS: Final = 4096
MAX_BATCH_I8_BYTES: Final = 16 << 20
MAX_BATCH_I64_BYTES: Final = 64 << 20
MAX_GEMM_BATCH_BLOCKS: Final = 64
MAX_GEMM_BATCH_BYTES: Final = 16 << 20
MAX_LEAN_PROJECTION_VECTORS: Final = 2048
MAX_LEAN_PROJECTION_BYTES: Final = 512 << 20
MAX_MERKLE_BUILD_LEAVES: Final = 1 << 20
COMMITMENT_BYTES: Final = 32
SCALAR_BYTES: Final = 32
OUTER_DIGEST_BYTES: Final = 32

_REGISTERED_CATALOG_USE_LOCK = RLock()


@contextmanager
def registered_catalog_use() -> Iterator[None]:
    """Serialize native catalog residency with the fold that consumes it."""

    with _REGISTERED_CATALOG_USE_LOCK:
        yield


# The scalar field of the Pallas curve (Pasta Fq).
PALLAS_SCALAR_MODULUS: Final = int(
    "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    16,
)


class PCSUnavailableError(RuntimeError):
    """Raised when the native proof-protocol-v2 library cannot be loaded."""


class PCSNativeError(RuntimeError):
    """Raised when the native PCS implementation reports an internal failure."""


class PCSFormatError(PCSNativeError, ValueError):
    """Raised for malformed, non-canonical, or out-of-bounds PCS input."""


@dataclass(frozen=True, slots=True)
class PCSOpeningV2:
    """A self-describing v2 MLE opening.

    ``evaluation`` and every coordinate supplied to :func:`verify` are
    canonical 32-byte little-endian Pallas scalars.  ``proof`` is Nova's
    canonical compressed Pallas IPA argument.
    """

    commitment: bytes
    evaluation: bytes
    proof: bytes
    vector_length: int
    padded_length: int
    encoding: int


_U8 = ctypes.c_uint8
_U8_PTR = ctypes.POINTER(_U8)
_I8_PTR = ctypes.POINTER(ctypes.c_int8)
_I64_PTR = ctypes.POINTER(ctypes.c_int64)
_U32_PTR = ctypes.POINTER(ctypes.c_uint32)
_U64_PTR = ctypes.POINTER(ctypes.c_uint64)
_SIZE_PTR = ctypes.POINTER(ctypes.c_size_t)


class _NativePCS:
    def __init__(self, path: Path) -> None:
        try:
            self.lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise PCSUnavailableError(
                f"failed to load PCS native library {path}: {exc}"
            ) from exc
        self.path = path
        self.lib.verathos_pcs_v2_abi_version.argtypes = []
        self.lib.verathos_pcs_v2_abi_version.restype = ctypes.c_uint32
        abi_version = int(self.lib.verathos_pcs_v2_abi_version())
        if abi_version != ABI_VERSION:
            raise PCSUnavailableError(
                f"PCS native ABI mismatch: expected {ABI_VERSION}, got {abi_version}"
            )
        self._configure()
        native_cap = int(self.lib.verathos_pcs_v2_max_vector_len())
        if native_cap != MAX_VECTOR_LEN:
            raise PCSUnavailableError(
                f"PCS native vector cap mismatch: expected {MAX_VECTOR_LEN}, got {native_cap}"
            )
        native_combine_cap = int(self.lib.verathos_pcs_v2_max_combine_terms())
        if native_combine_cap != MAX_COMBINE_TERMS:
            raise PCSUnavailableError(
                "PCS native combination cap mismatch: "
                f"expected {MAX_COMBINE_TERMS}, got {native_combine_cap}"
            )
        native_validate_cap = int(self.lib.verathos_pcs_v2_max_validate_commitments())
        if native_validate_cap != MAX_VALIDATE_COMMITMENTS:
            raise PCSUnavailableError(
                "PCS native commitment-validation cap mismatch: "
                f"expected {MAX_VALIDATE_COMMITMENTS}, got {native_validate_cap}"
            )
        native_batch_cap = int(self.lib.verathos_pcs_v2_max_batch_commitments())
        if native_batch_cap != MAX_BATCH_COMMITMENTS:
            raise PCSUnavailableError(
                "PCS native batch commitment cap mismatch: "
                f"expected {MAX_BATCH_COMMITMENTS}, got {native_batch_cap}"
            )
        native_batch_bytes = int(self.lib.verathos_pcs_v2_max_batch_i8_bytes())
        if native_batch_bytes != MAX_BATCH_I8_BYTES:
            raise PCSUnavailableError(
                "PCS native batch byte cap mismatch: "
                f"expected {MAX_BATCH_I8_BYTES}, got {native_batch_bytes}"
            )

    def _configure(self) -> None:
        lib = self.lib
        lib.verathos_pcs_v2_abi_version.argtypes = []
        lib.verathos_pcs_v2_abi_version.restype = ctypes.c_uint32
        lib.verathos_pcs_v2_max_vector_len.argtypes = []
        lib.verathos_pcs_v2_max_vector_len.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_max_combine_terms.argtypes = []
        lib.verathos_pcs_v2_max_combine_terms.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_max_validate_commitments.argtypes = []
        lib.verathos_pcs_v2_max_validate_commitments.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_max_batch_commitments.argtypes = []
        lib.verathos_pcs_v2_max_batch_commitments.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_max_batch_i8_bytes.argtypes = []
        lib.verathos_pcs_v2_max_batch_i8_bytes.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_proof_size.argtypes = [ctypes.c_size_t]
        lib.verathos_pcs_v2_proof_size.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_gemm_batch_proof_size.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v2_gemm_batch_proof_size.restype = ctypes.c_size_t
        lib.verathos_pcs_v2_last_error.argtypes = [_U8_PTR, ctypes.c_size_t]
        lib.verathos_pcs_v2_last_error.restype = ctypes.c_size_t

        for suffix, value_pointer in (
            ("i8", _I8_PTR),
            ("i64", _I64_PTR),
            ("field", _U8_PTR),
        ):
            commit = getattr(lib, f"verathos_pcs_v2_commit_{suffix}")
            commit.argtypes = [value_pointer, ctypes.c_size_t, _U8_PTR]
            commit.restype = ctypes.c_int32

            evaluate = getattr(lib, f"verathos_pcs_v2_evaluate_{suffix}")
            evaluate.argtypes = [
                value_pointer,
                ctypes.c_size_t,
                _U8_PTR,
                ctypes.c_size_t,
                _U8_PTR,
            ]
            evaluate.restype = ctypes.c_int32

            prove = getattr(lib, f"verathos_pcs_v2_prove_{suffix}")
            prove.argtypes = [
                value_pointer,
                ctypes.c_size_t,
                _U8_PTR,
                ctypes.c_size_t,
                _U8_PTR,
                _U8_PTR,
                _U8_PTR,
                _U8_PTR,
                ctypes.c_size_t,
                _SIZE_PTR,
            ]
            prove.restype = ctypes.c_int32
            if suffix != "field":
                prove_linear = getattr(
                    lib,
                    f"verathos_pcs_v3_prove_linear_{suffix}",
                )
                prove_linear.argtypes = [
                    value_pointer,
                    ctypes.c_size_t,
                    _U8_PTR,
                    ctypes.c_size_t,
                    _U8_PTR,
                    _U8_PTR,
                    _U8_PTR,
                    _U8_PTR,
                    ctypes.c_size_t,
                    _SIZE_PTR,
                ]
                prove_linear.restype = ctypes.c_int32

        lib.verathos_pcs_v2_commit_i8_batch.argtypes = [
            _I8_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v2_commit_i8_batch.restype = ctypes.c_int32

        lib.verathos_pcs_v2_evaluate_i8_matrix.argtypes = [
            _I8_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
        ]
        lib.verathos_pcs_v2_evaluate_i8_matrix.restype = ctypes.c_int32

        lib.verathos_pcs_v2_gemm_batch_prove_i8.argtypes = [
            _U8_PTR,
            _I8_PTR,
            ctypes.c_size_t,
            _I8_PTR,
            ctypes.c_size_t,
            _I64_PTR,
            ctypes.c_size_t,
            _U32_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _SIZE_PTR,
        ]
        lib.verathos_pcs_v2_gemm_batch_prove_i8.restype = ctypes.c_int32

        lib.verathos_pcs_v2_hash_fp16_blocks.argtypes = [
            _U8_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v2_hash_fp16_blocks.restype = ctypes.c_int32

        lib.verathos_pcs_v2_build_merkle_levels.argtypes = [
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
            _U64_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _SIZE_PTR,
        ]
        lib.verathos_pcs_v2_build_merkle_levels.restype = ctypes.c_int32

        lib.verathos_pcs_v2_prove_i8_linear_combination.argtypes = [
            _I8_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
            _SIZE_PTR,
        ]
        lib.verathos_pcs_v2_prove_i8_linear_combination.restype = ctypes.c_int32
        lib.verathos_pcs_v3_prove_i64_linear_combination_linear.argtypes = [
            _I64_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
            _SIZE_PTR,
        ]
        (
            lib.verathos_pcs_v3_prove_i64_linear_combination_linear.restype
        ) = ctypes.c_int32

        lib.verathos_pcs_v2_combine_commitments.argtypes = [
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
        ]
        lib.verathos_pcs_v2_combine_commitments.restype = ctypes.c_int32

        lib.verathos_pcs_v2_validate_commitments.argtypes = [
            _U8_PTR,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v2_validate_commitments.restype = ctypes.c_int32

        lib.verathos_pcs_v2_verify.argtypes = [
            _U8_PTR,
            ctypes.c_size_t,
            ctypes.c_uint32,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v2_verify.restype = ctypes.c_int32
        lib.verathos_pcs_v3_verify_linear.argtypes = [
            _U8_PTR,
            ctypes.c_size_t,
            ctypes.c_uint32,
            _U8_PTR,
            ctypes.c_size_t,
            _U8_PTR,
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
        ]
        lib.verathos_pcs_v3_verify_linear.restype = ctypes.c_int32
        lib.verathos_pcs_v3_fold_u31_linear_coefficients.argtypes = [
            _U8_PTR,
            _U8_PTR,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            _U8_PTR,
        ]
        lib.verathos_pcs_v3_fold_u31_linear_coefficients.restype = (
            ctypes.c_int32
        )

    def last_error(self) -> str:
        reported = int(self.lib.verathos_pcs_v2_last_error(None, 0))
        capacity = min(reported + 1, 4096)
        if capacity <= 1:
            return "native PCS operation failed"
        output = (_U8 * capacity)()
        self.lib.verathos_pcs_v2_last_error(output, capacity)
        return bytes(output).split(b"\0", 1)[0].decode("utf-8", errors="replace")


_NATIVE: _NativePCS | None = None
_NATIVE_LOCK = Lock()


LIBRARY_NAME = "libverathos_pcs_v2.dylib" if sys.platform == "darwin" else "libverathos_pcs_v2.so"
LIBRARY_PATH = Path(__file__).resolve().parents[1] / LIBRARY_NAME


def _native() -> _NativePCS:
    global _NATIVE
    if _NATIVE is not None:
        return _NATIVE
    with _NATIVE_LOCK:
        if _NATIVE is None:
            if not LIBRARY_PATH.is_file():
                raise PCSUnavailableError(f"{LIBRARY_PATH} not found; run verathos-build-native")
            _NATIVE = _NativePCS(LIBRARY_PATH)
        return _NATIVE


def native_library_path() -> Path:
    """Return the loaded native library path, loading and validating it first."""

    return _native().path


def _encoding_id(encoding: str | int) -> int:
    if isinstance(encoding, str):
        normalized = encoding.strip().lower().replace("-", "_")
        aliases = {
            "i8": ENCODING_SIGNED_I8,
            "int8": ENCODING_SIGNED_I8,
            "signed_i8": ENCODING_SIGNED_I8,
            "i64": ENCODING_SIGNED_I64,
            "int64": ENCODING_SIGNED_I64,
            "signed_i64": ENCODING_SIGNED_I64,
            "field": ENCODING_PALLAS_SCALAR,
            "fr": ENCODING_PALLAS_SCALAR,
            "pallas": ENCODING_PALLAS_SCALAR,
            "pallas_scalar": ENCODING_PALLAS_SCALAR,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise PCSFormatError(f"unsupported PCS encoding: {encoding!r}") from exc
    if isinstance(encoding, bool):
        raise PCSFormatError("PCS encoding must not be boolean")
    try:
        encoding_id = operator.index(encoding)
    except TypeError as exc:
        raise PCSFormatError(
            "PCS encoding must be 'i8', 'i64', 'field', 1, 2, or 3"
        ) from exc
    if encoding_id not in (
        ENCODING_SIGNED_I8,
        ENCODING_SIGNED_I64,
        ENCODING_PALLAS_SCALAR,
    ):
        raise PCSFormatError(f"unsupported PCS encoding id: {encoding_id}")
    return int(encoding_id)


def _checked_vector_length(vector_length: int) -> tuple[int, int, int]:
    if isinstance(vector_length, bool):
        raise PCSFormatError("vector length must not be boolean")
    try:
        length = operator.index(vector_length)
    except TypeError as exc:
        raise PCSFormatError("vector length must be an integer") from exc
    if length < 1 or length > MAX_VECTOR_LEN:
        raise PCSFormatError(f"vector length must be in 1..={MAX_VECTOR_LEN}")
    padded = 1 << (length - 1).bit_length()
    return int(length), padded, padded.bit_length() - 1


def encode_scalar(value: int | bytes | bytearray | memoryview) -> bytes:
    """Encode or validate one canonical little-endian Pallas scalar."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        if len(encoded) != SCALAR_BYTES:
            raise PCSFormatError("Pallas scalar must be exactly 32 bytes")
        integer = int.from_bytes(encoded, "little")
    else:
        if isinstance(value, bool):
            raise PCSFormatError("Pallas scalar must not be boolean")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise PCSFormatError(
                "Pallas scalar must be an integer or 32 bytes"
            ) from exc
    if integer < 0 or integer >= PALLAS_SCALAR_MODULUS:
        raise PCSFormatError("Pallas scalar is outside the canonical field range")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return encoded
    return int(integer).to_bytes(SCALAR_BYTES, "little", signed=False)


def scalar_to_int(value: bytes | bytearray | memoryview) -> int:
    """Decode one canonical little-endian Pallas scalar."""

    return int.from_bytes(encode_scalar(value), "little")


def _fixed_bytes(value: object, length: int, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise PCSFormatError(f"{label} must be bytes")
    encoded = bytes(value)
    if len(encoded) != length:
        raise PCSFormatError(f"{label} must be exactly {length} bytes")
    return encoded


def _point_bytes(
    point: Iterable[int | bytes], expected_count: int
) -> tuple[bytes, int]:
    if isinstance(point, (bytes, bytearray, memoryview)):
        raw = bytes(point)
        if len(raw) != expected_count * SCALAR_BYTES:
            raise PCSFormatError(
                f"MLE point must contain exactly {expected_count} scalars"
            )
        encoded = b"".join(
            encode_scalar(raw[offset : offset + SCALAR_BYTES])
            for offset in range(0, len(raw), SCALAR_BYTES)
        )
    else:
        try:
            coordinates = tuple(islice(iter(point), expected_count + 1))
        except TypeError as exc:
            raise PCSFormatError(
                "MLE point must be an iterable of Pallas scalars"
            ) from exc
        if len(coordinates) != expected_count:
            raise PCSFormatError(
                f"MLE point must contain exactly {expected_count} scalars"
            )
        encoded = b"".join(encode_scalar(coordinate) for coordinate in coordinates)
    return encoded, expected_count


def _byte_array(value: bytes) -> ctypes.Array[_U8]:
    return (_U8 * len(value)).from_buffer_copy(value)


def _point_pointer(value: bytes) -> ctypes.Array[_U8] | None:
    return _byte_array(value) if value else None


def _values_buffer(values: object, encoding: int) -> tuple[ctypes.Array, int, str]:
    if encoding == ENCODING_PALLAS_SCALAR:
        if isinstance(values, (bytes, bytearray, memoryview)):
            raw = bytes(values)
            if len(raw) == 0 or len(raw) % SCALAR_BYTES != 0:
                raise PCSFormatError(
                    "packed Pallas field vector must contain one or more 32-byte scalars"
                )
            length, _, _ = _checked_vector_length(len(raw) // SCALAR_BYTES)
            canonical = b"".join(
                encode_scalar(raw[offset : offset + SCALAR_BYTES])
                for offset in range(0, len(raw), SCALAR_BYTES)
            )
        else:
            if isinstance(values, str):
                raise PCSFormatError(
                    "Pallas field vectors require an iterable of scalars"
                )
            try:
                items = tuple(islice(iter(values), MAX_VECTOR_LEN + 1))  # type: ignore[arg-type]
            except TypeError as exc:
                raise PCSFormatError(
                    "Pallas field vectors require an iterable of scalars"
                ) from exc
            length, _, _ = _checked_vector_length(len(items))
            canonical = b"".join(encode_scalar(item) for item in items)
        return _byte_array(canonical), length, "field"

    if encoding == ENCODING_SIGNED_I8 and isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        raw = bytes(values)
        length, _, _ = _checked_vector_length(len(raw))
        return (ctypes.c_int8 * length).from_buffer_copy(raw), length, "i8"

    if isinstance(values, (str, bytes, bytearray, memoryview)):
        raise PCSFormatError("signed i64 vectors require an iterable of integers")
    try:
        items = tuple(islice(iter(values), MAX_VECTOR_LEN + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise PCSFormatError("PCS values must be an iterable of integers") from exc
    length, _, _ = _checked_vector_length(len(items))
    minimum, maximum, c_type, suffix = (
        (-128, 127, ctypes.c_int8, "i8")
        if encoding == ENCODING_SIGNED_I8
        else (-(1 << 63), (1 << 63) - 1, ctypes.c_int64, "i64")
    )
    normalized: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool):
            raise PCSFormatError(f"PCS value {index} must not be boolean")
        try:
            integer = operator.index(item)
        except TypeError as exc:
            raise PCSFormatError(f"PCS value {index} is not an integer") from exc
        if integer < minimum or integer > maximum:
            raise PCSFormatError(
                f"PCS value {index} is outside the signed {8 if encoding == 1 else 64}-bit range"
            )
        normalized.append(int(integer))
    return (c_type * length)(*normalized), length, suffix


def _raise_status(native: _NativePCS, status: int) -> None:
    message = native.last_error()
    if status in (-1, -2, -3, -4, -6):
        raise PCSFormatError(message)
    raise PCSNativeError(message)


def _checked_combine_count(count: int) -> int:
    if count < 1 or count > MAX_COMBINE_TERMS:
        raise PCSFormatError(
            f"commitment term count must be in 1..={MAX_COMBINE_TERMS}"
        )
    return count


def _commitment_sequence_bytes(commitments: object) -> tuple[bytes, int]:
    if isinstance(commitments, (bytes, bytearray, memoryview)):
        raw = bytes(commitments)
        if len(raw) == 0 or len(raw) % COMMITMENT_BYTES != 0:
            raise PCSFormatError(
                "packed commitments must contain one or more 32-byte commitments"
            )
        count = _checked_combine_count(len(raw) // COMMITMENT_BYTES)
        return raw, count
    if isinstance(commitments, str):
        raise PCSFormatError("commitments must be an iterable of 32-byte commitments")
    try:
        items = tuple(islice(iter(commitments), MAX_COMBINE_TERMS + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise PCSFormatError(
            "commitments must be an iterable of 32-byte commitments"
        ) from exc
    count = _checked_combine_count(len(items))
    return (
        b"".join(
            _fixed_bytes(item, COMMITMENT_BYTES, f"commitment {index}")
            for index, item in enumerate(items)
        ),
        count,
    )


def _coefficient_sequence_bytes(coefficients: object) -> tuple[bytes, int]:
    if isinstance(coefficients, (bytes, bytearray, memoryview)):
        raw = bytes(coefficients)
        if len(raw) == 0 or len(raw) % SCALAR_BYTES != 0:
            raise PCSFormatError(
                "packed coefficients must contain one or more 32-byte scalars"
            )
        count = _checked_combine_count(len(raw) // SCALAR_BYTES)
        return (
            b"".join(
                encode_scalar(raw[offset : offset + SCALAR_BYTES])
                for offset in range(0, len(raw), SCALAR_BYTES)
            ),
            count,
        )
    if isinstance(coefficients, str):
        raise PCSFormatError("coefficients must be an iterable of Pallas scalars")
    try:
        items = tuple(islice(iter(coefficients), MAX_COMBINE_TERMS + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise PCSFormatError(
            "coefficients must be an iterable of Pallas scalars"
        ) from exc
    count = _checked_combine_count(len(items))
    return b"".join(encode_scalar(item) for item in items), count


def combine_commitments(commitments: object, coefficients: object) -> bytes:
    """Return the canonical weighted sum of up to 256 Pallas commitments.

    Each commitment is a canonical 32-byte compressed Pallas point. Each
    coefficient is an integer or canonical 32-byte little-endian Pallas scalar.
    Either argument may also be supplied as one packed bytes buffer.
    """

    commitment_raw, commitment_count = _commitment_sequence_bytes(commitments)
    coefficient_raw, coefficient_count = _coefficient_sequence_bytes(coefficients)
    if commitment_count != coefficient_count:
        raise PCSFormatError("commitment and coefficient counts must match")

    native = _native()
    commitment_buffer = _byte_array(commitment_raw)
    coefficient_buffer = _byte_array(coefficient_raw)
    output = (_U8 * COMMITMENT_BYTES)()
    status = int(
        native.lib.verathos_pcs_v2_combine_commitments(
            commitment_buffer,
            coefficient_buffer,
            commitment_count,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def register_catalog_commitments(commitments: object) -> tuple[bytes, int]:
    """Validate and cache one static catalog commitment set in native memory."""

    if isinstance(commitments, (bytes, bytearray, memoryview)):
        commitment_raw = bytes(commitments)
        if not commitment_raw or len(commitment_raw) % COMMITMENT_BYTES:
            raise PCSFormatError(
                "packed catalog commitments must contain 32-byte commitments"
            )
        count = len(commitment_raw) // COMMITMENT_BYTES
    else:
        if isinstance(commitments, str):
            raise PCSFormatError("catalog commitments must be an iterable")
        try:
            items = tuple(
                islice(iter(commitments), MAX_CATALOG_COMBINE_TERMS + 1)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise PCSFormatError(
                "catalog commitments must be an iterable"
            ) from exc
        count = len(items)
        commitment_raw = b"".join(
            _fixed_bytes(item, COMMITMENT_BYTES, f"commitment {index}")
            for index, item in enumerate(items)
        )
    if count < 1 or count > MAX_CATALOG_COMBINE_TERMS:
        raise PCSFormatError("catalog commitment count is out of range")

    native = _native()
    try:
        function = native.lib.verathos_pcs_v2_register_catalog_commitments
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks registered catalog commitments"
        ) from exc
    function.argtypes = [_U8_PTR, ctypes.c_size_t, _U8_PTR]
    function.restype = ctypes.c_int32
    commitment_buffer = _byte_array(commitment_raw)
    catalog_id = (_U8 * 32)()
    status = int(function(commitment_buffer, count, catalog_id))
    if status != 0:
        _raise_status(native, status)
    return bytes(catalog_id), count


def combine_registered_catalog_u63(
    catalog_id: bytes,
    coefficients: object,
    *,
    term_count: int,
) -> bytes:
    """Fold a registered static catalog using unsigned u63 coefficients."""

    catalog_id = _fixed_bytes(catalog_id, 32, "catalog ID")
    if (
        isinstance(term_count, bool)
        or not isinstance(term_count, int)
        or not 0 < term_count <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("registered catalog term count is out of range")
    if isinstance(coefficients, (bytes, bytearray, memoryview)):
        coefficient_raw = bytes(coefficients)
        if len(coefficient_raw) != term_count * 8:
            raise PCSFormatError(
                "registered catalog coefficient count does not match"
            )
        if any(
            int.from_bytes(coefficient_raw[offset : offset + 8], "little")
            >= 1 << 63
            for offset in range(0, len(coefficient_raw), 8)
        ):
            raise PCSFormatError(
                "catalog coefficients must be unsigned 63-bit integers"
            )
    else:
        if isinstance(coefficients, str):
            raise PCSFormatError("catalog coefficients must be an iterable")
        try:
            items = tuple(
                islice(iter(coefficients), MAX_CATALOG_COMBINE_TERMS + 1)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise PCSFormatError(
                "catalog coefficients must be an iterable"
            ) from exc
        if len(items) != term_count:
            raise PCSFormatError(
                "registered catalog coefficient count does not match"
            )
        packed = []
        for index, value in enumerate(items):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 63
            ):
                raise PCSFormatError(
                    f"catalog coefficient {index} must be an unsigned "
                    "63-bit integer"
                )
            packed.append(value.to_bytes(8, "little"))
        coefficient_raw = b"".join(packed)

    native = _native()
    try:
        function = native.lib.verathos_pcs_v2_combine_registered_catalog_u63
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks registered u63 catalog folding"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
    ]
    function.restype = ctypes.c_int32
    catalog_id_buffer = _byte_array(catalog_id)
    coefficient_buffer = _byte_array(coefficient_raw)
    output = (_U8 * COMMITMENT_BYTES)()
    status = int(
        function(
            catalog_id_buffer,
            coefficient_buffer,
            term_count,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def combine_registered_catalog_u7_batch(
    catalog_id: bytes,
    coefficients: bytes,
    *,
    term_count: int,
    fold_count: int,
) -> tuple[bytes, ...]:
    """Fold one registered catalog by multiple packed u7 coefficient rows."""

    catalog_id = _fixed_bytes(catalog_id, 32, "catalog ID")
    if (
        isinstance(term_count, bool)
        or not isinstance(term_count, int)
        or not 0 < term_count <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("registered catalog term count is out of range")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or not 0 < fold_count <= MAX_CATALOG_FOLDS
    ):
        raise PCSFormatError("registered catalog fold count is out of range")
    if not isinstance(coefficients, bytes):
        raise PCSFormatError("catalog u7 coefficients must be packed bytes")
    if len(coefficients) != term_count * fold_count:
        raise PCSFormatError(
            "registered catalog coefficient count does not match"
        )
    if any(value >= 1 << 7 for value in coefficients):
        raise PCSFormatError(
            "catalog coefficients must be unsigned 7-bit integers"
        )
    native = _native()
    try:
        function = (
            native.lib.verathos_pcs_v2_combine_registered_catalog_u7_batch
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks registered u7 catalog batch folding"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
    ]
    function.restype = ctypes.c_int32
    catalog_id_buffer = _byte_array(catalog_id)
    coefficient_buffer = _byte_array(coefficients)
    output = (_U8 * (fold_count * COMMITMENT_BYTES))()
    status = int(
        function(
            catalog_id_buffer,
            coefficient_buffer,
            term_count,
            fold_count,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    raw = bytes(output)
    return tuple(
        raw[offset : offset + COMMITMENT_BYTES]
        for offset in range(0, len(raw), COMMITMENT_BYTES)
    )


def combine_registered_catalog_u31_batch(
    catalog_id: bytes,
    coefficients: bytes,
    *,
    term_count: int,
    fold_count: int,
) -> tuple[bytes, ...]:
    """Fold one registered catalog by packed little-endian u31 rows."""

    catalog_id = _fixed_bytes(catalog_id, 32, "catalog ID")
    if (
        isinstance(term_count, bool)
        or not isinstance(term_count, int)
        or not 0 < term_count <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("registered catalog term count is out of range")
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or not 0 < fold_count <= MAX_CATALOG_FOLDS
    ):
        raise PCSFormatError("registered catalog fold count is out of range")
    if not isinstance(coefficients, bytes):
        raise PCSFormatError("catalog u31 coefficients must be packed bytes")
    if len(coefficients) != term_count * fold_count * 4:
        raise PCSFormatError(
            "registered catalog coefficient count does not match"
        )
    # The native parser checks every decoded word is canonical u31 before
    # curve arithmetic. Avoid duplicating that multi-million-word scan in
    # Python on the validator hot path.
    native = _native()
    try:
        function = (
            native.lib.verathos_pcs_v2_combine_registered_catalog_u31_batch
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks registered u31 catalog batch folding"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
    ]
    function.restype = ctypes.c_int32
    catalog_id_buffer = _byte_array(catalog_id)
    coefficient_buffer = _byte_array(coefficients)
    output = (_U8 * (fold_count * COMMITMENT_BYTES))()
    status = int(
        function(
            catalog_id_buffer,
            coefficient_buffer,
            term_count,
            fold_count,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    raw = bytes(output)
    return tuple(
        raw[offset : offset + COMMITMENT_BYTES]
        for offset in range(0, len(raw), COMMITMENT_BYTES)
    )


def fold_u31_linear_coefficients(
    coefficients: bytes,
    betas: object,
    *,
    claim_count: int,
    term_count: int,
    fold_count: int,
    padded_length: int,
) -> bytes:
    """Fold claim-major packed u31 rows into packed Pallas scalars.

    The operation is verifier-side public arithmetic.  Keeping it native
    avoids materializing and reducing millions of Python big integers before
    an arbitrary-linear IPA check.
    """

    for value, name, limit in (
        (claim_count, "claim count", 512),
        (term_count, "term count", MAX_CATALOG_COMBINE_TERMS),
        (fold_count, "fold count", MAX_CATALOG_FOLDS),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= limit
        ):
            raise PCSFormatError(f"linear-fold {name} is out of range")
    active_length = claim_count * term_count
    if (
        isinstance(padded_length, bool)
        or not isinstance(padded_length, int)
        or padded_length < active_length
        or padded_length > MAX_VECTOR_LEN
        or padded_length & (padded_length - 1)
    ):
        raise PCSFormatError("linear-fold padded length is out of range")
    if not isinstance(coefficients, bytes) or len(coefficients) != (
        claim_count * fold_count * term_count * 4
    ):
        raise PCSFormatError(
            "linear-fold coefficient buffer has the wrong length"
        )
    expected_beta_count = claim_count * fold_count
    if isinstance(betas, (bytes, bytearray, memoryview)):
        beta_raw = bytes(betas)
        if len(beta_raw) != expected_beta_count * SCALAR_BYTES:
            raise PCSFormatError("linear-fold beta count is wrong")
    elif isinstance(betas, str):
        raise PCSFormatError("linear-fold betas must be packed or iterable")
    else:
        try:
            beta_values = tuple(
                islice(iter(betas), expected_beta_count + 1)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise PCSFormatError(
                "linear-fold betas must be packed or iterable"
            ) from exc
        if len(beta_values) != expected_beta_count:
            raise PCSFormatError("linear-fold beta count is wrong")
        beta_raw = b"".join(encode_scalar(value) for value in beta_values)
    native = _native()
    output = (_U8 * (padded_length * SCALAR_BYTES))()
    status = int(
        native.lib.verathos_pcs_v3_fold_u31_linear_coefficients(
            _byte_array(coefficients),
            _byte_array(beta_raw),
            claim_count,
            term_count,
            fold_count,
            padded_length,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def derive_registered_lean_projection_claim_materials(
    *,
    statement_digest: bytes,
    validator_nonce: bytes,
    catalog_id: bytes,
    input_rows_i8: bytes,
    surrogate_rows_i64_le: bytes,
    claim_count: int,
    input_dim: int,
    output_dim: int,
) -> tuple[tuple[tuple[int, ...], tuple[bytes, ...]], ...]:
    """Derive verifier claim targets and catalog folds in native code.

    The returned order is claim-major and fold-major. Canonical u31
    coefficient parsing and every curve operation remain fail-closed inside
    the existing PCS library; coefficients never cross back into Python.
    """

    statement_digest = _fixed_bytes(
        statement_digest, 32, "lean projection statement digest"
    )
    validator_nonce = _fixed_bytes(
        validator_nonce, 32, "lean projection validator nonce"
    )
    catalog_id = _fixed_bytes(catalog_id, 32, "catalog ID")
    if not any(validator_nonce):
        raise PCSFormatError("lean projection validator nonce must be nonzero")
    if (
        isinstance(claim_count, bool)
        or not isinstance(claim_count, int)
        or not 0 < claim_count <= 512
        or isinstance(input_dim, bool)
        or not isinstance(input_dim, int)
        or not 0 < input_dim <= MAX_VECTOR_LEN
        or isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or not 0 < output_dim <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("lean projection material geometry is invalid")
    if not isinstance(input_rows_i8, bytes):
        raise PCSFormatError("lean projection X rows must be packed bytes")
    if not isinstance(surrogate_rows_i64_le, bytes):
        raise PCSFormatError(
            "lean projection surrogate rows must be packed bytes"
        )
    if len(input_rows_i8) != claim_count * input_dim:
        raise PCSFormatError("lean projection X row buffer has the wrong length")
    if len(surrogate_rows_i64_le) != claim_count * output_dim * 8:
        raise PCSFormatError(
            "lean projection surrogate row buffer has the wrong length"
        )

    native = _native()
    try:
        function = (
            native.lib.verathos_pcs_v2_lean_projection_claim_materials
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks lean projection material derivation"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int32
    statement_buffer = _byte_array(statement_digest)
    nonce_buffer = _byte_array(validator_nonce)
    catalog_buffer = _byte_array(catalog_id)
    input_buffer = _byte_array(input_rows_i8)
    surrogate_buffer = _byte_array(surrogate_rows_i64_le)
    output_len = claim_count * 8 * SCALAR_BYTES
    output = (_U8 * output_len)()
    status = int(
        function(
            statement_buffer,
            nonce_buffer,
            catalog_buffer,
            input_buffer,
            len(input_rows_i8),
            surrogate_buffer,
            len(surrogate_rows_i64_le),
            claim_count,
            input_dim,
            output_dim,
            output,
            output_len,
        )
    )
    if status != 0:
        _raise_status(native, status)
    raw = bytes(output)
    result = []
    cursor = 0
    for _claim_index in range(claim_count):
        targets = tuple(
            scalar_to_int(
                raw[
                    cursor + fold * SCALAR_BYTES
                    : cursor + (fold + 1) * SCALAR_BYTES
                ]
            )
            for fold in range(4)
        )
        cursor += 4 * SCALAR_BYTES
        commitments = tuple(
            raw[
                cursor + fold * COMMITMENT_BYTES
                : cursor + (fold + 1) * COMMITMENT_BYTES
            ]
            for fold in range(4)
        )
        cursor += 4 * COMMITMENT_BYTES
        result.append((targets, commitments))
    if cursor != len(raw):
        raise PCSNativeError(
            "native lean projection material output has trailing bytes"
        )
    return tuple(result)


def derive_lean_projection_claim_targets(
    *,
    statement_digest: bytes,
    validator_nonce: bytes,
    input_rows_i8: bytes,
    surrogate_rows_i64_le: bytes,
    claim_count: int,
    input_dim: int,
    output_dim: int,
) -> tuple[tuple[int, ...], ...]:
    """Derive four transcript-bound exact targets for each projection claim."""

    statement_digest = _fixed_bytes(
        statement_digest, 32, "lean projection statement digest"
    )
    validator_nonce = _fixed_bytes(
        validator_nonce, 32, "lean projection validator nonce"
    )
    if not any(validator_nonce):
        raise PCSFormatError("lean projection validator nonce must be nonzero")
    if (
        isinstance(claim_count, bool)
        or not isinstance(claim_count, int)
        or not 0 < claim_count <= 512
        or isinstance(input_dim, bool)
        or not isinstance(input_dim, int)
        or not 0 < input_dim <= MAX_VECTOR_LEN
        or isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or not 0 < output_dim <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("lean projection target geometry is invalid")
    if (
        not isinstance(input_rows_i8, bytes)
        or len(input_rows_i8) != claim_count * input_dim
    ):
        raise PCSFormatError("lean projection X row buffer has the wrong length")
    if (
        not isinstance(surrogate_rows_i64_le, bytes)
        or len(surrogate_rows_i64_le) != claim_count * output_dim * 8
    ):
        raise PCSFormatError(
            "lean projection surrogate row buffer has the wrong length"
        )

    native = _native()
    try:
        function = native.lib.verathos_pcs_v2_lean_projection_claim_targets
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks lean projection target derivation"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int32
    statement_buffer = _byte_array(statement_digest)
    nonce_buffer = _byte_array(validator_nonce)
    input_buffer = _byte_array(input_rows_i8)
    surrogate_buffer = _byte_array(surrogate_rows_i64_le)
    output_len = claim_count * 4 * SCALAR_BYTES
    output = (_U8 * output_len)()
    status = int(
        function(
            statement_buffer,
            nonce_buffer,
            input_buffer,
            len(input_rows_i8),
            surrogate_buffer,
            len(surrogate_rows_i64_le),
            claim_count,
            input_dim,
            output_dim,
            output,
            output_len,
        )
    )
    if status != 0:
        _raise_status(native, status)
    raw = bytes(output)
    return tuple(
        tuple(
            scalar_to_int(
                raw[
                    (claim_index * 4 + fold_index) * SCALAR_BYTES
                    : (claim_index * 4 + fold_index + 1) * SCALAR_BYTES
                ]
            )
            for fold_index in range(4)
        )
        for claim_index in range(claim_count)
    )


def combine_registered_lean_projection_aggregate(
    *,
    statement_digest: bytes,
    validator_nonce: bytes,
    catalog_id: bytes,
    input_rows_i8: bytes,
    surrogate_rows_i64_le: bytes,
    beta_bytes: bytes,
    claim_count: int,
    input_dim: int,
    output_dim: int,
) -> bytes:
    """Commit one beta-weighted aggregate of registered projection folds."""

    statement_digest = _fixed_bytes(
        statement_digest, 32, "lean projection statement digest"
    )
    validator_nonce = _fixed_bytes(
        validator_nonce, 32, "lean projection validator nonce"
    )
    catalog_id = _fixed_bytes(catalog_id, 32, "catalog ID")
    if not any(validator_nonce):
        raise PCSFormatError("lean projection validator nonce must be nonzero")
    if (
        isinstance(claim_count, bool)
        or not isinstance(claim_count, int)
        or not 0 < claim_count <= 512
        or isinstance(input_dim, bool)
        or not isinstance(input_dim, int)
        or not 0 < input_dim <= MAX_VECTOR_LEN
        or isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or not 0 < output_dim <= MAX_CATALOG_COMBINE_TERMS
    ):
        raise PCSFormatError("lean projection aggregate geometry is invalid")
    if (
        not isinstance(input_rows_i8, bytes)
        or len(input_rows_i8) != claim_count * input_dim
    ):
        raise PCSFormatError("lean projection X row buffer has the wrong length")
    if (
        not isinstance(surrogate_rows_i64_le, bytes)
        or len(surrogate_rows_i64_le) != claim_count * output_dim * 8
    ):
        raise PCSFormatError(
            "lean projection surrogate row buffer has the wrong length"
        )
    if (
        not isinstance(beta_bytes, bytes)
        or len(beta_bytes) != claim_count * 4 * SCALAR_BYTES
    ):
        raise PCSFormatError(
            "lean projection aggregate beta buffer has the wrong length"
        )
    for offset in range(0, len(beta_bytes), SCALAR_BYTES):
        scalar_to_int(beta_bytes[offset : offset + SCALAR_BYTES])

    native = _native()
    try:
        function = (
            native.lib
            .verathos_pcs_v2_lean_projection_aggregate_commitment
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks lean projection aggregate commitment"
        ) from exc
    function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
    ]
    function.restype = ctypes.c_int32
    statement_buffer = _byte_array(statement_digest)
    nonce_buffer = _byte_array(validator_nonce)
    catalog_buffer = _byte_array(catalog_id)
    input_buffer = _byte_array(input_rows_i8)
    surrogate_buffer = _byte_array(surrogate_rows_i64_le)
    beta_buffer = _byte_array(beta_bytes)
    output = (_U8 * COMMITMENT_BYTES)()
    status = int(
        function(
            statement_buffer,
            nonce_buffer,
            catalog_buffer,
            input_buffer,
            len(input_rows_i8),
            surrogate_buffer,
            len(surrogate_rows_i64_le),
            beta_buffer,
            len(beta_bytes),
            claim_count,
            input_dim,
            output_dim,
            output,
            COMMITMENT_BYTES,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def validate_commitments(commitments: object) -> None:
    """Validate up to 2^20 canonical compressed Pallas commitments in one call."""

    if isinstance(commitments, (bytes, bytearray, memoryview)):
        raw = bytes(commitments)
        if len(raw) == 0 or len(raw) % COMMITMENT_BYTES != 0:
            raise PCSFormatError(
                "packed commitments must contain one or more 32-byte commitments"
            )
        count = len(raw) // COMMITMENT_BYTES
        if count > MAX_VALIDATE_COMMITMENTS:
            raise PCSFormatError(
                "commitment validation count must be in "
                f"1..={MAX_VALIDATE_COMMITMENTS}"
            )
    else:
        if isinstance(commitments, str):
            raise PCSFormatError(
                "commitments must be an iterable of 32-byte commitments"
            )
        try:
            items = tuple(
                islice(iter(commitments), MAX_VALIDATE_COMMITMENTS + 1)  # type: ignore[arg-type]
            )
        except TypeError as exc:
            raise PCSFormatError(
                "commitments must be an iterable of 32-byte commitments"
            ) from exc
        count = len(items)
        if count < 1 or count > MAX_VALIDATE_COMMITMENTS:
            raise PCSFormatError(
                "commitment validation count must be in "
                f"1..={MAX_VALIDATE_COMMITMENTS}"
            )
        raw = b"".join(
            _fixed_bytes(item, COMMITMENT_BYTES, f"commitment {index}")
            for index, item in enumerate(items)
        )

    native = _native()
    commitment_buffer = _byte_array(raw)
    status = int(
        native.lib.verathos_pcs_v2_validate_commitments(
            commitment_buffer,
            count,
        )
    )
    if status != 0:
        _raise_status(native, status)


def hash_fp16_blocks(
    blocks: bytes | bytearray | memoryview,
    *,
    logical_vector_length: int,
) -> tuple[bytes, ...]:
    """Hash packed equal-length canonical little-endian FP16 blocks."""

    length, _, _ = _checked_vector_length(logical_vector_length)
    if not isinstance(blocks, (bytes, bytearray, memoryview)):
        raise PCSFormatError("packed FP16 blocks must be bytes")
    raw = bytes(blocks)
    block_bytes = length * 2
    if not raw or len(raw) % block_bytes:
        raise PCSFormatError("packed FP16 blocks must contain complete vectors")
    count = len(raw) // block_bytes
    if count > MAX_MERKLE_BUILD_LEAVES:
        raise PCSFormatError(
            f"FP16 block count must be in 1..={MAX_MERKLE_BUILD_LEAVES}"
        )
    native = _native()
    input_buffer = _byte_array(raw)
    output_size = count * COMMITMENT_BYTES
    output = (_U8 * output_size)()
    status = int(
        native.lib.verathos_pcs_v2_hash_fp16_blocks(
            input_buffer,
            len(raw),
            count,
            length,
            output,
            output_size,
        )
    )
    if status != 0:
        _raise_status(native, status)
    encoded = bytes(output)
    return tuple(
        encoded[offset : offset + COMMITMENT_BYTES]
        for offset in range(0, len(encoded), COMMITMENT_BYTES)
    )


def _build_merkle_levels_native(
    context: bytes,
    commitment_bytes: bytes,
    lengths: tuple[int, ...],
    encodings: bytes,
) -> bytes:
    count = len(lengths)
    if len(encodings) != count:
        raise PCSFormatError("Merkle encoding count does not match leaf count")
    if len(commitment_bytes) != count * COMMITMENT_BYTES:
        raise PCSFormatError("Merkle commitment count does not match leaf count")

    padded = 1 << (count - 1).bit_length()
    output_capacity = (2 * padded - 1) * COMMITMENT_BYTES
    context_buffer = _byte_array(context)
    commitment_buffer = _byte_array(commitment_bytes)
    length_buffer = (ctypes.c_uint64 * count)(*lengths)
    encoding_buffer = (_U8 * count)(*encodings)
    output = (_U8 * output_capacity)()
    output_len = ctypes.c_size_t(0)
    native = _native()
    status = int(
        native.lib.verathos_pcs_v2_build_merkle_levels(
            context_buffer,
            commitment_buffer,
            len(commitment_bytes),
            length_buffer,
            count,
            encoding_buffer,
            count,
            output,
            output_capacity,
            ctypes.byref(output_len),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if output_len.value != output_capacity:
        raise PCSNativeError("native Merkle builder returned an invalid length")
    return bytes(output)


def build_merkle_levels(
    context_digest: bytes | bytearray | memoryview,
    commitments: object,
    logical_lengths: object,
    encoding_ids: object,
) -> bytes:
    """Build level-major unwrapped Merkle hashes for canonical leaves."""

    context = _fixed_bytes(context_digest, 32, "Merkle context digest")
    try:
        raw_lengths = tuple(logical_lengths)  # type: ignore[arg-type]
        raw_encoding_values = tuple(encoding_ids)  # type: ignore[arg-type]
        lengths = tuple(operator.index(item) for item in raw_lengths)
        encoding_values = tuple(operator.index(item) for item in raw_encoding_values)
    except TypeError as exc:
        raise PCSFormatError(
            "Merkle logical lengths and encodings must be integer iterables"
        ) from exc
    count = len(lengths)
    if count < 1 or count > MAX_MERKLE_BUILD_LEAVES:
        raise PCSFormatError(
            f"Merkle leaf count must be in 1..={MAX_MERKLE_BUILD_LEAVES}"
        )
    if len(encoding_values) != count:
        raise PCSFormatError("Merkle encoding count does not match leaf count")
    if any(
        isinstance(raw_value, bool) or value < 1 or value > MAX_VECTOR_LEN
        for raw_value, value in zip(raw_lengths, lengths, strict=True)
    ):
        raise PCSFormatError(
            f"Merkle logical lengths must be in 1..={MAX_VECTOR_LEN}"
        )
    if any(
        isinstance(raw_value, bool) or value < 1 or value > 5
        for raw_value, value in zip(raw_encoding_values, encoding_values, strict=True)
    ):
        raise PCSFormatError("Merkle leaf encoding is unsupported")

    if isinstance(commitments, (bytes, bytearray, memoryview)):
        commitment_bytes = bytes(commitments)
    else:
        try:
            items = tuple(commitments)  # type: ignore[arg-type]
        except TypeError as exc:
            raise PCSFormatError(
                "Merkle commitments must be bytes or an iterable"
            ) from exc
        commitment_bytes = b"".join(
            _fixed_bytes(item, COMMITMENT_BYTES, f"Merkle commitment {index}")
            for index, item in enumerate(items)
        )
    return _build_merkle_levels_native(
        context,
        commitment_bytes,
        lengths,
        bytes(encoding_values),
    )


def proof_size(vector_length: int) -> int:
    """Return the exact canonical IPA proof size for a logical vector length."""

    length, _, _ = _checked_vector_length(vector_length)
    size = int(_native().lib.verathos_pcs_v2_proof_size(length))
    if size <= 0:
        raise PCSNativeError("native PCS returned an invalid proof size")
    return size


def commit(values: object, *, encoding: str | int = ENCODING_SIGNED_I8) -> bytes:
    """Commit deterministically to an encoded vector with zero blinding."""

    encoding_id = _encoding_id(encoding)
    value_buffer, length, suffix = _values_buffer(values, encoding_id)
    native = _native()
    output = (_U8 * COMMITMENT_BYTES)()
    status = int(
        getattr(native.lib, f"verathos_pcs_v2_commit_{suffix}")(
            value_buffer, length, output
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def commit_i8_batch(
    values: bytes | bytearray | memoryview,
    *,
    vector_length: int,
) -> tuple[bytes, ...]:
    """Commit to packed equal-length signed-i8 vectors in canonical order."""

    length, _, _ = _checked_vector_length(vector_length)
    if not isinstance(values, (bytes, bytearray, memoryview)):
        raise PCSFormatError("packed signed-i8 batch values must be bytes")
    raw = bytes(values)
    if not raw or len(raw) % length:
        raise PCSFormatError(
            "packed signed-i8 batch values must contain complete vectors"
        )
    count = len(raw) // length
    if count > MAX_BATCH_COMMITMENTS:
        raise PCSFormatError(
            "signed-i8 commitment batch count must be in "
            f"1..={MAX_BATCH_COMMITMENTS}"
        )
    if len(raw) > MAX_BATCH_I8_BYTES:
        raise PCSFormatError(
            f"packed signed-i8 batch exceeds {MAX_BATCH_I8_BYTES} bytes"
        )

    value_buffer = (ctypes.c_int8 * len(raw)).from_buffer_copy(raw)
    output_size = count * COMMITMENT_BYTES
    output = (_U8 * output_size)()
    native = _native()
    status = int(
        native.lib.verathos_pcs_v2_commit_i8_batch(
            value_buffer,
            len(raw),
            length,
            count,
            output,
            output_size,
        )
    )
    if status != 0:
        _raise_status(native, status)
    packed = bytes(output)
    return tuple(
        packed[offset : offset + COMMITMENT_BYTES]
        for offset in range(0, len(packed), COMMITMENT_BYTES)
    )


def evaluate_i8_matrix(
    values: bytes | bytearray | memoryview,
    *,
    rows: int,
    columns: int,
    point: Iterable[int | bytes],
) -> bytes:
    """Evaluate one row-major signed-i8 matrix MLE in a single native call."""

    dimensions = (rows, columns)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in dimensions
    ):
        raise PCSFormatError("matrix dimensions must be integers")
    if any(value <= 0 or value & (value - 1) for value in dimensions):
        raise PCSFormatError("matrix dimensions must be positive powers of two")
    value_count = rows * columns
    if value_count > MAX_BATCH_I8_BYTES:
        raise PCSFormatError(
            f"signed-i8 matrix exceeds {MAX_BATCH_I8_BYTES} bytes"
        )
    if not isinstance(values, (bytes, bytearray, memoryview)):
        raise PCSFormatError("signed-i8 matrix values must be packed bytes")
    raw = bytes(values)
    if len(raw) != value_count:
        raise PCSFormatError("signed-i8 matrix buffer has the wrong length")

    point_count = (rows.bit_length() - 1) + (columns.bit_length() - 1)
    point_raw, _ = _point_bytes(point, point_count)
    point_buffer = _point_pointer(point_raw)
    value_buffer = (ctypes.c_int8 * len(raw)).from_buffer_copy(raw)
    output = (_U8 * SCALAR_BYTES)()
    native = _native()
    status = int(
        native.lib.verathos_pcs_v2_evaluate_i8_matrix(
            value_buffer,
            len(raw),
            rows,
            columns,
            point_buffer,
            point_count,
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def evaluate_lean_projection_terminal_relation(
    values: bytes | bytearray | memoryview,
    *,
    claim_count: int,
    width: int,
    point: Iterable[int | bytes],
    alphas: Iterable[int | bytes],
    terminal_fold_evaluations: Iterable[int | bytes],
) -> int:
    """Evaluate one projection group's terminal relation in native code."""

    if (
        isinstance(claim_count, bool)
        or not isinstance(claim_count, int)
        or not 0 < claim_count <= 512
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or width & (width - 1)
        or width > MAX_VECTOR_LEN
    ):
        raise PCSFormatError("lean projection terminal geometry is invalid")
    if not isinstance(values, (bytes, bytearray, memoryview)):
        raise PCSFormatError("lean projection terminal rows must be packed bytes")
    raw = bytes(values)
    if len(raw) != claim_count * width:
        raise PCSFormatError(
            "lean projection terminal row buffer has the wrong length"
        )
    point_raw, point_count = _point_bytes(point, width.bit_length() - 1)
    alpha_items = tuple(alphas)
    terminal_items = tuple(terminal_fold_evaluations)
    scalar_count = claim_count * 4
    if len(alpha_items) != scalar_count or len(terminal_items) != scalar_count:
        raise PCSFormatError(
            "lean projection terminal scalar count is inconsistent"
        )
    alpha_raw = b"".join(encode_scalar(value) for value in alpha_items)
    terminal_raw = b"".join(
        encode_scalar(value) for value in terminal_items
    )

    native = _native()
    try:
        function = (
            native.lib.verathos_pcs_v2_lean_projection_terminal_relation
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks lean projection terminal evaluation"
        ) from exc
    function.argtypes = [
        _I8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
    ]
    function.restype = ctypes.c_int32
    value_buffer = (ctypes.c_int8 * len(raw)).from_buffer_copy(raw)
    point_buffer = _point_pointer(point_raw)
    alpha_buffer = _byte_array(alpha_raw)
    terminal_buffer = _byte_array(terminal_raw)
    output = (_U8 * SCALAR_BYTES)()
    status = int(
        function(
            value_buffer,
            len(raw),
            claim_count,
            width,
            point_buffer,
            point_count,
            alpha_buffer,
            len(alpha_raw),
            terminal_buffer,
            len(terminal_raw),
            output,
        )
    )
    if status != 0:
        _raise_status(native, status)
    return scalar_to_int(bytes(output))


def prove_gemm_batch_sumcheck_i8(
    *,
    batch_digest: bytes,
    x_values: bytes,
    w_values: bytes,
    y_values_i64_le: bytes,
    valid_inner: Iterable[int],
    block_count: int,
    rows: int,
    inner: int,
    columns: int,
) -> bytes:
    """Produce the portable native aggregate GEMM sumcheck wire proof."""

    digest = _fixed_bytes(batch_digest, 32, "GEMM batch digest")
    dimensions = (block_count, rows, inner, columns)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in dimensions
    ):
        raise PCSFormatError("GEMM batch dimensions must be integers")
    if not 1 <= block_count <= MAX_GEMM_BATCH_BLOCKS:
        raise PCSFormatError(
            f"GEMM batch block count must be in 1..={MAX_GEMM_BATCH_BLOCKS}"
        )
    if any(value <= 0 or value & (value - 1) for value in (rows, inner, columns)):
        raise PCSFormatError("GEMM batch dimensions must be positive powers of two")
    _checked_vector_length(inner)
    if not all(
        isinstance(value, (bytes, bytearray, memoryview))
        for value in (x_values, w_values, y_values_i64_le)
    ):
        raise PCSFormatError("GEMM batch witness buffers must be bytes")
    x_raw = bytes(x_values)
    w_raw = bytes(w_values)
    y_raw = bytes(y_values_i64_le)
    expected_x = block_count * rows * inner
    expected_w = block_count * inner * columns
    expected_y_count = block_count * rows * columns
    if (
        len(x_raw) != expected_x
        or len(w_raw) != expected_w
        or len(y_raw) != expected_y_count * 8
    ):
        raise PCSFormatError("GEMM batch witness buffer has the wrong length")
    if len(x_raw) + len(w_raw) + len(y_raw) + block_count * 4 > MAX_GEMM_BATCH_BYTES:
        raise PCSFormatError(f"GEMM batch witness exceeds {MAX_GEMM_BATCH_BYTES} bytes")
    try:
        valid = tuple(operator.index(value) for value in valid_inner)
    except TypeError as exc:
        raise PCSFormatError("GEMM valid-inner values must be integers") from exc
    if len(valid) != block_count or any(
        value <= 0 or value > inner or value >= (1 << 32) for value in valid
    ):
        raise PCSFormatError("GEMM valid-inner values are out of range")

    native = _native()
    capacity = int(native.lib.verathos_pcs_v2_gemm_batch_proof_size(block_count, inner))
    if capacity <= 0:
        raise PCSNativeError("native PCS returned an invalid GEMM batch proof size")
    digest_buffer = _byte_array(digest)
    x_buffer = (ctypes.c_int8 * len(x_raw)).from_buffer_copy(x_raw)
    w_buffer = (ctypes.c_int8 * len(w_raw)).from_buffer_copy(w_raw)
    y_buffer = (ctypes.c_int64 * expected_y_count).from_buffer_copy(y_raw)
    valid_buffer = (ctypes.c_uint32 * block_count)(*valid)
    proof_output = (_U8 * capacity)()
    proof_length = ctypes.c_size_t(0)
    status = int(
        native.lib.verathos_pcs_v2_gemm_batch_prove_i8(
            digest_buffer,
            x_buffer,
            len(x_raw),
            w_buffer,
            len(w_raw),
            y_buffer,
            expected_y_count,
            valid_buffer,
            block_count,
            block_count,
            rows,
            inner,
            columns,
            proof_output,
            capacity,
            ctypes.byref(proof_length),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if proof_length.value != capacity:
        raise PCSNativeError(
            "native PCS returned a non-canonical GEMM batch proof length"
        )
    return bytes(proof_output)


def prove_lean_projection_group(
    *,
    transcript_state: bytes,
    x_values_i8: bytes,
    z_values_i64_le: bytes,
    alpha_bytes: bytes,
    target_bytes: bytes,
    vector_count: int,
    width: int,
) -> bytes:
    """Produce one native lean-projection aggregate sumcheck and IPA opening."""

    state = _fixed_bytes(
        transcript_state,
        OUTER_DIGEST_BYTES,
        "lean projection transcript state",
    )
    if (
        isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or not 0 < vector_count <= MAX_LEAN_PROJECTION_VECTORS
    ):
        raise PCSFormatError("lean projection vector count is out of range")
    _length, padded, _log_n = _checked_vector_length(width)
    if padded != width:
        raise PCSFormatError("lean projection width must be a power of two")
    buffers = (x_values_i8, z_values_i64_le, alpha_bytes, target_bytes)
    if not all(
        isinstance(value, (bytes, bytearray, memoryview))
        for value in buffers
    ):
        raise PCSFormatError("lean projection native witnesses must be bytes")
    x_raw, z_raw, alphas_raw, targets_raw = (
        bytes(value) for value in buffers
    )
    element_count = vector_count * width
    if (
        len(x_raw) != element_count
        or len(z_raw) != element_count * 8
        or len(alphas_raw) != vector_count * SCALAR_BYTES
        or len(targets_raw) != vector_count * SCALAR_BYTES
        or len(x_raw) + len(z_raw) > MAX_LEAN_PROJECTION_BYTES
    ):
        raise PCSFormatError(
            "lean projection native witness buffers have the wrong length"
        )
    native = _native()
    try:
        capacity_function = (
            native.lib.verathos_pcs_v2_lean_projection_group_proof_capacity
        )
        prove_function = (
            native.lib.verathos_pcs_v2_lean_projection_group_prove
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks lean projection aggregate proving"
        ) from exc
    capacity_function.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    capacity_function.restype = ctypes.c_size_t
    prove_function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _SIZE_PTR,
    ]
    prove_function.restype = ctypes.c_int32
    capacity = int(capacity_function(vector_count, width))
    if capacity <= 0:
        raise PCSFormatError("lean projection native proof shape is invalid")
    state_buffer = _byte_array(state)
    x_buffer = _byte_array(x_raw)
    z_buffer = _byte_array(z_raw)
    alpha_buffer = _byte_array(alphas_raw)
    target_buffer = _byte_array(targets_raw)
    output = (_U8 * capacity)()
    output_len = ctypes.c_size_t()
    status = int(
        prove_function(
            state_buffer,
            x_buffer,
            len(x_raw),
            z_buffer,
            len(z_raw),
            alpha_buffer,
            len(alphas_raw),
            target_buffer,
            len(targets_raw),
            vector_count,
            width,
            output,
            capacity,
            ctypes.byref(output_len),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if not 0 < output_len.value <= capacity:
        raise PCSNativeError(
            "lean projection native proof returned an invalid length"
        )
    return bytes(output[: output_len.value])


def prove_lean_projection_group_deduplicated(
    *,
    transcript_state: bytes,
    unique_x_values_i8: bytes,
    x_vector_map: Iterable[int],
    unique_z_values_i64_le: bytes,
    z_vector_map: Iterable[int],
    alpha_bytes: bytes,
    target_bytes: bytes,
    vector_count: int,
    width: int,
) -> bytes:
    """Prove the canonical group from deduplicated internal witness rows."""

    state = _fixed_bytes(
        transcript_state,
        OUTER_DIGEST_BYTES,
        "lean projection transcript state",
    )
    if (
        isinstance(vector_count, bool)
        or not isinstance(vector_count, int)
        or not 0 < vector_count <= MAX_LEAN_PROJECTION_VECTORS
    ):
        raise PCSFormatError("lean projection vector count is out of range")
    _length, padded, _log_n = _checked_vector_length(width)
    if padded != width:
        raise PCSFormatError("lean projection width must be a power of two")
    buffers = (
        unique_x_values_i8,
        unique_z_values_i64_le,
        alpha_bytes,
        target_bytes,
    )
    if not all(
        isinstance(value, (bytes, bytearray, memoryview))
        for value in buffers
    ):
        raise PCSFormatError(
            "lean projection deduplicated witnesses must be bytes"
        )
    x_raw, z_raw, alphas_raw, targets_raw = (
        bytes(value) for value in buffers
    )
    try:
        x_map = tuple(int(value) for value in x_vector_map)
        z_map = tuple(int(value) for value in z_vector_map)
    except (TypeError, ValueError) as exc:
        raise PCSFormatError(
            "lean projection deduplicated maps are malformed"
        ) from exc
    if (
        len(x_map) != vector_count
        or len(z_map) != vector_count
        or len(alphas_raw) != vector_count * SCALAR_BYTES
        or len(targets_raw) != vector_count * SCALAR_BYTES
        or len(x_raw) % width
        or len(z_raw) % (width * 8)
    ):
        raise PCSFormatError(
            "lean projection deduplicated witness geometry is malformed"
        )
    unique_x_count = len(x_raw) // width
    unique_z_count = len(z_raw) // (width * 8)
    if (
        not 0 < unique_x_count <= vector_count
        or not 0 < unique_z_count <= vector_count
        or any(value < 0 or value >= unique_x_count for value in x_map)
        or any(value < 0 or value >= unique_z_count for value in z_map)
    ):
        raise PCSFormatError(
            "lean projection deduplicated map is out of range"
        )
    native = _native()
    try:
        capacity_function = (
            native.lib.verathos_pcs_v2_lean_projection_group_proof_capacity
        )
        prove_function = (
            native.lib
            .verathos_pcs_v2_lean_projection_group_prove_deduplicated
        )
    except AttributeError as exc:
        raise PCSUnavailableError(
            "PCS native library lacks deduplicated lean projection proving"
        ) from exc
    capacity_function.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    capacity_function.restype = ctypes.c_size_t
    prove_function.argtypes = [
        _U8_PTR,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U32_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U32_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _U8_PTR,
        ctypes.c_size_t,
        _SIZE_PTR,
    ]
    prove_function.restype = ctypes.c_int32
    capacity = int(capacity_function(vector_count, width))
    if capacity <= 0:
        raise PCSFormatError("lean projection native proof shape is invalid")
    state_buffer = _byte_array(state)
    x_buffer = _byte_array(x_raw)
    x_map_buffer = (ctypes.c_uint32 * len(x_map))(*x_map)
    z_buffer = _byte_array(z_raw)
    z_map_buffer = (ctypes.c_uint32 * len(z_map))(*z_map)
    alpha_buffer = _byte_array(alphas_raw)
    target_buffer = _byte_array(targets_raw)
    output = (_U8 * capacity)()
    output_len = ctypes.c_size_t()
    status = int(
        prove_function(
            state_buffer,
            x_buffer,
            len(x_raw),
            unique_x_count,
            x_map_buffer,
            len(x_map),
            z_buffer,
            len(z_raw),
            unique_z_count,
            z_map_buffer,
            len(z_map),
            alpha_buffer,
            len(alphas_raw),
            target_buffer,
            len(targets_raw),
            vector_count,
            width,
            output,
            capacity,
            ctypes.byref(output_len),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if not 0 < output_len.value <= capacity:
        raise PCSNativeError(
            "lean projection native proof returned an invalid length"
        )
    return bytes(output[: output_len.value])


def prove_i8_linear_combination(
    values: bytes,
    *,
    vector_length: int,
    coefficients: Iterable[int | bytes],
    point: Iterable[int | bytes],
    outer_digest: bytes,
) -> PCSOpeningV2:
    """Open one weighted sum of equal signed-i8 vectors at a shared point."""

    length, padded, log_n = _checked_vector_length(vector_length)
    if length != padded:
        raise PCSFormatError("linear-combination vector length must be a power of two")
    if not isinstance(values, (bytes, bytearray, memoryview)):
        raise PCSFormatError("linear-combination values must be packed bytes")
    raw = bytes(values)
    if not raw or len(raw) % length:
        raise PCSFormatError("linear-combination values contain an incomplete vector")
    term_count = len(raw) // length
    if term_count > MAX_COMBINE_TERMS or len(raw) > MAX_BATCH_I8_BYTES:
        raise PCSFormatError(
            "linear-combination term count or byte size is out of range"
        )
    coefficient_raw, coefficient_count = _coefficient_sequence_bytes(coefficients)
    if coefficient_count != term_count:
        raise PCSFormatError("linear-combination coefficient count does not match")
    point_raw, point_count = _point_bytes(point, log_n)
    outer_raw = _fixed_bytes(outer_digest, 32, "outer digest")

    native = _native()
    value_buffer = (ctypes.c_int8 * len(raw)).from_buffer_copy(raw)
    coefficient_buffer = _byte_array(coefficient_raw)
    point_buffer = _point_pointer(point_raw)
    outer_buffer = _byte_array(outer_raw)
    commitment_output = (_U8 * COMMITMENT_BYTES)()
    evaluation_output = (_U8 * SCALAR_BYTES)()
    capacity = int(native.lib.verathos_pcs_v2_proof_size(length))
    proof_output = (_U8 * capacity)()
    proof_length = ctypes.c_size_t(0)
    status = int(
        native.lib.verathos_pcs_v2_prove_i8_linear_combination(
            value_buffer,
            len(raw),
            length,
            term_count,
            coefficient_buffer,
            len(coefficient_raw),
            point_buffer,
            point_count,
            outer_buffer,
            commitment_output,
            evaluation_output,
            proof_output,
            capacity,
            ctypes.byref(proof_length),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if proof_length.value != capacity:
        raise PCSNativeError("native PCS returned a non-canonical proof length")
    return PCSOpeningV2(
        bytes(commitment_output),
        bytes(evaluation_output),
        bytes(proof_output),
        length,
        padded,
        ENCODING_PALLAS_SCALAR,
    )


def prove_i64_linear_combination_linear(
    values_i64_le: bytes | bytearray | memoryview,
    *,
    vector_length: int,
    combination_coefficients: Iterable[int | bytes],
    linear_coefficients: object,
    outer_digest: bytes | bytearray | memoryview,
) -> PCSOpeningV2:
    """Open a public linear functional of combined signed-i64 vectors.

    ``values_i64_le`` is term-major packed little-endian signed-i64 data.
    The native prover forms one Pallas-field polynomial using
    ``combination_coefficients`` and proves its dot product with the exact
    ``linear_coefficients`` vector. The resulting commitment is the matching
    linear combination of the individual signed-i64 commitments.
    """

    length, padded, _log_n = _checked_vector_length(vector_length)
    if length != padded:
        raise PCSFormatError(
            "linear-combination vector length must be a power of two"
        )
    if not isinstance(values_i64_le, (bytes, bytearray, memoryview)):
        raise PCSFormatError(
            "linear-combination signed-i64 values must be packed bytes"
        )
    raw = bytes(values_i64_le)
    vector_bytes = length * 8
    if (
        not raw
        or len(raw) % vector_bytes
        or len(raw) > MAX_BATCH_I64_BYTES
    ):
        raise PCSFormatError(
            "linear-combination signed-i64 values have a wrong length"
        )
    term_count = len(raw) // vector_bytes
    if term_count > MAX_COMBINE_TERMS:
        raise PCSFormatError(
            "linear-combination term count is out of range"
        )
    combination_raw, combination_count = _coefficient_sequence_bytes(
        combination_coefficients
    )
    if combination_count != term_count:
        raise PCSFormatError(
            "linear-combination coefficient count does not match"
        )
    linear_raw = _linear_coefficient_bytes(linear_coefficients, padded)
    outer_raw = _fixed_bytes(
        outer_digest,
        OUTER_DIGEST_BYTES,
        "outer digest",
    )
    value_count = len(raw) // 8
    value_buffer = (ctypes.c_int64 * value_count).from_buffer_copy(raw)
    combination_buffer = _byte_array(combination_raw)
    linear_buffer = _byte_array(linear_raw)
    outer_buffer = _byte_array(outer_raw)
    native = _native()
    commitment_output = (_U8 * COMMITMENT_BYTES)()
    evaluation_output = (_U8 * SCALAR_BYTES)()
    capacity = int(native.lib.verathos_pcs_v2_proof_size(length))
    if capacity <= 0:
        raise PCSNativeError("native PCS returned an invalid proof size")
    proof_output = (_U8 * capacity)()
    proof_length = ctypes.c_size_t(0)
    status = int(
        native.lib.verathos_pcs_v3_prove_i64_linear_combination_linear(
            value_buffer,
            value_count,
            length,
            term_count,
            combination_buffer,
            len(combination_raw),
            linear_buffer,
            len(linear_raw),
            outer_buffer,
            commitment_output,
            evaluation_output,
            proof_output,
            capacity,
            ctypes.byref(proof_length),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if proof_length.value != capacity:
        raise PCSNativeError(
            "native PCS returned a non-canonical linear proof length"
        )
    return PCSOpeningV2(
        commitment=bytes(commitment_output),
        evaluation=bytes(evaluation_output),
        proof=bytes(proof_output),
        vector_length=length,
        padded_length=padded,
        encoding=ENCODING_SIGNED_I64,
    )


def evaluate(
    values: object,
    point: Iterable[int | bytes],
    *,
    encoding: str | int = ENCODING_SIGNED_I8,
) -> bytes:
    """Evaluate the padded vector's multilinear extension at ``point``."""

    encoding_id = _encoding_id(encoding)
    value_buffer, length, suffix = _values_buffer(values, encoding_id)
    _, _, log_n = _checked_vector_length(length)
    point_raw, point_count = _point_bytes(point, log_n)
    point_buffer = _point_pointer(point_raw)
    native = _native()
    output = (_U8 * SCALAR_BYTES)()
    status = int(
        getattr(native.lib, f"verathos_pcs_v2_evaluate_{suffix}")(
            value_buffer, length, point_buffer, point_count, output
        )
    )
    if status != 0:
        _raise_status(native, status)
    return bytes(output)


def prove(
    values: object,
    point: Iterable[int | bytes],
    outer_digest: bytes | bytearray | memoryview,
    *,
    encoding: str | int = ENCODING_SIGNED_I8,
) -> PCSOpeningV2:
    """Create a statement-bound exact MLE opening for an encoded vector."""

    encoding_id = _encoding_id(encoding)
    value_buffer, length, suffix = _values_buffer(values, encoding_id)
    _, padded, log_n = _checked_vector_length(length)
    point_raw, point_count = _point_bytes(point, log_n)
    outer_raw = _fixed_bytes(outer_digest, OUTER_DIGEST_BYTES, "outer digest")
    point_buffer = _point_pointer(point_raw)
    outer_buffer = _byte_array(outer_raw)
    native = _native()
    commitment_output = (_U8 * COMMITMENT_BYTES)()
    evaluation_output = (_U8 * SCALAR_BYTES)()
    capacity = int(native.lib.verathos_pcs_v2_proof_size(length))
    if capacity <= 0:
        raise PCSNativeError("native PCS returned an invalid proof size")
    proof_output = (_U8 * capacity)()
    proof_length = ctypes.c_size_t(0)
    status = int(
        getattr(native.lib, f"verathos_pcs_v2_prove_{suffix}")(
            value_buffer,
            length,
            point_buffer,
            point_count,
            outer_buffer,
            commitment_output,
            evaluation_output,
            proof_output,
            capacity,
            ctypes.byref(proof_length),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if proof_length.value != capacity:
        raise PCSNativeError("native PCS returned a non-canonical proof length")
    return PCSOpeningV2(
        commitment=bytes(commitment_output),
        evaluation=bytes(evaluation_output),
        proof=bytes(proof_output),
        vector_length=length,
        padded_length=padded,
        encoding=encoding_id,
    )


def _linear_coefficient_bytes(
    coefficients: object,
    expected_count: int,
) -> bytes:
    if isinstance(coefficients, (bytes, bytearray, memoryview)):
        raw = bytes(coefficients)
        if len(raw) != expected_count * SCALAR_BYTES:
            raise PCSFormatError(
                "linear coefficient vector has the wrong byte length"
            )
        # The native linear-opening entry points decode every scalar with
        # `decode_scalar` before using it and fail closed on non-canonical
        # encodings.  Re-decoding and re-encoding a packed public vector in
        # Python duplicated that validation for millions of coefficients on
        # production proofs.
        return raw
    if isinstance(coefficients, str):
        raise PCSFormatError(
            "linear coefficients must be packed bytes or an iterable"
        )
    try:
        items = tuple(islice(iter(coefficients), expected_count + 1))  # type: ignore[arg-type]
    except TypeError as exc:
        raise PCSFormatError(
            "linear coefficients must be packed bytes or an iterable"
        ) from exc
    if len(items) != expected_count:
        raise PCSFormatError("linear coefficient count is wrong")
    return b"".join(encode_scalar(item) for item in items)


def prove_linear(
    values: object,
    coefficients: object,
    outer_digest: bytes | bytearray | memoryview,
    *,
    encoding: str | int = ENCODING_SIGNED_I8,
) -> PCSOpeningV2:
    """Open an arbitrary public linear functional of a signed vector.

    ``coefficients`` contains exactly the zero-padded vector length in
    canonical Pallas scalars. The vector is transcript-bound but is not
    serialized into the logarithmic IPA proof.
    """

    encoding_id = _encoding_id(encoding)
    if encoding_id not in (ENCODING_SIGNED_I8, ENCODING_SIGNED_I64):
        raise PCSFormatError(
            "arbitrary linear openings support signed i8 or signed i64"
        )
    value_buffer, length, suffix = _values_buffer(values, encoding_id)
    _, padded, _log_n = _checked_vector_length(length)
    coefficient_raw = _linear_coefficient_bytes(coefficients, padded)
    coefficient_buffer = _byte_array(coefficient_raw)
    outer_raw = _fixed_bytes(outer_digest, OUTER_DIGEST_BYTES, "outer digest")
    outer_buffer = _byte_array(outer_raw)
    native = _native()
    commitment_output = (_U8 * COMMITMENT_BYTES)()
    evaluation_output = (_U8 * SCALAR_BYTES)()
    capacity = int(native.lib.verathos_pcs_v2_proof_size(length))
    if capacity <= 0:
        raise PCSNativeError("native PCS returned an invalid proof size")
    proof_output = (_U8 * capacity)()
    proof_length = ctypes.c_size_t(0)
    status = int(
        getattr(native.lib, f"verathos_pcs_v3_prove_linear_{suffix}")(
            value_buffer,
            length,
            coefficient_buffer,
            padded,
            outer_buffer,
            commitment_output,
            evaluation_output,
            proof_output,
            capacity,
            ctypes.byref(proof_length),
        )
    )
    if status != 0:
        _raise_status(native, status)
    if proof_length.value != capacity:
        raise PCSNativeError(
            "native PCS returned a non-canonical linear proof length"
        )
    return PCSOpeningV2(
        commitment=bytes(commitment_output),
        evaluation=bytes(evaluation_output),
        proof=bytes(proof_output),
        vector_length=length,
        padded_length=padded,
        encoding=encoding_id,
    )


def verify_linear(
    opening: PCSOpeningV2,
    coefficients: object,
    outer_digest: bytes | bytearray | memoryview,
) -> bool:
    """Verify an arbitrary-public-vector IPA opening."""

    if not isinstance(opening, PCSOpeningV2):
        raise PCSFormatError("opening must be PCSOpeningV2")
    encoding_id = _encoding_id(opening.encoding)
    if encoding_id not in (ENCODING_SIGNED_I8, ENCODING_SIGNED_I64):
        raise PCSFormatError(
            "arbitrary linear openings support signed i8 or signed i64"
        )
    length, padded, _log_n = _checked_vector_length(opening.vector_length)
    if opening.padded_length != padded:
        raise PCSFormatError("opening has an inconsistent padded vector length")
    coefficient_raw = _linear_coefficient_bytes(coefficients, padded)
    commitment = _fixed_bytes(
        opening.commitment,
        COMMITMENT_BYTES,
        "commitment",
    )
    evaluation = encode_scalar(opening.evaluation)
    outer_raw = _fixed_bytes(
        outer_digest,
        OUTER_DIGEST_BYTES,
        "outer digest",
    )
    native = _native()
    expected_proof = int(native.lib.verathos_pcs_v2_proof_size(length))
    proof = _fixed_bytes(opening.proof, expected_proof, "IPA proof")
    status = int(
        native.lib.verathos_pcs_v3_verify_linear(
            _byte_array(commitment),
            length,
            encoding_id,
            _byte_array(coefficient_raw),
            padded,
            _byte_array(outer_raw),
            _byte_array(evaluation),
            _byte_array(proof),
            len(proof),
        )
    )
    if status == 1:
        return True
    if status == 0:
        return False
    _raise_status(native, status)
    raise AssertionError("unreachable")


def verify(
    opening: PCSOpeningV2,
    point: Iterable[int | bytes],
    outer_digest: bytes | bytearray | memoryview,
) -> bool:
    """Verify an opening.

    Returns ``False`` for a canonical but cryptographically invalid opening.
    Malformed/non-canonical data raises :class:`PCSFormatError`.
    """

    if not isinstance(opening, PCSOpeningV2):
        raise PCSFormatError("opening must be PCSOpeningV2")
    encoding_id = _encoding_id(opening.encoding)
    length, padded, log_n = _checked_vector_length(opening.vector_length)
    if opening.padded_length != padded:
        raise PCSFormatError("opening has an inconsistent padded vector length")
    commitment = _fixed_bytes(opening.commitment, COMMITMENT_BYTES, "commitment")
    evaluation = encode_scalar(opening.evaluation)
    point_raw, point_count = _point_bytes(point, log_n)
    outer_raw = _fixed_bytes(outer_digest, OUTER_DIGEST_BYTES, "outer digest")
    native = _native()
    expected_proof = int(native.lib.verathos_pcs_v2_proof_size(length))
    proof = _fixed_bytes(opening.proof, expected_proof, "IPA proof")

    commitment_buffer = _byte_array(commitment)
    point_buffer = _point_pointer(point_raw)
    outer_buffer = _byte_array(outer_raw)
    evaluation_buffer = _byte_array(evaluation)
    proof_buffer = _byte_array(proof)
    status = int(
        native.lib.verathos_pcs_v2_verify(
            commitment_buffer,
            length,
            encoding_id,
            point_buffer,
            point_count,
            outer_buffer,
            evaluation_buffer,
            proof_buffer,
            len(proof),
        )
    )
    if status == 1:
        return True
    if status == 0:
        return False
    _raise_status(native, status)
    raise AssertionError("unreachable")


__all__ = [
    "ABI_VERSION",
    "COMMITMENT_BYTES",
    "ENCODING_PALLAS_SCALAR",
    "ENCODING_SIGNED_I8",
    "ENCODING_SIGNED_I64",
    "MAX_BATCH_COMMITMENTS",
    "MAX_BATCH_I8_BYTES",
    "MAX_BATCH_I64_BYTES",
    "MAX_COMBINE_TERMS",
    "MAX_CATALOG_COMBINE_TERMS",
    "MAX_CATALOG_FOLDS",
    "MAX_LEAN_PROJECTION_BYTES",
    "MAX_LEAN_PROJECTION_VECTORS",
    "MAX_MERKLE_BUILD_LEAVES",
    "MAX_VALIDATE_COMMITMENTS",
    "MAX_VECTOR_LEN",
    "OUTER_DIGEST_BYTES",
    "PALLAS_SCALAR_MODULUS",
    "PCSFormatError",
    "PCSNativeError",
    "PCSOpeningV2",
    "PCSUnavailableError",
    "PROTOCOL_VERSION",
    "SCALAR_BYTES",
    "combine_commitments",
    "combine_registered_catalog_u31_batch",
    "combine_registered_catalog_u7_batch",
    "combine_registered_catalog_u63",
    "combine_registered_lean_projection_aggregate",
    "derive_lean_projection_claim_targets",
    "derive_registered_lean_projection_claim_materials",
    "build_merkle_levels",
    "commit",
    "commit_i8_batch",
    "encode_scalar",
    "evaluate",
    "evaluate_i8_matrix",
    "evaluate_lean_projection_terminal_relation",
    "fold_u31_linear_coefficients",
    "hash_fp16_blocks",
    "native_library_path",
    "proof_size",
    "prove",
    "prove_i64_linear_combination_linear",
    "prove_linear",
    "prove_lean_projection_group",
    "prove_lean_projection_group_deduplicated",
    "register_catalog_commitments",
    "scalar_to_int",
    "validate_commitments",
    "verify",
    "verify_linear",
]
