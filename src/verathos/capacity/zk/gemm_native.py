"""ctypes binding to ``libverathos_gemm`` (``native/gemm``): the GEMM sumcheck prover over the Pallas scalar field and
a fast int8 matrix MLE evaluator. Built in place by ``verathos-build-native``."""

from __future__ import annotations

import ctypes
import operator
import sys
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Final

ABI_VERSION: Final = 1
MAX_GEMM_BATCH_BLOCKS: Final = 64
MAX_GEMM_BATCH_BYTES: Final = 16 << 20
MAX_MATRIX_BYTES: Final = 16 << 20
SCALAR_BYTES: Final = 32
PALLAS_SCALAR_MODULUS: Final = int("40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001", 16)

LIBRARY_NAME = "libverathos_gemm.dylib" if sys.platform == "darwin" else "libverathos_gemm.so"
LIBRARY_PATH = Path(__file__).resolve().parent / LIBRARY_NAME

_U8 = ctypes.c_uint8
_U8_PTR = ctypes.POINTER(_U8)
_I8_PTR = ctypes.POINTER(ctypes.c_int8)
_I64_PTR = ctypes.POINTER(ctypes.c_int64)
_U32_PTR = ctypes.POINTER(ctypes.c_uint32)
_SIZE = ctypes.c_size_t


class NativeUnavailableError(RuntimeError):
    """The native library is not built or does not match this binding."""


class NativeError(RuntimeError):
    """The native library reported a failure."""


class NativeFormatError(NativeError, ValueError):
    """Malformed, non-canonical or out-of-bounds input."""


class _Library:
    def __init__(self, path: Path) -> None:
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise NativeUnavailableError(f"failed to load {path}: {exc}") from exc
        lib.verathos_gemm_abi_version.argtypes = []
        lib.verathos_gemm_abi_version.restype = ctypes.c_uint32
        if lib.verathos_gemm_abi_version() != ABI_VERSION:
            raise NativeUnavailableError(f"{path} has a different ABI version; rebuild with verathos-build-native")
        lib.verathos_gemm_last_error.argtypes = [_U8_PTR, _SIZE]
        lib.verathos_gemm_last_error.restype = _SIZE
        lib.verathos_gemm_batch_proof_size.argtypes = [_SIZE, _SIZE]
        lib.verathos_gemm_batch_proof_size.restype = _SIZE
        lib.verathos_gemm_batch_prove_i8.argtypes = [
            _U8_PTR, _I8_PTR, _SIZE, _I8_PTR, _SIZE, _I64_PTR, _SIZE, _U32_PTR, _SIZE, _SIZE, _SIZE, _SIZE, _SIZE,
            _U8_PTR, _SIZE, ctypes.POINTER(_SIZE)]
        lib.verathos_gemm_batch_prove_i8.restype = ctypes.c_int32
        lib.verathos_gemm_evaluate_i8_matrix.argtypes = [_I8_PTR, _SIZE, _SIZE, _SIZE, _U8_PTR, _SIZE, _U8_PTR]
        lib.verathos_gemm_evaluate_i8_matrix.restype = ctypes.c_int32
        self.lib = lib

    def raise_status(self, status: int) -> None:
        size = min(int(self.lib.verathos_gemm_last_error(None, 0)) + 1, 4096)
        message = "native GEMM operation failed"
        if size > 1:
            out = (_U8 * size)()
            self.lib.verathos_gemm_last_error(out, size)
            message = bytes(out).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        raise (NativeFormatError if status in (-1, -2, -3, -4, -6) else NativeError)(message)


_LIBRARY: _Library | None = None
_LOCK = Lock()


def _library() -> _Library:
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            if not LIBRARY_PATH.is_file():
                raise NativeUnavailableError(f"{LIBRARY_PATH} not found; run verathos-build-native")
            _LIBRARY = _Library(LIBRARY_PATH)
        return _LIBRARY


def _power_of_two(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value & (value - 1):
        raise NativeFormatError(f"{label} must be a positive power of two")
    return value


def _scalar_bytes(value: int) -> bytes:
    if isinstance(value, bool) or not 0 <= operator.index(value) < PALLAS_SCALAR_MODULUS:
        raise NativeFormatError("coordinate is not a canonical Pallas scalar")
    return int(value).to_bytes(SCALAR_BYTES, "little")


def evaluate_i8_matrix(values: bytes, *, rows: int, columns: int, point: Iterable[int]) -> int:
    """MLE of a row-major int8 matrix at ``point`` (row coordinates first, MSB-first order)."""
    rows, columns = _power_of_two(rows, "rows"), _power_of_two(columns, "columns")
    if not isinstance(values, bytes) or len(values) != rows * columns or len(values) > MAX_MATRIX_BYTES:
        raise NativeFormatError("int8 matrix buffer has the wrong length")
    coordinates = tuple(point)
    count = rows.bit_length() - 1 + columns.bit_length() - 1
    if len(coordinates) != count:
        raise NativeFormatError(f"point must have {count} coordinates")
    raw_point = b"".join(_scalar_bytes(c) for c in coordinates)
    point_buffer = (_U8 * len(raw_point)).from_buffer_copy(raw_point) if raw_point else None
    out = (_U8 * SCALAR_BYTES)()
    library = _library()
    status = library.lib.verathos_gemm_evaluate_i8_matrix(
        (ctypes.c_int8 * len(values)).from_buffer_copy(values), len(values), rows, columns, point_buffer, count, out)
    if status:
        library.raise_status(status)
    return int.from_bytes(bytes(out), "little")


def prove_gemm_batch_i8(*, batch_digest: bytes, x_values: bytes, w_values: bytes, y_values_i64_le: bytes,
                        valid_inner: Iterable[int], block_count: int, rows: int, inner: int, columns: int) -> bytes:
    """Aggregate sumcheck proof that ``Y = X @ W`` for ``block_count`` int8 blocks, bound to ``batch_digest``."""
    if not isinstance(batch_digest, bytes) or len(batch_digest) != 32:
        raise NativeFormatError("batch digest must be 32 bytes")
    if isinstance(block_count, bool) or not isinstance(block_count, int) \
            or not 1 <= block_count <= MAX_GEMM_BATCH_BLOCKS:
        raise NativeFormatError(f"block count must be in 1..={MAX_GEMM_BATCH_BLOCKS}")
    rows, inner, columns = (_power_of_two(v, n) for v, n in ((rows, "rows"), (inner, "inner"), (columns, "columns")))
    y_count = block_count * rows * columns
    if (len(x_values), len(w_values), len(y_values_i64_le)) != \
            (block_count * rows * inner, block_count * inner * columns, y_count * 8):
        raise NativeFormatError("GEMM witness buffer has the wrong length")
    if len(x_values) + len(w_values) + len(y_values_i64_le) + block_count * 4 > MAX_GEMM_BATCH_BYTES:
        raise NativeFormatError(f"GEMM witness exceeds {MAX_GEMM_BATCH_BYTES} bytes")
    valid = tuple(operator.index(v) for v in valid_inner)
    if len(valid) != block_count or any(not 0 < v <= inner for v in valid):
        raise NativeFormatError("valid inner lengths are out of range")

    library = _library()
    capacity = int(library.lib.verathos_gemm_batch_proof_size(block_count, inner))
    out = (_U8 * capacity)()
    written = _SIZE(0)
    status = library.lib.verathos_gemm_batch_prove_i8(
        (_U8 * 32).from_buffer_copy(batch_digest),
        (ctypes.c_int8 * len(x_values)).from_buffer_copy(x_values), len(x_values),
        (ctypes.c_int8 * len(w_values)).from_buffer_copy(w_values), len(w_values),
        (ctypes.c_int64 * y_count).from_buffer_copy(y_values_i64_le), y_count,
        (ctypes.c_uint32 * block_count)(*valid), block_count,
        block_count, rows, inner, columns, out, capacity, ctypes.byref(written))
    if status:
        library.raise_status(status)
    if written.value != capacity:
        raise NativeError("native GEMM proof has a non-canonical length")
    return bytes(out)
