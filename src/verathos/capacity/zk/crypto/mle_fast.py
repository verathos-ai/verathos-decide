"""
Fast vectorized Multilinear Extension (MLE) operations.

Uses NumPy for batch operations. Falls back to native extension when available.
Uses CUDA for GPU acceleration when available.
"""

from typing import List, Optional
import math
import numpy as np
import torch

from verathos.capacity.zk.crypto.field import P
from verathos.capacity.zk.crypto.field_fast import (
    build_eq_table_vec,
    fold_table_vec,
    dot_product_field,
)

# Try to import native extension
_HAS_NATIVE = False
_HAS_CUDA = False
try:
    from verathos.capacity.zk.cuda import zkllm_native as _native
    if _native is None:
        raise ImportError("zkllm_native unavailable")
    _HAS_NATIVE = True
    _HAS_CUDA = getattr(_native, 'HAS_CUDA', False)
except ImportError:
    try:
        import sys
        from pathlib import Path
        cuda_path = Path(__file__).parent.parent / "cuda"
        if str(cuda_path) not in sys.path:
            sys.path.insert(0, str(cuda_path))
        import zkllm_native as _native
        _HAS_NATIVE = True
        _HAS_CUDA = getattr(_native, 'HAS_CUDA', False)
    except ImportError:
        pass

# Minimum size to use CUDA (CUDA only faster for large arrays due to kernel launch overhead)
_CUDA_MIN_SIZE = 8192


def int_to_bits(n: int, num_bits: int) -> List[int]:
    """Convert integer to list of bits (LSB first)."""
    return [(n >> i) & 1 for i in range(num_bits)]


def eq_eval_fast(x: np.ndarray, w: np.ndarray) -> int:
    """
    Fast eq evaluation using vectorized operations.

    eq(x, w) = prod_i (x_i * w_i + (1 - x_i) * (1 - w_i))
    """
    if _HAS_NATIVE:
        return _native.native_eq_eval(x, w)

    # Vectorized computation
    # For binary w_i: term_i = x_i if w_i=1 else (1-x_i)
    x_obj = x.astype(object)
    w_int = w.astype(int)

    terms = np.where(w_int == 1, x_obj, (P - x_obj + 1) % P)

    # Product of all terms
    result = 1
    for t in terms:
        result = (result * int(t)) % P
        # Mersenne reduction
        lo = result & P
        hi = result >> 61
        result = lo + hi
        if result >= P:
            result -= P

    return result


def build_eq_table_fast(point: List[int]) -> np.ndarray:
    """
    Fast eq table construction using best backend.
    Uses CUDA for large tables when available.
    """
    size = 1 << len(point)

    # Use CUDA for large tables
    if _HAS_CUDA and size >= _CUDA_MIN_SIZE:
        point_t = torch.tensor(point, dtype=torch.int64).cuda()
        result_t = _native.cuda_build_eq_table(point_t, 0)
        return result_t.cpu().numpy().astype(np.uint64)

    if _HAS_NATIVE:
        return _native.native_build_eq_table(np.array(point, dtype=np.uint64))
    else:
        return build_eq_table_vec(point)


class MultilinearExtensionFast:
    """
    Fast MLE using vectorized operations.
    """

    def __init__(self, evaluations: np.ndarray, num_vars: Optional[int] = None):
        """
        Create MLE from evaluations at all 2^n Boolean points.

        Args:
            evaluations: Array of f(w) for all w in {0,1}^n (in lex order)
            num_vars: Number of variables (inferred from len if not given)
        """
        if isinstance(evaluations, list):
            evaluations = np.array(evaluations, dtype=np.uint64)

        if num_vars is None:
            num_vars = int(math.log2(len(evaluations)))
            if 1 << num_vars != len(evaluations):
                raise ValueError("evaluations length must be power of 2")

        self.num_vars = num_vars
        self.evals = evaluations.astype(np.uint64)

    def evaluate(self, point: List[int]) -> int:
        """
        Evaluate MLE at arbitrary field point using fast backend.

        f~(x) = sum_{w in {0,1}^n} f(w) * eq(x, w)
        """
        if len(point) != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} variables, got {len(point)}")

        if _HAS_NATIVE:
            return _native.native_mle_evaluate(self.evals, np.array(point, dtype=np.uint64))

        # Build eq table and dot product
        eq_table = build_eq_table_fast(point)

        # Ensure same size
        n = min(len(self.evals), len(eq_table))

        return dot_product_field(self.evals[:n], eq_table[:n])

    def evaluate_partial(self, point: List[int]) -> "MultilinearExtensionFast":
        """
        Partially evaluate MLE, fixing the first len(point) variables.
        """
        if len(point) > self.num_vars:
            raise ValueError("Too many variables to fix")

        if len(point) == 0:
            return self

        # Use bookkeeping table approach
        table = self.evals.copy()

        for r in point:
            table = fold_table_vec(table, r)

        return MultilinearExtensionFast(table, self.num_vars - len(point))

    def partial_evaluate_first(self, point: List[int]) -> np.ndarray:
        """
        Partially evaluate, fixing first len(point) variables.
        Returns flat table of remaining evaluations.
        """
        mle = self.evaluate_partial(point)
        return mle.evals

    def get_table(self) -> np.ndarray:
        """Get raw evaluation table."""
        return self.evals.copy()


def build_mle_from_matrix_fast(matrix: np.ndarray) -> MultilinearExtensionFast:
    """
    Build MLE from 2D matrix using vectorized operations.

    Matrix is interpreted as function M(i, j) with:
    - i in {0,1}^m (row index)
    - j in {0,1}^n (column index)
    """
    rows = matrix.shape[0]
    cols = matrix.shape[1] if len(matrix.shape) > 1 else 1

    # Pad to powers of 2
    m = max(1, int(math.ceil(math.log2(rows)))) if rows > 0 else 1
    n = max(1, int(math.ceil(math.log2(cols)))) if cols > 0 else 1

    padded_rows = 1 << m
    padded_cols = 1 << n

    # Build flat evaluation table in column-major order for LSB-first bits
    # Vectorized: pad matrix, then flatten in Fortran (column-major) order
    padded = np.zeros((padded_rows, padded_cols), dtype=np.uint64)
    padded[:rows, :cols] = matrix[:rows, :cols]

    # Column-major flatten gives exactly evals[i + j * padded_rows] = matrix[i, j]
    evals = padded.ravel(order='F').copy()

    return MultilinearExtensionFast(evals, m + n)


def fold_table_fast(table: np.ndarray, r: int) -> np.ndarray:
    """
    Fast table folding using best backend.

    new[i] = table[2i] * (1-r) + table[2i+1] * r
    Uses CUDA for large tables when available.
    """
    # Use CUDA for large tables
    if _HAS_CUDA and len(table) >= _CUDA_MIN_SIZE:
        table_t = torch.from_numpy(table.astype(np.int64)).cuda()
        result_t = _native.cuda_fold_table(table_t, r, 0)
        return result_t.cpu().numpy().astype(np.uint64)

    if _HAS_NATIVE:
        return _native.native_fold_table(table, r)
    else:
        return fold_table_vec(table, r)


# Export as drop-in replacements
MultilinearExtension = MultilinearExtensionFast
build_mle_from_matrix = build_mle_from_matrix_fast
fold_table = fold_table_fast
build_eq_table = build_eq_table_fast
