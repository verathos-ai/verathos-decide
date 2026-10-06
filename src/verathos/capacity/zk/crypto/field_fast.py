"""
Fast vectorized field arithmetic using NumPy.

For Mersenne prime p = 2^61 - 1, field operations are vectorized using
a uint64-only split-multiply technique that avoids Python object dtype.

Key optimization: mul_f_vec_mersenne() splits operands into 30/31-bit
halves so all partial products fit in uint64. This eliminates the
astype(object) bottleneck that dominated proof generation time.
"""

import numpy as np
import torch
from typing import List

# Mersenne prime p = 2^61 - 1
P = (1 << 61) - 1
P_NP = np.uint64(P)

# Constants for split multiplication
_MASK30 = np.uint64((1 << 30) - 1)
_MASK31 = np.uint64((1 << 31) - 1)
_SHIFT30 = np.uint64(30)
_SHIFT31 = np.uint64(31)
_SHIFT60 = np.uint64(60)
_SHIFT61 = np.uint64(61)
_ONE = np.uint64(1)

# Try to import native extension with CUDA support
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

# When True, skip GPU for proof field ops (frees GPU for parallel inference)
_PROOF_CPU_ONLY = False

def set_proof_cpu_only(cpu_only: bool):
    """Set whether proof generation should avoid GPU entirely."""
    global _PROOF_CPU_ONLY
    _PROOF_CPU_ONLY = cpu_only

# Try to import Numba for JIT compilation
_HAS_NUMBA = False
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    pass


# ============================================================================
# Core vectorized Mersenne field arithmetic (uint64-only, no object dtype)
# ============================================================================

def mul_f_vec_mersenne(a, b):
    """
    Vectorized field multiplication: a * b mod (2^61 - 1).

    Uses uint64-only arithmetic with 30/31-bit split to avoid overflow.
    Supports broadcasting (scalar x array, array x array, etc.).

    Math: Split a = a_hi * 2^30 + a_lo, b = b_hi * 2^30 + b_lo.
    All four partial products fit in uint64 (max 62 bits).
    Recombine using 2^61 = 1 (mod P) for efficient reduction.
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)

    a_lo = a & _MASK30
    a_hi = a >> _SHIFT30
    b_lo = b & _MASK30
    b_hi = b >> _SHIFT30

    # Partial products (all fit in uint64)
    ll = a_lo * b_lo         # max 60 bits
    lh = a_lo * b_hi         # max 61 bits
    hl = a_hi * b_lo         # max 61 bits
    hh = a_hi * b_hi         # max 62 bits

    # cross = lh + hl (max ~2^62, fits uint64)
    cross = lh + hl

    # Reduce cross * 2^30 mod P using 2^61 = 1:
    # cross * 2^30 = (cross >> 31) * 2^61 + (cross & MASK31) * 2^30
    #              = (cross >> 31) + (cross & MASK31) * 2^30
    cross_reduced = (cross >> _SHIFT31) + ((cross & _MASK31) << _SHIFT30)

    # Reduce hh * 2^60 mod P:
    # hh * 2^60 = (hh >> 1) * 2^61 + (hh & 1) * 2^60
    #           = (hh >> 1) + (hh & 1) * 2^60
    hh_reduced = (hh >> _ONE) + ((hh & _ONE) << _SHIFT60)

    # Total (max ~2^62.6, fits uint64)
    total = ll + cross_reduced + hh_reduced

    # Final Mersenne reduction
    result = (total >> _SHIFT61) + (total & P_NP)
    return np.where(result >= P_NP, result - P_NP, result)


def mersenne_vec_sum(arr):
    """
    Sum array of Mersenne field elements mod P using tree reduction.

    All values must be < P. Returns Python int.
    Fully vectorized: O(n) work in O(log n) vectorized passes.
    """
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return int(arr[0])

    current = arr.astype(np.uint64)
    while len(current) > 1:
        n = len(current)
        if n % 2 == 1:
            current = np.append(current, np.uint64(0))
        left = current[0::2]
        right = current[1::2]
        current = left + right  # Each sum < 2P < 2^62, fits uint64
        current = np.where(current >= P_NP, current - P_NP, current)

    return int(current[0])


# ============================================================================
# 2D Mersenne field sum - GPU-accelerated with CPU tree reduction fallback
# ============================================================================

def _mersenne_sum_axis0_tree(arr: np.ndarray) -> np.ndarray:
    """
    Tree reduction for summing 2D array along axis 0 in Mersenne field.
    O(log n) vectorized passes instead of O(n) sequential.

    Args:
        arr: [num_rows, num_cols] uint64 array, all values < P

    Returns:
        [num_cols] uint64 array with T[j] = sum_i arr[i,j] mod P
    """
    num_rows, num_cols = arr.shape

    if num_rows == 0:
        return np.zeros(num_cols, dtype=np.uint64)
    if num_rows == 1:
        return arr[0].copy()

    current = arr.astype(np.uint64)
    while current.shape[0] > 1:
        n = current.shape[0]
        if n % 2 == 1:
            # Pad with zero row
            current = np.vstack([current, np.zeros((1, num_cols), dtype=np.uint64)])
            n += 1
        left = current[:n//2]
        right = current[n//2:]
        current = left + right
        current = np.where(current >= P_NP, current - P_NP, current)

    return current[0]


def _mersenne_sum_axis1_tree(arr: np.ndarray) -> np.ndarray:
    """
    Tree reduction for summing 2D array along axis 1 in Mersenne field.

    Args:
        arr: [num_rows, num_cols] uint64 array, all values < P

    Returns:
        [num_rows] uint64 array with T[i] = sum_j arr[i,j] mod P
    """
    # Transpose and use axis 0 reduction
    return _mersenne_sum_axis0_tree(arr.T)


def mersenne_sum_axis0_fast(arr: np.ndarray) -> np.ndarray:
    """
    Sum 2D array along axis 0 in Mersenne field.
    Uses CUDA for large arrays, falls back to CPU tree reduction.

    Args:
        arr: [num_rows, num_cols] uint64 array

    Returns:
        [num_cols] uint64 array with T[j] = sum_i arr[i,j] mod P
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    num_rows, num_cols = arr.shape

    # Use CUDA for large arrays
    if _HAS_CUDA and num_rows * num_cols >= _CUDA_MIN_SIZE:
        arr_t = torch.from_numpy(arr.astype(np.int64)).cuda()
        result_t = _native.cuda_field_sum_axis0(arr_t, 0)
        return result_t.cpu().numpy().astype(np.uint64)

    # CPU tree reduction fallback
    return _mersenne_sum_axis0_tree(arr)


def mersenne_sum_axis1_fast(arr: np.ndarray) -> np.ndarray:
    """
    Sum 2D array along axis 1 in Mersenne field.
    Uses CUDA for large arrays, falls back to CPU tree reduction.

    Args:
        arr: [num_rows, num_cols] uint64 array

    Returns:
        [num_rows] uint64 array with T[i] = sum_j arr[i,j] mod P
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")

    num_rows, num_cols = arr.shape

    # Use CUDA for large arrays
    if _HAS_CUDA and num_rows * num_cols >= _CUDA_MIN_SIZE:
        arr_t = torch.from_numpy(arr.astype(np.int64)).cuda()
        result_t = _native.cuda_field_sum_axis1(arr_t, 0)
        return result_t.cpu().numpy().astype(np.uint64)

    # CPU tree reduction fallback
    return _mersenne_sum_axis1_tree(arr)


# ============================================================================
# Fused T table computation - GPU-native path avoiding CPU roundtrips
# ============================================================================

def compute_T_fixing_row_gpu(M: np.ndarray, eq: np.ndarray, col_size: int) -> np.ndarray:
    """
    Compute T[j] = sum_i M[i,j] * eq[i] entirely on GPU.

    Fused kernel: avoids separate multiply and reduce steps.
    ~10x faster than mul_f_vec_mersenne + mersenne_sum_axis0_fast.

    Args:
        M: [num_rows, num_cols] matrix (uint64)
        eq: [eq_len] eq table values (uint64)
        col_size: Desired output size (will pad if needed)

    Returns:
        [col_size] T table values (uint64)
    """
    if not _HAS_CUDA or _PROOF_CPU_ONLY:
        # CPU fallback: separate multiply + reduce (native C++ or NumPy)
        eq_len = min(M.shape[0], len(eq))
        M_sub = M[:eq_len, :col_size]
        eq_col = eq[:eq_len, np.newaxis]
        products = mul_f_vec_mersenne(M_sub, eq_col)
        return mersenne_sum_axis0_fast(products)

    # Convert to GPU tensors
    M_t = torch.from_numpy(M.astype(np.int64)).cuda()
    eq_t = torch.from_numpy(eq.astype(np.int64)).cuda()

    # Fused kernel computation
    T_t = _native.cuda_compute_T_fixing_row(M_t, eq_t, 0)
    T = T_t.cpu().numpy().astype(np.uint64)

    # Pad to col_size if needed
    if len(T) < col_size:
        T_padded = np.zeros(col_size, dtype=np.uint64)
        T_padded[:len(T)] = T
        return T_padded

    return T[:col_size]


def compute_T_fixing_col_gpu(M: np.ndarray, eq: np.ndarray, row_size: int) -> np.ndarray:
    """
    Compute T[i] = sum_j M[i,j] * eq[j] entirely on GPU.

    Fused kernel: avoids separate multiply and reduce steps.

    Args:
        M: [num_rows, num_cols] matrix (uint64)
        eq: [eq_len] eq table values (uint64)
        row_size: Desired output size (will pad if needed)

    Returns:
        [row_size] T table values (uint64)
    """
    if not _HAS_CUDA or _PROOF_CPU_ONLY:
        # CPU fallback: separate multiply + reduce (native C++ or NumPy)
        eq_len = min(M.shape[1], len(eq))
        M_sub = M[:row_size, :eq_len]
        eq_row = eq[:eq_len][np.newaxis, :]
        products = mul_f_vec_mersenne(M_sub, eq_row)
        return mersenne_sum_axis1_fast(products)

    # Convert to GPU tensors
    M_t = torch.from_numpy(M.astype(np.int64)).cuda()
    eq_t = torch.from_numpy(eq.astype(np.int64)).cuda()

    # Fused kernel computation
    T_t = _native.cuda_compute_T_fixing_col(M_t, eq_t, 0)
    T = T_t.cpu().numpy().astype(np.uint64)

    # Pad to row_size if needed
    if len(T) < row_size:
        T_padded = np.zeros(row_size, dtype=np.uint64)
        T_padded[:len(T)] = T
        return T_padded

    return T[:row_size]


def evaluate_mle_matrix_point_fast(M: np.ndarray, r_row: list, r_col: list) -> int:
    """
    Evaluate MLE of matrix at point (r_row, r_col) WITHOUT building full table.

    claimed_sum = sum_i sum_j M[i,j] * eq(r_row, i) * eq(r_col, j)
                = (eq_row @ M) @ eq_col

    This is O(rows * cols) like building full MLE, but uses two BLAS-like
    operations which are much faster (~100x for large matrices).

    Args:
        M: [num_rows, num_cols] matrix (uint64 field elements)
        r_row: Random point for row dimension
        r_col: Random point for column dimension

    Returns:
        Field element: M_tilde(r_row, r_col)
    """
    rows, cols = M.shape

    # Build eq tables (small: 2^|r_row| and 2^|r_col|)
    eq_row = build_eq_table_vec(r_row)  # [padded_rows]
    eq_col = build_eq_table_vec(r_col)  # [padded_cols]

    # Truncate to actual matrix size
    eq_row = eq_row[:rows]
    eq_col = eq_col[:cols]

    if _HAS_CUDA and rows * cols >= _CUDA_MIN_SIZE:
        # GPU path: compute (eq_row @ M) on GPU, final dot product on CPU
        M_t = torch.from_numpy(M.astype(np.int64)).cuda()
        eq_row_t = torch.from_numpy(eq_row.astype(np.int64)).cuda()

        # Step 1: tmp[j] = sum_i eq_row[i] * M[i,j] in Mersenne field (GPU)
        tmp_t = _native.cuda_compute_T_fixing_row(M_t, eq_row_t, 0)  # [cols]
        tmp = tmp_t.cpu().numpy().astype(np.uint64)[:cols]

        # Step 2: result = sum_j tmp[j] * eq_col[j] (small 1D, CPU is fast)
        products = mul_f_vec_mersenne(tmp, eq_col)
        return int(mersenne_vec_sum(products))

    # CPU fallback: vectorized operations
    # Step 1: tmp[j] = sum_i eq_row[i] * M[i,j]
    # Use our fused function
    tmp = compute_T_fixing_row_gpu(M, eq_row, cols)[:cols]

    # Step 2: result = sum_j tmp[j] * eq_col[j]
    products = mul_f_vec_mersenne(tmp, eq_col)
    result = mersenne_vec_sum(products)

    return int(result)


def evaluate_mle_tensor_point_gpu(tensor: torch.Tensor, r_row: list, r_col: list,
                                   padded_rows: int, padded_cols: int) -> int:
    """
    Evaluate MLE of a tensor at point (r_row, r_col) entirely on GPU.

    This is the fastest path - avoids CPU round-trip entirely.
    Converts tensor to field on GPU and computes claimed_sum in one pass.

    Args:
        tensor: GPU tensor [rows, cols] (any dtype)
        r_row: Random point for row dimension
        r_col: Random point for column dimension
        padded_rows: Padded row dimension (power of 2)
        padded_cols: Padded col dimension (power of 2)

    Returns:
        Field element: M_tilde(r_row, r_col)
    """
    if not _HAS_CUDA or not tensor.is_cuda:
        # Fallback: convert to numpy and use numpy path
        M = tensor_to_field_vec(tensor, padded_rows, padded_cols)
        return evaluate_mle_matrix_point_fast(M, r_row, r_col)

    rows, cols = tensor.shape

    # Build eq tables (small, fast)
    eq_row = build_eq_table_vec(r_row)[:rows]
    eq_col = build_eq_table_vec(r_col)[:cols]

    # Convert tensor to field on GPU (handles negatives properly)
    # For small integers: field_val = val if val >= 0 else val + P
    M_gpu = tensor.to(torch.int64)  # Keep on GPU
    # Handle negatives: add P where < 0
    M_gpu = torch.where(M_gpu < 0, M_gpu + P, M_gpu)

    # Upload eq tables to GPU
    eq_row_t = torch.from_numpy(eq_row.astype(np.int64)).cuda()
    eq_col_t = torch.from_numpy(eq_col.astype(np.int64)).cuda()

    # Step 1: tmp[j] = sum_i M[i,j] * eq_row[i] using fused kernel
    tmp_t = _native.cuda_compute_T_fixing_row(M_gpu, eq_row_t, 0)  # [cols]

    # Step 2: result = sum_j tmp[j] * eq_col[j]
    # Transfer to CPU for proper Mersenne field arithmetic
    # (Step 2 is a small 1D operation, CPU is fine)
    tmp = tmp_t.cpu().numpy().astype(np.uint64)[:cols]
    products = mul_f_vec_mersenne(tmp, eq_col)
    return int(mersenne_vec_sum(products))


# ============================================================================
# Standard field operations (updated to use mul_f_vec_mersenne)
# ============================================================================

def mod_p_vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorized Mersenne reduction for values < 2^63.
    """
    lo = x & P_NP
    hi = x >> _SHIFT61
    r = lo + hi
    return np.where(r >= P_NP, r - P_NP, r)


def add_f_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized field addition."""
    r = a.astype(np.uint64) + b.astype(np.uint64)
    return np.where(r >= P_NP, r - P_NP, r)


def sub_f_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized field subtraction."""
    return np.where(a >= b, a - b, P_NP - (b - a))


def mul_f_scalar(a: int, b: int) -> int:
    """Scalar field multiplication with full precision."""
    prod = a * b
    lo = prod & P
    hi = prod >> 61
    r = lo + hi
    if r >= P:
        r -= P
    return r


def mul_f_vec_safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized field multiplication using uint64 split-multiply.

    Drop-in replacement for the old object-dtype version.
    """
    return mul_f_vec_mersenne(a, b)


def mul_f_vec_approx(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast approximate multiplication for values where overflow is unlikely.

    Only use when a * b < 2^64 is guaranteed (small values).
    """
    prod = a.astype(np.uint64) * b.astype(np.uint64)
    return mod_p_vec(prod)


def dot_product_field(a: np.ndarray, b: np.ndarray) -> int:
    """
    Compute sum_i a[i] * b[i] in the Mersenne field.

    Uses vectorized multiplication + tree-reduction sum.
    """
    if len(a) == 0:
        return 0

    products = mul_f_vec_mersenne(a, b)
    return mersenne_vec_sum(products)


def batch_mul_reduce(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply arrays and reduce each element mod p.
    """
    return mul_f_vec_mersenne(a, b)


# ============================================================================
# Optimized table operations
# ============================================================================

def fold_table_vec(table: np.ndarray, r: int) -> np.ndarray:
    """
    Vectorized table folding: new[i] = table[2i]*(1-r) + table[2i+1]*r
    """
    n = len(table) // 2
    if n == 0:
        return np.array([], dtype=np.uint64)

    one_minus_r = (P - r + 1) if r > 0 else 1
    if one_minus_r >= P:
        one_minus_r -= P

    t0 = table[0::2]  # Even indices
    t1 = table[1::2]  # Odd indices

    # Vectorized Mersenne multiply (no object dtype)
    term1 = mul_f_vec_mersenne(t0, np.uint64(one_minus_r))
    term2 = mul_f_vec_mersenne(t1, np.uint64(r))

    # Add with single reduction (both terms < P, sum < 2P < 2^62)
    result = term1 + term2
    return np.where(result >= P_NP, result - P_NP, result)


def build_eq_table_vec(point: List[int]) -> np.ndarray:
    """
    Vectorized eq table construction.

    Build table[i] = eq(point, int_to_bits(i, n)) for all i.
    Uses CUDA for large tables when available.
    """
    n = len(point)
    size = 1 << n

    # Use CUDA for large tables
    if _HAS_CUDA and size >= _CUDA_MIN_SIZE:
        point_t = torch.tensor(point, dtype=torch.int64).cuda()
        result_t = _native.cuda_build_eq_table(point_t, 0)
        return result_t.cpu().numpy().astype(np.uint64)

    # Use native C++ if available
    if _HAS_NATIVE:
        return _native.native_build_eq_table(np.array(point, dtype=np.uint64))

    # Vectorized NumPy fallback using uint64 Mersenne multiply
    table = np.array([1], dtype=np.uint64)

    for j, xj in enumerate(point):
        one_minus_xj = np.uint64((P - xj + 1) if xj > 0 else 1)
        if one_minus_xj >= P_NP:
            one_minus_xj -= P_NP
        xj_u = np.uint64(xj)

        old_size = len(table)
        new_table = np.empty(old_size * 2, dtype=np.uint64)

        # Vectorized multiply by scalar (broadcasts correctly)
        new_table[:old_size] = mul_f_vec_mersenne(table, one_minus_xj)  # bit j = 0
        new_table[old_size:] = mul_f_vec_mersenne(table, xj_u)          # bit j = 1

        table = new_table

    return table


def compute_round_evals_vec(T_A: np.ndarray, T_B: np.ndarray) -> tuple:
    """
    Vectorized sumcheck round evaluation.

    Computes (e0, e1, e2) for one sumcheck round.
    """
    # Extract even and odd indexed elements
    A0 = T_A[0::2]
    A1 = T_A[1::2]
    B0 = T_B[0::2]
    B1 = T_B[1::2]

    # e0 = sum_i A0[i] * B0[i]
    e0 = dot_product_field(A0, B0)

    # e1 = sum_i A1[i] * B1[i]
    e1 = dot_product_field(A1, B1)

    # e2 = sum_i (2*A1[i] - A0[i]) * (2*B1[i] - B0[i])
    # Use vectorized field arithmetic (no object dtype)
    A2 = add_f_vec(A1, A1)      # 2*A1
    A2 = sub_f_vec(A2, A0)       # 2*A1 - A0
    B2 = add_f_vec(B1, B1)      # 2*B1
    B2 = sub_f_vec(B2, B0)       # 2*B1 - B0

    e2 = dot_product_field(A2, B2)

    return int(e0), int(e1), int(e2)


# ============================================================================
# Conversion utilities
# ============================================================================

# Numba JIT kernel for tensor->field conversion (fused copy + neg handling)
if _HAS_NUMBA:
    @njit(parallel=True, cache=True, fastmath=False)
    def _tensor_to_field_numba(arr_int64, result_u64, P_val):
        """
        Numba JIT kernel: convert int64 array to uint64 field elements.
        Fuses copy + negative handling into single pass with parallel execution.
        """
        rows, cols = arr_int64.shape
        for i in prange(rows):
            for j in range(cols):
                val = arr_int64[i, j]
                if val < 0:
                    # For negative int64 v: reinterpret as uint64 gives 2^64 - |v|
                    # Adding P: (2^64 - |v|) + P = P - |v| (mod 2^64)
                    result_u64[i, j] = np.uint64(val) + P_val
                else:
                    result_u64[i, j] = np.uint64(val)

    @njit(parallel=True, cache=True, fastmath=False)
    def _tensor_to_field_padded_numba(arr_int64, result_u64, P_val, src_rows, src_cols):
        """
        Numba JIT kernel: convert with padding (zeros in padded region).
        """
        rows, cols = result_u64.shape
        for i in prange(rows):
            for j in range(cols):
                if i < src_rows and j < src_cols:
                    val = arr_int64[i, j]
                    if val < 0:
                        result_u64[i, j] = np.uint64(val) + P_val
                    else:
                        result_u64[i, j] = np.uint64(val)
                else:
                    result_u64[i, j] = np.uint64(0)


def tensor_to_field_vec(tensor, padded_rows: int, padded_cols: int) -> np.ndarray:
    """
    Convert 2D tensor to field elements with padding.

    Returns a 2D numpy array of uint64 field elements.
    Handles negative values by adding P (values are small compared to P).

    Fast paths (in priority order):
      1. CUDA kernel — keeps tensor on GPU, converts there, then transfers result
      2. Numba JIT — 3-5x faster than NumPy for large arrays
      3. NumPy fallback
    """
    # CUDA fast path: convert on GPU, then transfer result to CPU
    if _HAS_CUDA and not _PROOF_CPU_ONLY and isinstance(tensor, torch.Tensor) and tensor.is_cuda:
        t = tensor.to(torch.int64) if tensor.dtype != torch.int64 else tensor.contiguous()
        result_gpu = _native.cuda_tensor_to_field(t, padded_rows, padded_cols, 0)
        return result_gpu.cpu().numpy().view(np.uint64)

    if isinstance(tensor, torch.Tensor):
        arr = tensor.numpy() if tensor.is_cpu else tensor.cpu().numpy()
    else:
        arr = np.asarray(tensor)

    rows, cols = arr.shape
    needs_padding = (padded_rows != rows or padded_cols != cols)

    # Ensure contiguous int64 input
    if arr.dtype != np.int64 or not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr, dtype=np.int64)

    # Use Numba JIT if available (much faster for large arrays)
    if _HAS_NUMBA:
        result_u64 = np.empty((padded_rows, padded_cols), dtype=np.uint64)
        if needs_padding:
            _tensor_to_field_padded_numba(arr, result_u64, P_NP, rows, cols)
        else:
            _tensor_to_field_numba(arr, result_u64, P_NP)
        return result_u64

    # NumPy fallback
    if needs_padding:
        result = np.zeros((padded_rows, padded_cols), dtype=np.int64)
        result[:rows, :cols] = arr[:rows, :cols]
    else:
        result = arr.copy()

    result_u64 = result.view(np.uint64)
    if result.min() < 0:
        np.add(result_u64, P_NP, where=result < 0, out=result_u64)

    return result_u64


def list_to_numpy(lst: List[int]) -> np.ndarray:
    """Convert list of field elements to numpy array."""
    return np.array(lst, dtype=np.uint64)


def numpy_to_list(arr: np.ndarray) -> List[int]:
    """Convert numpy array to list of field elements."""
    return arr.astype(int).tolist()


# ============================================================================
# Numba JIT warmup - compile kernels at import time to avoid latency during proof
# ============================================================================

def _warmup_numba_kernels():
    """Pre-compile Numba kernels with representative array shapes."""
    if not _HAS_NUMBA:
        return

    # Small test arrays - just enough to trigger JIT compilation
    # The compiled code works for any size arrays of the same types
    test_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    result = np.empty((2, 3), dtype=np.uint64)

    # Warm up non-padded kernel
    _tensor_to_field_numba(test_arr, result, P_NP)

    # Warm up padded kernel
    result_pad = np.empty((4, 4), dtype=np.uint64)
    _tensor_to_field_padded_numba(test_arr, result_pad, P_NP, 2, 3)


# Run warmup at import time
_warmup_numba_kernels()
