"""
Field arithmetic for F_p where p = 2^61 - 1 (Mersenne prime).

Optimized for:
- Fast reduction via bit operations
- Compatibility with INT8/INT32 tensor values
- Efficient batch operations
"""

from typing import List
import numpy as np

# Mersenne prime: p = 2^61 - 1
P = (1 << 61) - 1

# Useful constants
P_MASK = P  # Same as P for masking


def mod_p(x: int) -> int:
    """
    Reduce x modulo p using Mersenne prime optimization.

    For Mersenne prime p = 2^k - 1:
    x mod p = (x >> k) + (x & p)

    If result >= p, subtract p.
    """
    if x < 0:
        # Handle negative numbers
        x = x % P
        if x < 0:
            x += P
        return x

    # Fast Mersenne reduction
    lo = x & P
    hi = x >> 61
    r = lo + hi

    # One more reduction if needed
    if r >= P:
        r -= P

    return r


def add_f(a: int, b: int) -> int:
    """Add two field elements."""
    r = a + b
    if r >= P:
        r -= P
    return r


def sub_f(a: int, b: int) -> int:
    """Subtract two field elements."""
    r = a - b
    if r < 0:
        r += P
    return r


def mul_f(a: int, b: int) -> int:
    """Multiply two field elements."""
    return mod_p(a * b)


def neg_f(a: int) -> int:
    """Negate a field element."""
    if a == 0:
        return 0
    return P - a


def pow_f(base: int, exp: int) -> int:
    """
    Compute base^exp mod p using square-and-multiply.
    """
    if exp == 0:
        return 1
    if exp == 1:
        return base

    result = 1
    base = mod_p(base)

    while exp > 0:
        if exp & 1:
            result = mul_f(result, base)
        base = mul_f(base, base)
        exp >>= 1

    return result


def inv_f(a: int) -> int:
    """
    Compute multiplicative inverse using Fermat's little theorem.

    a^(-1) = a^(p-2) mod p
    """
    if a == 0:
        raise ValueError("Cannot invert zero")
    return pow_f(a, P - 2)


def batch_mul_f(a: List[int], b: List[int]) -> List[int]:
    """Multiply two lists element-wise."""
    return [mul_f(ai, bi) for ai, bi in zip(a, b)]


def batch_add_f(a: List[int], b: List[int]) -> List[int]:
    """Add two lists element-wise."""
    return [add_f(ai, bi) for ai, bi in zip(a, b)]


# Lagrange interpolation coefficients for degree-2 polynomial
# at points {0, 1, 2}
def lagrange_interpolate_degree2(e0: int, e1: int, e2: int, t: int) -> int:
    """
    Interpolate degree-2 polynomial through (0, e0), (1, e1), (2, e2) at point t.

    L_0(t) = (t-1)(t-2)/2
    L_1(t) = t(2-t) = -t^2 + 2t
    L_2(t) = t(t-1)/2

    g(t) = e0*L_0(t) + e1*L_1(t) + e2*L_2(t)
    """
    # Compute in field
    t = mod_p(t)

    # L_0(t) = (t-1)(t-2)/2
    t_minus_1 = sub_f(t, 1)
    t_minus_2 = sub_f(t, 2)
    l0_num = mul_f(t_minus_1, t_minus_2)
    inv_2 = inv_f(2)
    l0 = mul_f(l0_num, inv_2)

    # L_1(t) = t(2-t) = 2t - t^2
    two_t = mul_f(2, t)
    t_sq = mul_f(t, t)
    l1 = sub_f(two_t, t_sq)

    # L_2(t) = t(t-1)/2
    l2_num = mul_f(t, t_minus_1)
    l2 = mul_f(l2_num, inv_2)

    # g(t) = e0*L_0 + e1*L_1 + e2*L_2
    term0 = mul_f(e0, l0)
    term1 = mul_f(e1, l1)
    term2 = mul_f(e2, l2)

    result = add_f(term0, add_f(term1, term2))
    return result


# Conversion utilities
def int8_to_field(x: int) -> int:
    """Convert signed int8 to field element."""
    if x < 0:
        return P + x  # Wrap negative values
    return x


def int32_to_field(x: int) -> int:
    """Convert signed int32 to field element."""
    return mod_p(x)


def tensor_to_field_list(tensor: "torch.Tensor") -> List[int]:
    """Convert a tensor to a list of field elements."""
    flat = tensor.flatten().tolist()
    return [mod_p(int(x)) for x in flat]


def numpy_to_field_array(arr: np.ndarray) -> np.ndarray:
    """
    Convert numpy array to field elements.

    Uses numpy operations for speed, then ensures values are in [0, P).
    """
    # Handle negative values
    result = arr.astype(np.int64)

    # Mask negative values
    neg_mask = result < 0
    result[neg_mask] = result[neg_mask] + P

    # Reduce any values >= P (shouldn't happen for int8/int32, but safety)
    large_mask = result >= P
    if large_mask.any():
        result[large_mask] = result[large_mask] % P

    return result


# Batch field operations using numpy for speed
def numpy_add_f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise field addition."""
    result = a.astype(np.int64) + b.astype(np.int64)
    result[result >= P] -= P
    return result


def numpy_sub_f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise field subtraction."""
    result = a.astype(np.int64) - b.astype(np.int64)
    result[result < 0] += P
    return result


def numpy_mul_f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Element-wise field multiplication.

    Note: This can overflow for large values. For production,
    we'd need either uint128 or chunk-based multiplication.
    For INT8 * INT8 -> INT16 range, this is safe.
    """
    result = (a.astype(np.int64) * b.astype(np.int64)) % P
    return result
