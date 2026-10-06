"""
Multilinear Extensions (MLE) for sumcheck protocol.

A multilinear extension uniquely extends a function f: {0,1}^n -> F
to a polynomial f~: F^n -> F that agrees with f on the Boolean hypercube.

Key formula:
f~(x) = sum_{w in {0,1}^n} f(w) * eq(x, w)

where eq(x, w) = prod_i (x_i * w_i + (1 - x_i) * (1 - w_i))
"""

import math

from verathos.capacity.zk.field import add_f, mod_p, mul_f, sub_f


def eq_eval(x: list[int], w: list[int]) -> int:
    """
    Evaluate eq(x, w) = prod_i (x_i * w_i + (1 - x_i) * (1 - w_i)).

    For binary w_i:
    - If w_i = 1: x_i
    - If w_i = 0: 1 - x_i

    Args:
        x: Field elements (evaluation point)
        w: Binary values (0 or 1)

    Returns:
        eq(x, w) as field element
    """
    if len(x) != len(w):
        raise ValueError("x and w must have same length")

    result = 1
    for xi, wi in zip(x, w):
        if wi == 1:
            term = xi
        else:
            term = sub_f(1, xi)
        result = mul_f(result, term)

    return result


def int_to_bits(n: int, num_bits: int) -> list[int]:
    """Convert integer to list of bits (LSB first)."""
    return [(n >> i) & 1 for i in range(num_bits)]


class MultilinearExtension:
    """
    Multilinear extension of a function over the Boolean hypercube.

    Stores evaluations at all 2^n points and supports:
    - Evaluation at arbitrary field points
    - Partial evaluation (fixing some variables)
    - Efficient bookkeeping table for sumcheck
    """

    def __init__(self, evaluations: list[int], num_vars: int | None = None):
        """
        Create MLE from evaluations at all 2^n Boolean points.

        Args:
            evaluations: List of f(w) for all w in {0,1}^n (in lex order)
            num_vars: Number of variables (inferred from len if not given)
        """
        if num_vars is None:
            num_vars = int(math.log2(len(evaluations)))
            if 1 << num_vars != len(evaluations):
                raise ValueError("evaluations length must be power of 2")

        self.num_vars = num_vars
        self.evals = [mod_p(e) for e in evaluations]

    def evaluate(self, point: list[int]) -> int:
        """
        Evaluate MLE at arbitrary field point.

        f~(x) = sum_{w in {0,1}^n} f(w) * eq(x, w)

        Args:
            point: List of n field elements

        Returns:
            f~(point)
        """
        if len(point) != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} variables, got {len(point)}")

        result = 0
        for i, ev in enumerate(self.evals):
            w = int_to_bits(i, self.num_vars)
            eq_w = eq_eval(point, w)
            result = add_f(result, mul_f(ev, eq_w))

        return result

def build_mle_from_matrix(matrix: list[list[int]]) -> MultilinearExtension:
    """
    Build MLE from 2D matrix.

    Matrix is interpreted as function M(i, j) with:
    - i in {0,1}^m (row index)
    - j in {0,1}^n (column index)

    The MLE has m+n variables, where:
    - First m variables encode the row index
    - Last n variables encode the column index

    Storage order matches bit interpretation (LSB-first):
    For evaluate([row_bits, col_bits]), we need index = row + col * padded_rows

    Args:
        matrix: 2D list of field elements

    Returns:
        MLE with row variables first, then column variables
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    # Pad to powers of 2
    m = max(1, int(math.ceil(math.log2(rows)))) if rows > 0 else 1
    n = max(1, int(math.ceil(math.log2(cols)))) if cols > 0 else 1

    padded_rows = 1 << m
    padded_cols = 1 << n

    # Build flat evaluation table in correct order for bit interpretation
    # index = row + col * padded_rows (column-major for LSB-first bits)
    evals = [0] * (padded_rows * padded_cols)
    for i in range(rows):
        for j in range(cols):
            idx = i + j * padded_rows
            evals[idx] = mod_p(matrix[i][j])

    return MultilinearExtension(evals, m + n)


