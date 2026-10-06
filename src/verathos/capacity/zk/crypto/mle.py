"""
Multilinear Extensions (MLE) for sumcheck protocol.

A multilinear extension uniquely extends a function f: {0,1}^n -> F
to a polynomial f~: F^n -> F that agrees with f on the Boolean hypercube.

Key formula:
f~(x) = sum_{w in {0,1}^n} f(w) * eq(x, w)

where eq(x, w) = prod_i (x_i * w_i + (1 - x_i) * (1 - w_i))
"""

from typing import List, Optional
import math

from verathos.capacity.zk.crypto.field import mod_p, add_f, sub_f, mul_f


def eq_eval(x: List[int], w: List[int]) -> int:
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


def bits_to_int(bits: List[int]) -> int:
    """Convert list of bits to integer (LSB first)."""
    result = 0
    for i, b in enumerate(bits):
        if b:
            result |= (1 << i)
    return result


def int_to_bits(n: int, num_bits: int) -> List[int]:
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

    def __init__(self, evaluations: List[int], num_vars: Optional[int] = None):
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

    def evaluate(self, point: List[int]) -> int:
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

    def evaluate_partial(self, point: List[int]) -> "MultilinearExtension":
        """
        Partially evaluate MLE, fixing the first len(point) variables.

        Returns MLE in remaining variables.

        Args:
            point: Values for first k variables

        Returns:
            MLE in n-k remaining variables
        """
        if len(point) > self.num_vars:
            raise ValueError("Too many variables to fix")

        if len(point) == 0:
            return self

        # Use bookkeeping table approach for efficiency
        # Each variable fixing halves the table size
        table = list(self.evals)

        for r in point:
            new_size = len(table) // 2
            new_table = []

            for i in range(new_size):
                # table[2i] is value when this var = 0
                # table[2i+1] is value when this var = 1
                # new_table[i] = table[2i] + r * (table[2i+1] - table[2i])
                #              = table[2i] * (1 - r) + table[2i+1] * r
                t0 = table[2 * i]
                t1 = table[2 * i + 1]
                diff = sub_f(t1, t0)
                new_val = add_f(t0, mul_f(r, diff))
                new_table.append(new_val)

            table = new_table

        return MultilinearExtension(table, self.num_vars - len(point))

    def partial_evaluate_first(self, point: List[int]) -> List[int]:
        """
        Partially evaluate, fixing first len(point) variables.

        Returns flat table of remaining evaluations.

        This is used in sumcheck for building T_A, T_B tables.
        """
        mle = self.evaluate_partial(point)
        return mle.evals

    def get_table(self) -> List[int]:
        """Get raw evaluation table."""
        return list(self.evals)


def build_mle_from_matrix(matrix: List[List[int]]) -> MultilinearExtension:
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


def fold_table(table: List[int], r: int) -> List[int]:
    """
    Fold a table by fixing the first variable to r.

    new_table[i] = table[2i] * (1-r) + table[2i+1] * r

    Args:
        table: Current table (size 2^k)
        r: Value to fix first variable to

    Returns:
        Folded table (size 2^(k-1))
    """
    new_size = len(table) // 2
    new_table = []

    one_minus_r = sub_f(1, r)

    for i in range(new_size):
        t0 = table[2 * i]
        t1 = table[2 * i + 1]

        # t0 * (1-r) + t1 * r
        new_val = add_f(mul_f(t0, one_minus_r), mul_f(t1, r))
        new_table.append(new_val)

    return new_table


def build_eq_table(point: List[int]) -> List[int]:
    """
    Build table of eq(point, w) for all w in {0,1}^n.

    This is useful for batch evaluation.
    Uses LSB-first bit ordering to match int_to_bits.

    Args:
        point: Field elements [x_0, x_1, ..., x_{n-1}]

    Returns:
        Table where table[i] = eq(point, int_to_bits(i, n))
        i.e., table[i] = prod_j (x_j * b_j + (1-x_j) * (1-b_j))
              where b_j = (i >> j) & 1
    """
    n = len(point)
    size = 1 << n

    # Start with [1]
    table = [1]

    # For each variable (LSB first), double the table
    # Variable j corresponds to bit j in the index
    for j, xj in enumerate(point):
        one_minus_xj = sub_f(1, xj)
        new_table = [0] * (len(table) * 2)

        for i, v in enumerate(table):
            # Index i in old table maps to indices 2i and 2i+1...
            # No wait, with LSB-first, bit j being 0 vs 1 affects index by 2^j
            # For existing index i (in table of size 2^j):
            #   - If bit j = 0: new index = i (same lower bits)
            #   - If bit j = 1: new index = i + 2^j
            new_table[i] = mul_f(v, one_minus_xj)  # bit j = 0
            new_table[i + len(table)] = mul_f(v, xj)  # bit j = 1

        table = new_table

    return table
