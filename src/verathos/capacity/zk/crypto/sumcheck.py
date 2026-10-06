"""
Sumcheck protocol for verifying polynomial sums.

The sumcheck protocol proves:
    H = sum_{x in {0,1}^n} g(x)

where g is a low-degree polynomial, in O(n) rounds.

For GEMM verification, we prove:
    H = sum_k T_A[k] * T_B[k]

where T_A, T_B are derived from matrix MLEs.
"""

import logging
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from verathos.capacity.zk.crypto.field import add_f, sub_f, mul_f, lagrange_interpolate_degree2
from verathos.capacity.zk.crypto.mle import fold_table
from verathos.capacity.zk.crypto.transcript import Transcript
from verathos.capacity.zk.types import SumcheckRound, SumcheckProof


@dataclass
class SumcheckProverState:
    """Internal state for sumcheck prover."""

    T_A: List[int]  # Current table for first polynomial
    T_B: List[int]  # Current table for second polynomial
    num_vars: int  # Number of remaining variables
    rounds: List[SumcheckRound]
    challenges: List[int]


def compute_round_evals(T_A: List[int], T_B: List[int]) -> Tuple[int, int, int]:
    """
    Compute g(0), g(1), g(2) for the current sumcheck round.

    g(t) = sum_{x2,...,xn in {0,1}} T_A(t, x2,...) * T_B(t, x2,...)

    For the sum over x1:
    - g(0) = sum where x1=0
    - g(1) = sum where x1=1
    - g(2) = extrapolated via degree-2 interpolation

    Since T_A * T_B is degree 2 in each variable, g is degree 2.
    """
    n = len(T_A)
    half = n // 2

    # g(0) = sum over pairs (T_A[2i], T_B[2i])
    e0 = 0
    for i in range(half):
        e0 = add_f(e0, mul_f(T_A[2 * i], T_B[2 * i]))

    # g(1) = sum over pairs (T_A[2i+1], T_B[2i+1])
    e1 = 0
    for i in range(half):
        e1 = add_f(e1, mul_f(T_A[2 * i + 1], T_B[2 * i + 1]))

    # g(2) = sum over pairs interpolated at t=2
    # For each pair, we have:
    #   T_A at t=0: T_A[2i]
    #   T_A at t=1: T_A[2i+1]
    #   T_A at t=2: 2*T_A[2i+1] - T_A[2i] (linear extrapolation)
    #
    # Same for T_B, and we need T_A(2) * T_B(2)

    e2 = 0
    for i in range(half):
        # Linear extrapolation: f(2) = 2*f(1) - f(0)
        a0 = T_A[2 * i]
        a1 = T_A[2 * i + 1]
        a2 = sub_f(mul_f(2, a1), a0)

        b0 = T_B[2 * i]
        b1 = T_B[2 * i + 1]
        b2 = sub_f(mul_f(2, b1), b0)

        e2 = add_f(e2, mul_f(a2, b2))

    return e0, e1, e2


def sumcheck_prove(
    T_A: List[int],
    T_B: List[int],
    claimed_sum: int,
    transcript: Transcript,
) -> Tuple[SumcheckProof, List[int]]:
    """
    Run the sumcheck prover.

    Proves: claimed_sum = sum_{x in {0,1}^n} T_A[x] * T_B[x]

    Args:
        T_A: Table of first polynomial evaluations (size 2^n)
        T_B: Table of second polynomial evaluations (size 2^n)
        claimed_sum: The sum we claim equals the product sum
        transcript: Fiat-Shamir transcript

    Returns:
        (proof, challenges) where challenges are the random points used
    """
    if len(T_A) != len(T_B):
        raise ValueError("T_A and T_B must have same length")

    n = len(T_A)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("Table size must be power of 2")

    num_vars = n.bit_length() - 1

    rounds: List[SumcheckRound] = []
    challenges: List[int] = []

    cur_A = list(T_A)
    cur_B = list(T_B)
    current_claim = claimed_sum

    for round_idx in range(num_vars):
        # Compute g(0), g(1), g(2) for this round
        e0, e1, e2 = compute_round_evals(cur_A, cur_B)

        rounds.append(SumcheckRound(e0=e0, e1=e1, e2=e2))

        # Absorb into transcript
        transcript.absorb(b"round", round_idx)
        transcript.absorb(b"e0", e0)
        transcript.absorb(b"e1", e1)
        transcript.absorb(b"e2", e2)

        # Get challenge
        r = transcript.squeeze(b"challenge")
        challenges.append(r)

        # Fold tables for next round
        cur_A = fold_table(cur_A, r)
        cur_B = fold_table(cur_B, r)

    # Final values
    final_A = cur_A[0] if cur_A else 0
    final_B = cur_B[0] if cur_B else 0

    proof = SumcheckProof(
        claimed_sum=claimed_sum,
        rounds=rounds,
        final_A=final_A,
        final_B=final_B,
    )

    return proof, challenges


def sumcheck_verify(
    proof: SumcheckProof,
    transcript: Transcript,
) -> Tuple[bool, int, List[int]]:
    """
    Verify a sumcheck proof.

    Args:
        proof: The sumcheck proof to verify
        transcript: Fiat-Shamir transcript (must match prover's)

    Returns:
        (valid, final_claim, challenges) where:
        - valid: True if proof passes sumcheck verification
        - final_claim: The final reduced claim (should equal final_A * final_B)
        - challenges: The random challenges used (for debugging/further verification)
    """
    current_claim = proof.claimed_sum
    challenges: List[int] = []

    for round_idx, round_data in enumerate(proof.rounds):
        e0, e1, e2 = round_data.e0, round_data.e1, round_data.e2

        # Check: e0 + e1 == current_claim
        if add_f(e0, e1) != current_claim:
            logger.warning(
                "VERIFY-FAIL Round %d/%d: e0+e1=%s != claim=%s "
                "e0=%s e1=%s e2=%s type(e0)=%s type(claim)=%s "
                "claimed_sum=%s final_A=%s final_B=%s",
                round_idx, len(proof.rounds), add_f(e0, e1), current_claim,
                e0, e1, e2, type(e0).__name__, type(current_claim).__name__,
                proof.claimed_sum, proof.final_A, proof.final_B,
            )
            return False, 0, challenges

        # Absorb into transcript (must match prover)
        transcript.absorb(b"round", round_idx)
        transcript.absorb(b"e0", e0)
        transcript.absorb(b"e1", e1)
        transcript.absorb(b"e2", e2)

        # Get challenge
        r = transcript.squeeze(b"challenge")
        challenges.append(r)

        # Update claim: current_claim = g(r)
        current_claim = lagrange_interpolate_degree2(e0, e1, e2, r)

    # Final check: current_claim should equal final_A * final_B
    expected_final = mul_f(proof.final_A, proof.final_B)

    if current_claim != expected_final:
        return False, current_claim, challenges

    return True, current_claim, challenges


def sumcheck_verify_with_final_check(
    proof: SumcheckProof,
    transcript: Transcript,
    verify_final: callable,
) -> Tuple[bool, str]:
    """
    Verify sumcheck proof with custom final value verification.

    Args:
        proof: The sumcheck proof
        transcript: Fiat-Shamir transcript
        verify_final: Function(final_A, final_B, challenges) -> (bool, message)
                     Called to verify the final polynomial evaluations

    Returns:
        (valid, message)
    """
    valid, final_claim, challenges = sumcheck_verify(proof, transcript)

    if not valid:
        return False, "Sumcheck round verification failed"

    # Verify final values using custom function
    final_ok, msg = verify_final(proof.final_A, proof.final_B, challenges)

    if not final_ok:
        return False, f"Final verification failed: {msg}"

    return True, "Sumcheck verified"


# Utility functions for GEMM-specific sumcheck

def build_gemm_tables(
    X_mle_partial: List[int],
    W_mle_partial: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Build sumcheck tables for GEMM verification.

    Given MLEs partially evaluated at random points (r_i, r_j),
    we have:
        T_A[k] = X_tilde(r_i, k)
        T_B[k] = W_tilde(k, r_j)

    The inner product sum_k T_A[k] * T_B[k] equals Y_tilde(r_i, r_j).

    Args:
        X_mle_partial: X MLE partially evaluated at row point
        W_mle_partial: W MLE partially evaluated at column point

    Returns:
        (T_A, T_B) tables for sumcheck
    """
    # Tables should already be the right format from partial evaluation
    return list(X_mle_partial), list(W_mle_partial)


def compute_gemm_claimed_sum(
    Y_mle: "MultilinearExtension",
    r_i: List[int],
    r_j: List[int],
) -> int:
    """
    Compute the claimed sum for GEMM sumcheck.

    The claimed sum is Y_tilde(r_i, r_j) where Y is the output matrix.

    Args:
        Y_mle: MLE of output matrix Y
        r_i: Random point for row index
        r_j: Random point for column index

    Returns:
        Y_tilde(r_i || r_j)
    """

    point = list(r_i) + list(r_j)
    return Y_mle.evaluate(point)
