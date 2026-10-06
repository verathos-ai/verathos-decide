"""Sumcheck verifier for H = sum_k T_A[k] * T_B[k], the reduction of one GEMM output-block claim."""


from verathos.capacity.zk.field import add_f, lagrange_interpolate_degree2, mul_f
from verathos.capacity.zk.transcript import Transcript
from verathos.capacity.zk.types import SumcheckProof


def sumcheck_verify(
    proof: SumcheckProof,
    transcript: Transcript,
) -> tuple[bool, int, list[int]]:
    """
    Verify a sumcheck proof.

    Args:
        proof: The sumcheck proof to verify
        transcript: Fiat-Shamir transcript (must match prover's)

    Returns:
        (valid, final_claim, challenges) where:
        - valid: True if proof passes sumcheck verification
        - final_claim: The final reduced claim (should equal final_A * final_B)
        - challenges: The random challenges used
    """
    current_claim = proof.claimed_sum
    challenges: list[int] = []

    for round_idx, round_data in enumerate(proof.rounds):
        e0, e1, e2 = round_data.e0, round_data.e1, round_data.e2

        # Check: e0 + e1 == current_claim
        if add_f(e0, e1) != current_claim:
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

