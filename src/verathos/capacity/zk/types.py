"""
Core proof types for ZkLLM.

These types are shared across inference (verallm) and training (trainllm).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SpotCheck:
    """A single spot check entry."""

    row: int
    col: int
    value: int  # Field element


@dataclass
class SpotCheckWithProof:
    """Spot check with Merkle proof for verifiable binding.

    Used for X (activation) spot checks where the validator doesn't have
    the actual tensor values - only the Merkle commitment. The miner must
    provide a Merkle proof that the claimed value is in the committed tensor.
    """

    row: int
    col: int
    value: int  # Field element
    merkle_path: "MerklePath"  # Proof that value is in committed tensor
    leaf_data: bytes  # The actual leaf data (for verification)


@dataclass
class SumcheckRound:
    """One round of sumcheck proof."""

    e0: int  # g(0)
    e1: int  # g(1)
    e2: int  # g(2)


@dataclass
class SumcheckProof:
    """Complete sumcheck proof."""

    claimed_sum: int
    rounds: List[SumcheckRound]
    final_A: int  # Final evaluation of T_A
    final_B: int  # Final evaluation of T_B


@dataclass
class MerklePath:
    """Authentication path in Merkle tree."""

    leaf_index: int
    siblings: List[Tuple[bytes, bool]]  # (hash, is_left)


@dataclass
class GEMMBlockProof:
    """Proof for one block of GEMM output."""

    bi: int  # Block row index
    bj: int  # Block column index
    leaf_hash: bytes  # Hash of this block
    merkle_path: MerklePath
    sumcheck_proof: SumcheckProof
    spot_X: List[SpotCheck]  # Spot checks on input (values only, for demo)
    spot_W: List[SpotCheck]  # Spot checks on weights

    # Dimension info for verifier to reconstruct transcript
    m_bits: int = 0  # Number of bits for row index
    n_bits: int = 0  # Number of bits for column index
    k_bits: int = 0  # Number of bits for inner dimension (= len(sumcheck_proof.rounds))

    # Production: Merkle-verified X spot checks (replaces spot_X when set)
    # Each entry proves the X value is in the committed layer activation
    spot_X_with_proofs: List[SpotCheckWithProof] = field(default_factory=list)

    # Merkle-verified W spot checks for lightweight validation
    # Each entry proves the W value is in the committed weight matrix
    # Allows validators to verify without loading the model
    spot_W_with_proofs: List[SpotCheckWithProof] = field(default_factory=list)


@dataclass
class GEMMProof:
    """Complete proof for one GEMM operation."""

    output_root: bytes  # Merkle root of output matrix
    block_proofs: List[GEMMBlockProof]


@dataclass
class VerificationResult:
    """Result of verification."""

    passed: bool
    message: str
    details: Optional[dict] = None

    @staticmethod
    def success(message: str = "Verification passed") -> "VerificationResult":
        return VerificationResult(passed=True, message=message)

    @staticmethod
    def failure(message: str, details: Optional[dict] = None) -> "VerificationResult":
        return VerificationResult(passed=False, message=message, details=details)
