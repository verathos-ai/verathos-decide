"""
ZkLLM: Shared zero-knowledge proof primitives for LLM verification.

Provides field arithmetic, Merkle trees, sumcheck protocols, GEMM provers/verifiers,
and other cryptographic building blocks used by both verallm (inference) and
trainllm (training).
"""

__version__ = "0.1.1"

from verathos.capacity.zk.config import Config, ToleranceSpec
from verathos.capacity.zk.types import (
    SpotCheck,
    SpotCheckWithProof,
    SumcheckRound,
    SumcheckProof,
    MerklePath,
    GEMMBlockProof,
    GEMMProof,
    VerificationResult,
)

__all__ = [
    "Config",
    "ToleranceSpec",
    "SpotCheck",
    "SpotCheckWithProof",
    "SumcheckRound",
    "SumcheckProof",
    "MerklePath",
    "GEMMBlockProof",
    "GEMMProof",
    "VerificationResult",
]
