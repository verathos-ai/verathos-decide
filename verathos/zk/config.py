"""
Configuration for ZkLLM.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# Mersenne prime: p = 2^61 - 1
FIELD_PRIME = (1 << 61) - 1

# Default block size for Merkle commitments
DEFAULT_BLOCK_SIZE = 256

# Default number of layers to challenge (0 = use auto-k)
DEFAULT_K_LAYERS = 0

# Default target detection probability for auto-k formula
# k = max(2, round(num_layers × target_detection))
DEFAULT_TARGET_DETECTION = 0.0625  # 6.25% per inference

# Default number of blocks per GEMM to verify.
# 1 is sufficient: single-block sumcheck proves Y=X@W with soundness
# 1 - 1/p (p = 2^61 - 1).  k_blocks is defense-in-depth only —
# detection probability depends solely on k_layers/L.
DEFAULT_K_BLOCKS = 1

# Default number of spot checks per block.
# Spot checks are defense-in-depth: they verify X/W values match
# committed Merkle roots, but the sumcheck already proves Y=X@W
# mathematically.  25 Fiat-Shamir-derived positions are sufficient
# to bind values to Merkle commitments.
# Original was 100 (54.2% per-block), then 50 (32.5%).
DEFAULT_SPOT_CHECKS = 25

# Default chunk size for flat weight Merkle trees (elements per leaf)
# 128 elements * 8 bytes = 1KB per leaf (vs 512KB with block_size=256)
DEFAULT_W_MERKLE_CHUNK_SIZE = 128

# Challenge window duration (seconds)
DEFAULT_CHALLENGE_WINDOW = 60.0


@dataclass
class ToleranceSpec:
    """
    Error tolerance for verification.

    Since we use INT8 quantization, most operations are exact.
    Tolerance is needed for:
    - Dequantization (INT32 -> FP32)
    - Non-linear operations (computed in FP32)
    """

    # Per-element relative tolerance
    gemm_rel_tol: float = 1e-3  # After dequantization
    softmax_rel_tol: float = 1e-5  # Softmax is sensitive
    activation_rel_tol: float = 1e-4
    norm_rel_tol: float = 1e-4

    # Per-element absolute tolerance (for values near zero)
    abs_tol: float = 1e-6

    def check_close(
        self, computed: "torch.Tensor", expected: "torch.Tensor", tol_type: str = "gemm"
    ) -> tuple[bool, float]:
        """
        Check if tensors match within tolerance.

        Returns:
            (passed, max_relative_error)
        """
        import torch

        diff = torch.abs(computed.float() - expected.float())
        rel_err = diff / (torch.abs(expected.float()) + self.abs_tol)
        max_rel_err = rel_err.max().item()

        tol_map = {
            "gemm": self.gemm_rel_tol,
            "softmax": self.softmax_rel_tol,
            "activation": self.activation_rel_tol,
            "norm": self.norm_rel_tol,
        }
        tol = tol_map.get(tol_type, self.gemm_rel_tol)

        return max_rel_err <= tol, max_rel_err


@dataclass
class Config:
    """Global configuration for ZkLLM."""

    # Field prime for cryptographic operations
    field_prime: int = FIELD_PRIME

    # Block size for Merkle trees
    block_size: int = DEFAULT_BLOCK_SIZE

    # Challenge parameters
    k_layers: int = DEFAULT_K_LAYERS  # Layers to challenge (0 = use auto-k)
    target_detection: float = DEFAULT_TARGET_DETECTION  # Auto-k: k = max(2, round(L × target))
    k_blocks: int = DEFAULT_K_BLOCKS  # Blocks per GEMM
    spot_checks: int = DEFAULT_SPOT_CHECKS  # Spot checks per block

    # Timing
    challenge_window_sec: float = DEFAULT_CHALLENGE_WINDOW

    # Tolerance
    tolerance: ToleranceSpec = field(default_factory=ToleranceSpec)

    # Witness storage
    witness_retention_sec: float = DEFAULT_CHALLENGE_WINDOW * 2
    max_witness_memory_gb: float = 4.0  # Max memory for witness storage

    # Weight Merkle tree chunk size (elements per leaf)
    w_merkle_chunk_size: int = DEFAULT_W_MERKLE_CHUNK_SIZE

    # Parallelism
    parallel_proofs: bool = True  # Enable parallel proof generation
    max_workers: int = 0  # 0 = auto (cpu_count), >0 = specific number
    synchronous_proofs: bool = False  # Skip ThreadPoolExecutor in generate_proofs

    # Proof matmul backend: how proof generation computes Y = X @ W
    #   "gpu"      - GPU-first with non-blocking CPU spillover.  Default.
    #                Auto-sized GPU semaphore (sm_count // 64) limits SM contention;
    #                excess matmuls spill to fast CPU f32 SGEMM.
    #   "cpu"      - CPU-only float32 SGEMM with concurrency limiter.
    #                Zero GPU contention — best for large dense models (14B+) at B32+.
    #   "adaptive" - Alias for "gpu" (backward compat)
    proof_matmul_backend: str = "gpu"

    # Max concurrent GPU matmul allocations.  0 = auto (sm_count // 64).
    proof_gpu_matmul_limit: int = 0

    # Max parallel layers within a single proof (0 = auto = len(challenges))
    proof_layer_parallelism: int = 0

    # Debug
    debug: bool = False
    verbose: bool = False

    # =========================================================================
    # MoE (Mixture of Experts) Challenge Parameters
    # =========================================================================

    # Total unique experts to challenge per MoE layer (0 = auto from target_expert_detection).
    # Candidates are collected from k_tokens sampled tokens' routing, then capped to this value.
    k_experts_per_layer: int = 0

    # Auto k_experts: k = max(1, round(num_experts × target))
    # 6.25% of 256 experts → k=16.  Use explicit k_experts_per_layer to override.
    target_expert_detection: float = 0.0625

    # Hard cap on experts challenged per MoE layer (safety net).
    # Prevents runaway proof overhead on models with many experts
    # (e.g. raw formula gives k=16 for 256 experts before k_tokens division).
    max_k_experts: int = 8

    # Number of tokens to sample for expert challenge derivation
    # Higher = more coverage but more proof overhead
    k_tokens_per_expert: int = 4

    # Whether to verify router/gating decisions
    # When True, spot-checks router logits consistency with claimed routing
    verify_routing: bool = True

    # Challenge shared experts (always-active experts in some MoE architectures)
    # Some models like DeepSeek-V2 have shared experts that process all tokens
    challenge_shared_experts: bool = True


# Global default config
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config, creating default if needed."""
    global _default_config
    if _default_config is None:
        _default_config = Config()
    return _default_config


def set_config(config: Config) -> None:
    """Set global config."""
    global _default_config
    _default_config = config
