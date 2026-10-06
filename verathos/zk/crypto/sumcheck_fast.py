"""
Fast vectorized sumcheck protocol implementation.

Uses NumPy for batch operations to eliminate Python loop overhead.
Falls back to native extension when available. Uses CUDA for GPU acceleration.
"""

import logging
from typing import List, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

from verathos.zk.crypto.field import add_f, mul_f, lagrange_interpolate_degree2
from verathos.zk.crypto.transcript import Transcript
from verathos.zk.types import SumcheckRound, SumcheckProof
from verathos.zk.crypto.field_fast import (
    compute_round_evals_vec,
    fold_table_vec,
)

# Try to import native extension. Prefer the package loader because it selects
# the torch/Python-compatible .so or JIT-builds one when running from source.
_HAS_NATIVE = False
_HAS_CUDA = False
try:
    from verathos.zk.cuda import zkllm_native as _native
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

# Minimum table size to use CUDA (CUDA only faster for large tables due to kernel launch overhead)
_CUDA_MIN_SIZE = 8192

# Debug, not info: this fires at import time and every interactive CLI
# (manage, chat) imports through here, so at info it printed a stray
# diagnostic above the welcome banner. The slow-fallback case below
# stays a warning: that one is actionable.
if _HAS_CUDA:
    logger.debug("zkllm: Using CUDA acceleration for sumcheck (GPU fast path)")
elif _HAS_NATIVE:
    logger.debug("zkllm: Using native extension for sumcheck (CPU fast path)")
else:
    logger.warning("zkllm CUDA extension not available — sumcheck using slow NumPy fallback! "
                   "Build it: cd zkllm/cuda && python build.py")

# Dev-only escape hatch. Production proving MUST run on the native/CUDA backend:
# the NumPy fallback is orders of magnitude slower and is not the audited path,
# so a miner that silently fell back would look healthy while serving degraded.
_ALLOW_NUMPY_ENV = "VERALLM_ALLOW_NUMPY_SUMCHECK"


def native_backend_available() -> bool:
    """True when the native (C++/CUDA) sumcheck backend is loaded."""
    return _HAS_NATIVE


def assert_native_backend(context: str = "proof generation") -> None:
    """Fail closed unless the native backend is present.

    Raises RuntimeError when only the NumPy fallback is available, so a proving
    process refuses to start rather than silently running the slow/unaudited
    path. Set VERALLM_ALLOW_NUMPY_SUMCHECK=1 to override for local dev/CI only.
    """
    import os

    if _HAS_NATIVE:
        return
    if os.environ.get(_ALLOW_NUMPY_ENV, "").strip() == "1":
        logger.warning(
            "%s running on the NumPy sumcheck fallback (%s=1). This is DEV-ONLY "
            "and must never be used in production.",
            context,
            _ALLOW_NUMPY_ENV,
        )
        return
    raise RuntimeError(
        f"zkllm native CUDA extension not available — refusing to run {context} on the "
        f"slow, unaudited NumPy fallback. Build the extension for this Python/torch "
        f"(cd zkllm/cuda && python build.py), or copy a matching prebuilt "
        f"zkllm_native.*.so into zkllm/cuda/. Set {_ALLOW_NUMPY_ENV}=1 to override "
        f"(local dev only)."
    )


def compute_round_evals_fast(T_A: np.ndarray, T_B: np.ndarray) -> Tuple[int, int, int]:
    """
    Fast round evaluation using best available backend.
    Uses CUDA for large tables, native C++ otherwise.
    """
    if _HAS_CUDA and len(T_A) >= _CUDA_MIN_SIZE:
        # Convert to torch tensors for CUDA
        T_A_t = torch.from_numpy(T_A.astype(np.int64)).cuda()
        T_B_t = torch.from_numpy(T_B.astype(np.int64)).cuda()
        return _native.cuda_compute_round_evals(T_A_t, T_B_t, 0)
    elif _HAS_NATIVE:
        return _native.native_compute_round_evals(T_A, T_B)
    else:
        return compute_round_evals_vec(T_A, T_B)


def fold_table_fast(table: np.ndarray, r: int) -> np.ndarray:
    """
    Fast table folding using best available backend.
    Uses CUDA for large tables, native C++ otherwise.
    """
    if _HAS_CUDA and len(table) >= _CUDA_MIN_SIZE:
        # Convert to torch tensor for CUDA
        table_t = torch.from_numpy(table.astype(np.int64)).cuda()
        result_t = _native.cuda_fold_table(table_t, r, 0)
        return result_t.cpu().numpy().astype(np.uint64)
    elif _HAS_NATIVE:
        return _native.native_fold_table(table, r)
    else:
        return fold_table_vec(table, r)


def sumcheck_prove_fast(
    T_A,
    T_B,
    claimed_sum: int,
    transcript: Transcript,
) -> Tuple[SumcheckProof, List[int]]:
    """
    Optimized sumcheck prover using vectorized operations.

    Proves: claimed_sum = sum_{x in {0,1}^n} T_A[x] * T_B[x]

    Args:
        T_A: Table of first polynomial evaluations (size 2^n, list or ndarray)
        T_B: Table of second polynomial evaluations (size 2^n, list or ndarray)
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

    # Accept both lists and numpy arrays without redundant conversion
    cur_A = np.asarray(T_A, dtype=np.uint64)
    cur_B = np.asarray(T_B, dtype=np.uint64)
    current_claim = claimed_sum

    for round_idx in range(num_vars):
        # Compute g(0), g(1), g(2) for this round using fast path
        e0, e1, e2 = compute_round_evals_fast(cur_A, cur_B)

        rounds.append(SumcheckRound(e0=e0, e1=e1, e2=e2))

        # Absorb into transcript
        transcript.absorb(b"round", round_idx)
        transcript.absorb(b"e0", e0)
        transcript.absorb(b"e1", e1)
        transcript.absorb(b"e2", e2)

        # Get challenge
        r = transcript.squeeze(b"challenge")
        challenges.append(r)

        # Fold tables for next round using fast path
        cur_A = fold_table_fast(cur_A, r)
        cur_B = fold_table_fast(cur_B, r)

    # Final values
    final_A = int(cur_A[0]) if len(cur_A) > 0 else 0
    final_B = int(cur_B[0]) if len(cur_B) > 0 else 0

    proof = SumcheckProof(
        claimed_sum=claimed_sum,
        rounds=rounds,
        final_A=final_A,
        final_B=final_B,
    )

    return proof, challenges


def sumcheck_verify_fast(
    proof: SumcheckProof,
    transcript: Transcript,
) -> Tuple[bool, int, List[int]]:
    """
    Verify a sumcheck proof (same as original, verification is already fast).

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


# Expose original names for drop-in replacement
sumcheck_prove = sumcheck_prove_fast
sumcheck_verify = sumcheck_verify_fast
