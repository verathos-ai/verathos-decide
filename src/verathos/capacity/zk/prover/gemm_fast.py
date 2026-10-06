"""
Fast GEMM Prover using vectorized operations.

Optimized version using NumPy for batch field operations.
Supports parallel proof generation across multiple CPU cores.
"""

from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

import torch

from verathos.capacity.zk.config import Config, get_config
from verathos.capacity.zk.types import (
    GEMMProof,
    GEMMBlockProof,
    SumcheckProof,
    SpotCheck,
    SpotCheckWithProof,
)
from verathos.capacity.zk.crypto.field import mod_p
from verathos.capacity.zk.crypto.merkle import build_block_merkle, BlockMerkleTree, FlatWeightMerkle
from verathos.capacity.zk.crypto.transcript import Transcript
from verathos.capacity.zk.crypto.field_fast import (
    build_eq_table_vec,
    tensor_to_field_vec,
    mul_f_vec_mersenne,
    mod_p_vec,
)
from verathos.capacity.zk.crypto.sumcheck_fast import sumcheck_prove_fast
from verathos.capacity.zk.crypto.mle_fast import (
    build_mle_from_matrix_fast,
)

# Try to import native extension for additional speedup
_HAS_NATIVE = False
try:
    from verathos.capacity.zk.cuda import zkllm_native as _native
    if _native is None:
        raise ImportError("zkllm_native unavailable")
    _HAS_NATIVE = True
except ImportError:
    try:
        import sys
        from pathlib import Path
        cuda_path = Path(__file__).parent.parent / "cuda"
        if str(cuda_path) not in sys.path:
            sys.path.insert(0, str(cuda_path))
        import zkllm_native as _native
        _HAS_NATIVE = True
    except ImportError:
        pass


class GEMMProverFast:
    """
    Fast GEMM prover using vectorized operations.

    For Y = X @ W where:
    - X: [M, K] input activations
    - W: [K, N] weights
    - Y: [M, N] output

    Supports parallel proof generation across CPU cores when enabled.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize fast GEMM prover."""
        self.config = config or get_config()
        self.block_size = self.config.block_size

    def _get_num_workers(self) -> int:
        """Get number of parallel workers."""
        if self.config.max_workers > 0:
            return self.config.max_workers
        return os.cpu_count() or 4

    @staticmethod
    def _cuda_min_k_bits() -> int:
        """Minimum inner-dimension bits where CUDA proof helpers amortize setup cost."""
        try:
            return int(os.environ.get("ZKLLM_PROOF_CUDA_MIN_K_BITS", "12"))
        except ValueError:
            return 12

    def prove(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Y: torch.Tensor,
        challenged_blocks: List[Tuple[int, int]],
        transcript: Transcript,
        W_merkle: Optional[BlockMerkleTree] = None,
        Y_merkle: Optional[BlockMerkleTree] = None,
    ) -> GEMMProof:
        """
        Generate proof for GEMM operation using fast path.

        Args:
            W_merkle: Optional pre-built BlockMerkleTree for W. When provided,
                      generates Merkle-verified W spot checks (spot_W_with_proofs)
                      enabling lightweight validation without model loading.
            Y_merkle: Optional pre-built BlockMerkleTree for Y. When provided,
                      skips redundant rebuild (useful when multiple GEMMs share
                      the same Y in a layer).
        """
        if X.dim() != 2 or W.dim() != 2 or Y.dim() != 2:
            raise ValueError("All tensors must be 2D")

        M, K = X.shape
        K2, N = W.shape
        if K != K2:
            raise ValueError(f"Inner dimensions must match: X[{M},{K}] @ W[{K2},{N}]")

        # Build Merkle tree over Y blocks (or reuse if provided)
        if Y_merkle is None:
            Y_merkle = build_block_merkle(Y, self.block_size)

        # Absorb Y root into transcript
        transcript.absorb_bytes(b"Y_root", Y_merkle.root)

        # Pre-flatten W for Merkle proof extraction: single .contiguous()
        # + .cpu() here instead of per-block (avoids N×GPU alloc + sync).
        W_flat_cpu = None
        W_raw_bytes = None
        if W_merkle is not None:
            W_flat_cpu = W.contiguous().view(-1)
            if W_flat_cpu.is_cuda:
                W_flat_cpu = W_flat_cpu.cpu()
            # Pre-serialize to raw bytes once — chunk extraction in
            # get_proof_batch becomes O(1) byte slicing instead of
            # 50× per-block numpy conversions.  Skip for large W
            # (>10M elems) where tobytes() overhead exceeds savings.
            if W_flat_cpu.numel() < 10_000_000:
                W_raw_bytes = W_flat_cpu.numpy().astype(
                    W_merkle._dtype_code).tobytes()

        # Generate proofs - parallel or sequential
        # NOTE: GC suppression is handled by the caller (generate_proofs)
        # which wraps the entire proof pipeline.  Do NOT add gc.disable
        # here — it would re-enable GC between prove() calls and cause
        # ~200ms pauses that regress B1 prove time.
        use_parallel = (
            self.config.parallel_proofs
            and len(challenged_blocks) > 1
        )

        if use_parallel:
            block_proofs = self._prove_blocks_parallel(
                X, W, Y, Y_merkle, challenged_blocks, transcript,
                W_merkle=W_merkle,
                W_flat_cpu=W_flat_cpu,
                W_raw_bytes=W_raw_bytes,
            )
        else:
            block_proofs = []
            for bi, bj in challenged_blocks:
                block_proof = self._prove_block_fast(
                    X=X,
                    W=W,
                    Y=Y,
                    Y_merkle=Y_merkle,
                    bi=bi,
                    bj=bj,
                    transcript=transcript.fork(f"block_{bi}_{bj}".encode()),
                    W_merkle=W_merkle,
                    W_flat_cpu=W_flat_cpu,
                    W_raw_bytes=W_raw_bytes,
                )
                block_proofs.append(block_proof)

        return GEMMProof(
            output_root=Y_merkle.root,
            block_proofs=block_proofs,
        )

    def _prove_blocks_parallel(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Y: torch.Tensor,
        Y_merkle: BlockMerkleTree,
        challenged_blocks: List[Tuple[int, int]],
        transcript: Transcript,
        W_merkle: Optional[BlockMerkleTree] = None,
        W_flat_cpu: Optional[torch.Tensor] = None,
        W_raw_bytes: Optional[bytes] = None,
    ) -> List[GEMMBlockProof]:
        """
        Generate block proofs in parallel using ThreadPoolExecutor.

        ThreadPoolExecutor works well here because:
        1. NumPy operations release the GIL
        2. Native extensions (if available) release the GIL
        3. Avoids serialization overhead of ProcessPoolExecutor
        """
        num_workers = min(self._get_num_workers(), len(challenged_blocks))

        def prove_single_block(args):
            bi, bj = args
            return self._prove_block_fast(
                X=X,
                W=W,
                Y=Y,
                Y_merkle=Y_merkle,
                bi=bi,
                bj=bj,
                transcript=transcript.fork(f"block_{bi}_{bj}".encode()),
                W_merkle=W_merkle,
                W_flat_cpu=W_flat_cpu,
                W_raw_bytes=W_raw_bytes,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            block_proofs = list(executor.map(prove_single_block, challenged_blocks))

        return block_proofs

    def _prove_block_fast(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Y: torch.Tensor,
        Y_merkle: BlockMerkleTree,
        bi: int,
        bj: int,
        transcript: Transcript,
        W_merkle: Optional[BlockMerkleTree] = None,
        W_flat_cpu: Optional[torch.Tensor] = None,
        W_raw_bytes: Optional[bytes] = None,
    ) -> GEMMBlockProof:
        """Prove a single block using fast vectorized operations."""
        import time as _time

        M, K = X.shape
        _, N = W.shape
        bs = self.block_size

        # Extract block boundaries
        row_start = bi * bs
        row_end = min(row_start + bs, M)
        col_start = bj * bs
        col_end = min(col_start + bs, N)

        # Extract the block (contiguous() avoids strided memory access in field conversion)
        t0_slice = _time.perf_counter()
        Y_block = Y[row_start:row_end, col_start:col_end]
        X_rows = X[row_start:row_end, :]
        W_cols = W[:, col_start:col_end].contiguous()
        t_slice = (_time.perf_counter() - t0_slice) * 1000

        # Get Merkle path and leaf hash
        t0_ypath = _time.perf_counter()
        merkle_path = Y_merkle.get_path(bi, bj)
        leaf_hash = Y_merkle.get_leaf(bi, bj)
        t_ypath = (_time.perf_counter() - t0_ypath) * 1000

        # ================================================================
        # Fast path: complete block proof in one C++ call
        # Handles: transcript absorb/squeeze, field conversion, MLE eval,
        # T_A/T_B, sumcheck loop, and spot check sampling.
        # Only W Merkle proof extraction remains in Python.
        # ================================================================
        _has_native_block = (_HAS_NATIVE
                             and hasattr(_native, 'native_prove_block'))
        if _has_native_block:
            # -- Tensor prep (C++ accepts int8/int32/int64 via dtype dispatch) --
            t0_cast = _time.perf_counter()
            # X and Y are already int64 from matmul; W may be int8 from
            # quantization — passed as-is to avoid 4MB+ int64 allocation.
            t_cast = (_time.perf_counter() - t0_cast) * 1000

            # -- Core C++ proof --
            t0 = _time.perf_counter()
            try:
                (rounds_raw, claimed_sum, final_A, final_B, challenges,
                 spot_X_raw, spot_W_raw, new_state
                ) = _native.native_prove_block(
                    X_rows, W_cols, Y_block,
                    transcript.get_state(),
                    bi, bj, leaf_hash,
                    self.config.spot_checks,
                    row_start, col_start,
                    W.shape[1] if W_merkle else 0,
                    W_merkle.chunk_size if W_merkle else 128,
                )
            except Exception as exc:
                # Intermittent live failures ("cannot create std::vector
                # larger than max_size()") arrive here with no context; the
                # exact native inputs are the forensic trail.
                raise RuntimeError(
                    "native_prove_block failed: "
                    f"X_rows={tuple(X_rows.shape)}/{X_rows.dtype} "
                    f"W_cols={tuple(W_cols.shape)}/{W_cols.dtype} "
                    f"Y_block={tuple(Y_block.shape)}/{Y_block.dtype} "
                    f"contig=({X_rows.is_contiguous()},"
                    f"{W_cols.is_contiguous()},{Y_block.is_contiguous()}) "
                    f"bi={bi} bj={bj} bs={bs} M={M} K={K} N={N} "
                    f"rows={row_start}:{row_end} cols={col_start}:{col_end} "
                    f"spot_checks={self.config.spot_checks} "
                    f"W_total_cols={W.shape[1] if W_merkle else 0} "
                    f"w_chunk={W_merkle.chunk_size if W_merkle else 128} "
                    f"leaf_hash_len={len(leaf_hash)} "
                    f"state_len={len(transcript.get_state())}"
                ) from exc
            transcript._state = new_state
            t_cpp = (_time.perf_counter() - t0) * 1000

            # -- Build proof objects from raw C++ data --
            t0_obj = _time.perf_counter()
            from verathos.capacity.zk.types import SumcheckRound
            rounds = [SumcheckRound(e0=e0, e1=e1, e2=e2)
                      for e0, e1, e2 in rounds_raw]
            sumcheck_proof = SumcheckProof(
                claimed_sum=claimed_sum, rounds=rounds,
                final_A=final_A, final_B=final_B)
            spot_X = [SpotCheck(row=r, col=c, value=v)
                      for r, c, v in spot_X_raw]
            spot_W = [SpotCheck(row=r, col=c, value=v)
                      for r, c, v, _ in spot_W_raw]
            t_obj = (_time.perf_counter() - t0_obj) * 1000

            # W Merkle proofs: batch-extract using chunk indices from C++
            t0_merkle = _time.perf_counter()
            spot_W_with_proofs = []
            if W_merkle is not None:
                unique_chunks = list(dict.fromkeys(
                    ci for _, _, _, ci in spot_W_raw))
                # Use pre-flattened CPU tensor (avoids per-block GPU
                # .contiguous() + GPU→CPU transfer — the W Merkle bottleneck).
                W_full_flat = W_flat_cpu if W_flat_cpu is not None else W.contiguous().view(-1)
                if hasattr(W_merkle, 'get_proof_batch'):
                    batch = W_merkle.get_proof_batch(
                        unique_chunks, W_tensor=W_full_flat,
                        raw_bytes=W_raw_bytes)
                    chunk_cache = {
                        ci: (ld, mp)
                        for ci, (mp, ld) in zip(unique_chunks, batch)}
                else:
                    chunk_cache = {}
                    for ci in unique_chunks:
                        mp, ld = W_merkle.get_proof(
                            ci, W_tensor=W_full_flat,
                            raw_bytes=W_raw_bytes)
                        chunk_cache[ci] = (ld, mp)

                for r, c, v, ci in spot_W_raw:
                    ld, mp = chunk_cache[ci]
                    spot_W_with_proofs.append(
                        SpotCheckWithProof(
                            row=r, col=c, value=v,
                            merkle_path=mp, leaf_data=ld))
            t_merkle = (_time.perf_counter() - t0_merkle) * 1000

            # Store timing
            if not hasattr(self, '_block_timings'):
                self._block_timings = []
            m, K_dim = X_rows.shape
            _, n = W_cols.shape
            self._block_timings.append({
                'slice': t_slice,
                'y_path': t_ypath,
                'cast': t_cast,
                'native_cpp': t_cpp,
                'obj_build': t_obj,
                'w_merkle': t_merkle,
                'dims': f'[{m},{K_dim}]x[{K_dim},{n}]',
            })

            m_bits = max(1, (m - 1).bit_length()) if m > 1 else 1
            n_bits = max(1, (n - 1).bit_length()) if n > 1 else 1
            k_bits = len(sumcheck_proof.rounds)

            return GEMMBlockProof(
                bi=bi, bj=bj,
                leaf_hash=leaf_hash, merkle_path=merkle_path,
                sumcheck_proof=sumcheck_proof,
                spot_X=spot_X, spot_W=spot_W,
                m_bits=m_bits, n_bits=n_bits, k_bits=k_bits,
                spot_W_with_proofs=spot_W_with_proofs,
            )

        # ================================================================
        # Fallback: existing Python path
        # ================================================================

        # Absorb block info into transcript
        transcript.absorb(b"bi", bi)
        transcript.absorb(b"bj", bj)
        transcript.absorb_bytes(b"leaf_hash", leaf_hash)

        # Generate sumcheck proof using fast path
        sumcheck_proof, challenges = self._sumcheck_for_block_fast(
            X_rows=X_rows,
            W_cols=W_cols,
            Y_block=Y_block,
            transcript=transcript,
        )

        # Generate spot checks with global indices
        t_spot_x = _time.perf_counter()
        spot_X = self._sample_spots_with_offset(
            X_rows, transcript, b"spot_X", row_offset=row_start, col_offset=0
        )
        t_spot_x = (_time.perf_counter() - t_spot_x) * 1000

        # Generate W spot checks (with optional Merkle proofs)
        t_spot_w = _time.perf_counter()
        spot_W_with_proofs = []
        if W_merkle is not None:
            spot_W, spot_W_with_proofs = self._sample_spots_with_merkle_proof(
                W_full=W,
                W_cols=W_cols,
                merkle_tree=W_merkle,
                transcript=transcript,
                label=b"spot_W",
                row_offset=0,
                col_offset=col_start,
                W_flat_cpu=W_flat_cpu,
                W_raw_bytes=W_raw_bytes,
            )
        else:
            spot_W = self._sample_spots_with_offset(
                W_cols, transcript, b"spot_W", row_offset=0, col_offset=col_start
            )
        t_spot_w = (_time.perf_counter() - t_spot_w) * 1000

        # Store spot check timing in _block_timings (appended by _sumcheck_for_block_fast)
        if hasattr(self, '_block_timings') and self._block_timings:
            self._block_timings[-1]['spot_X'] = t_spot_x
            self._block_timings[-1]['spot_W'] = t_spot_w

        # Compute dimension bits for verifier
        m, K_dim = X_rows.shape
        _, n = W_cols.shape
        m_bits = max(1, (m - 1).bit_length()) if m > 1 else 1
        n_bits = max(1, (n - 1).bit_length()) if n > 1 else 1
        k_bits = len(sumcheck_proof.rounds)

        return GEMMBlockProof(
            bi=bi,
            bj=bj,
            leaf_hash=leaf_hash,
            merkle_path=merkle_path,
            sumcheck_proof=sumcheck_proof,
            spot_X=spot_X,
            spot_W=spot_W,
            m_bits=m_bits,
            n_bits=n_bits,
            k_bits=k_bits,
            spot_W_with_proofs=spot_W_with_proofs,
        )

    def _sumcheck_for_block_fast(
        self,
        X_rows: torch.Tensor,
        W_cols: torch.Tensor,
        Y_block: torch.Tensor,
        transcript: Transcript,
    ) -> Tuple[SumcheckProof, List[int]]:
        """
        Generate sumcheck proof using fast vectorized operations.

        Fast path: When CUDA extension has cuda_prove_sumcheck_block, the entire
        block proof (field conversion, MLE eval, T_A/T_B, sumcheck loop) runs
        in a single C++ call with GPU-resident data and C++ Fiat-Shamir transcript.
        This eliminates ~40 GPU<->CPU roundtrips per block.
        """
        import time as _time

        m, K = X_rows.shape
        _, n = W_cols.shape

        # Pad dimensions to powers of 2
        m_bits = max(1, (m - 1).bit_length())
        n_bits = max(1, (n - 1).bit_length())
        k_bits = max(1, (K - 1).bit_length())

        m_padded = 1 << m_bits
        n_padded = 1 << n_bits
        k_padded = 1 << k_bits

        from verathos.capacity.zk.crypto.field_fast import _PROOF_CPU_ONLY
        _has_cuda = (_HAS_NATIVE
                     and getattr(_native, 'HAS_CUDA', False)
                     and not _PROOF_CPU_ONLY)
        _use_cuda_helpers = _has_cuda and k_bits >= self._cuda_min_k_bits()
        _has_fused = (_has_cuda
                      and hasattr(_native, 'cuda_prove_sumcheck_block'))

        # ================================================================
        # Fast path: fused GPU sumcheck (single C++ call, no roundtrips)
        # Only beneficial when k_bits is large enough that the sumcheck
        # loop dominates.  For small MoE expert blocks ([1,2048]x[2048,256]),
        # GPU data transfer overhead exceeds the savings.
        # Threshold: k_bits >= 12 (~4096 inner dim) where the sumcheck
        # loop has 12+ rounds and GPU transfer is amortized.
        # ================================================================
        _FUSED_MIN_K_BITS = self._cuda_min_k_bits()
        if _has_fused and k_bits >= _FUSED_MIN_K_BITS:
            t0 = _time.perf_counter()

            # Prepare int64 tensors on GPU
            X_gpu = X_rows.to(device='cuda', dtype=torch.int64).contiguous()
            W_gpu = W_cols.to(device='cuda', dtype=torch.int64).contiguous()
            Y_gpu = Y_block.to(device='cuda', dtype=torch.int64).contiguous()

            # Single C++ call: field conversion + MLE eval + T_A/T_B + sumcheck
            rounds_raw, claimed_sum, final_A, final_B, challenges, new_state = \
                _native.cuda_prove_sumcheck_block(
                    X_gpu, W_gpu, Y_gpu,
                    transcript.get_state(),
                    m_bits, n_bits, k_bits, 0)

            # Update transcript state to match what Python would have computed
            transcript._state = new_state

            t_total = (_time.perf_counter() - t0) * 1000

            # Build proof object
            from verathos.capacity.zk.types import SumcheckRound
            rounds = [SumcheckRound(e0=e0, e1=e1, e2=e2)
                      for e0, e1, e2 in rounds_raw]
            proof = SumcheckProof(
                claimed_sum=claimed_sum,
                rounds=rounds,
                final_A=final_A,
                final_B=final_B,
            )

            # Store timing
            if not hasattr(self, '_block_timings'):
                self._block_timings = []
            self._block_timings.append({
                'fused_gpu': t_total,
                'dims': f'[{m},{K}]x[{K},{n}]',
            })

            return proof, challenges

        # ================================================================
        # Fallback: existing multi-step path (CPU/GPU mix)
        # ================================================================
        t0 = _time.perf_counter()
        if _use_cuda_helpers and not X_rows.is_cuda:
            X_np = tensor_to_field_vec(
                X_rows.to(device='cuda', dtype=torch.int64), m_padded, k_padded)
        else:
            X_np = tensor_to_field_vec(X_rows, m_padded, k_padded)
        _t_X = (_time.perf_counter() - t0) * 1000

        # W: upload to GPU for fast field conversion + T_B compute.
        t0 = _time.perf_counter()
        W_gpu_field = None
        if _use_cuda_helpers and isinstance(W_cols, torch.Tensor):
            t = W_cols.to(device='cuda', dtype=torch.int64)
            if not t.is_contiguous():
                t = t.contiguous()
            W_gpu_field = _native.cuda_tensor_to_field(t, k_padded, n_padded, 0)
            del t
        else:
            W_np = tensor_to_field_vec(W_cols, k_padded, n_padded)
        _t_W = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        if _use_cuda_helpers and not Y_block.is_cuda:
            Y_np = tensor_to_field_vec(
                Y_block.to(device='cuda', dtype=torch.int64), m_padded, n_padded)
        else:
            Y_np = tensor_to_field_vec(Y_block, m_padded, n_padded)
        _t_Y = (_time.perf_counter() - t0) * 1000

        t_field = _t_X + _t_W + _t_Y

        # Build MLEs
        t0 = _time.perf_counter()
        Y_mle = build_mle_from_matrix_fast(Y_np)
        t_mle_build = (_time.perf_counter() - t0) * 1000

        # Derive random evaluation points
        r_i = transcript.squeeze_n(b"r_i", m_bits)
        r_j = transcript.squeeze_n(b"r_j", n_bits)

        # Compute claimed sum
        t0 = _time.perf_counter()
        claimed_sum = Y_mle.evaluate(r_i + r_j)
        t_mle_eval = (_time.perf_counter() - t0) * 1000

        # Build sumcheck tables
        t0 = _time.perf_counter()
        T_A = self._compute_T_A_fast(X_np, r_i, k_bits)
        t_T_A = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        if W_gpu_field is not None:
            T_B = self._compute_T_B_gpu(W_gpu_field, r_j)
        else:
            T_B = self._compute_T_B_fast(W_np, r_j, k_bits)
        t_T_B = (_time.perf_counter() - t0) * 1000

        # Run sumcheck
        t0 = _time.perf_counter()
        proof, challenges = sumcheck_prove_fast(
            T_A, T_B, claimed_sum, transcript
        )
        t_sumcheck = (_time.perf_counter() - t0) * 1000

        # Store timing
        if not hasattr(self, '_block_timings'):
            self._block_timings = []
        self._block_timings.append({
            'field_convert': t_field,
            'field_X': _t_X,
            'field_W': _t_W,
            'field_Y': _t_Y,
            'mle_build': t_mle_build,
            'mle_eval': t_mle_eval,
            'T_A': t_T_A,
            'T_B': t_T_B,
            'sumcheck': t_sumcheck,
            'dims': f'[{m},{K}]x[{K},{n}]',
        })

        return proof, challenges

    def _compute_T_A_fast(
        self,
        X: np.ndarray,
        r_i: List[int],
        k_bits: int,
    ) -> np.ndarray:
        """
        Compute T_A[k] = X_tilde(r_i, k) using vectorized operations.

        X_tilde(r_i, k) = sum_{i in {0,1}^m} X[i,k] * eq(r_i, i)

        Vectorized: broadcast-multiply X[i,:] * eq[i] for all i at once,
        then sum rows with periodic Mersenne reduction. No Python element loops.
        """
        if _HAS_NATIVE:
            return _native.native_compute_T_A(X, np.array(r_i, dtype=np.uint64), k_bits)

        m, _ = X.shape
        k_size = 1 << k_bits

        # Build eq table for r_i
        eq_table = build_eq_table_vec(r_i)
        eq_len = min(m, len(eq_table))

        # Vectorized: compute all products at once using broadcasting
        # X_sub: [eq_len, k_size], eq_col: [eq_len, 1] -> broadcast to [eq_len, k_size]
        X_sub = X[:eq_len, :k_size]
        eq_col = eq_table[:eq_len, np.newaxis]

        # Mersenne multiply: products[i, k] = X[i,k] * eq[i] mod P
        products = mul_f_vec_mersenne(X_sub, eq_col)

        # Modular sum along axis 0: T_A[k] = sum_i products[i, k]
        # Add rows with periodic reduction to prevent uint64 overflow
        # (4 values < P summed safely fit in uint64: 4P < 2^63)
        T_A = np.zeros(k_size, dtype=np.uint64)
        for i in range(eq_len):
            T_A += products[i]
            if (i + 1) % 4 == 0:
                T_A = mod_p_vec(T_A)
        if eq_len % 4 != 0:
            T_A = mod_p_vec(T_A)

        return T_A

    def _compute_T_B_fast(
        self,
        W: np.ndarray,
        r_j: List[int],
        k_bits: int,
    ) -> np.ndarray:
        """
        Compute T_B[k] = W_tilde(k, r_j) using vectorized operations.

        W_tilde(k, r_j) = sum_{j in {0,1}^n} W[k,j] * eq(r_j, j)

        Vectorized: broadcast-multiply W[k,:] * eq[j] for all k at once,
        then sum columns with periodic Mersenne reduction. No Python element loops.
        """
        if _HAS_NATIVE:
            return _native.native_compute_T_B(W, np.array(r_j, dtype=np.uint64), k_bits)

        k_size = 1 << k_bits
        n = W.shape[1] if len(W.shape) > 1 else 0

        # Build eq table for r_j
        eq_table = build_eq_table_vec(r_j)
        eq_len = min(n, len(eq_table))

        # Vectorized: compute all products at once using broadcasting
        # W_sub: [k_size, eq_len], eq_row: [1, eq_len] -> broadcast to [k_size, eq_len]
        W_sub = W[:k_size, :eq_len]
        eq_row = eq_table[:eq_len][np.newaxis, :]

        # Mersenne multiply: products[k, j] = W[k,j] * eq[j] mod P
        products = mul_f_vec_mersenne(W_sub, eq_row)

        # Modular sum along axis 1: T_B[k] = sum_j products[k, j]
        # Sum columns with periodic reduction
        T_B = np.zeros(k_size, dtype=np.uint64)
        for j in range(eq_len):
            T_B += products[:, j]
            if (j + 1) % 4 == 0:
                T_B = mod_p_vec(T_B)
        if eq_len % 4 != 0:
            T_B = mod_p_vec(T_B)

        return T_B

    def _compute_T_B_gpu(
        self,
        W_field_gpu: torch.Tensor,
        r_j: list,
    ) -> np.ndarray:
        """
        Compute T_B[k] = sum_j W[k,j] * eq(r_j, j) entirely on GPU.

        Avoids the large GPU→CPU transfer of W and the CPU T_B computation.
        Only the result vector (k_size elements) is transferred to CPU.
        """
        point_t = torch.tensor(r_j, dtype=torch.int64, device=W_field_gpu.device)
        eq_gpu = _native.cuda_build_eq_table(point_t, 0)
        T_B_gpu = _native.cuda_compute_T_fixing_col(W_field_gpu, eq_gpu, 0)
        return T_B_gpu.cpu().numpy().view(np.uint64)

    def _sample_spots_with_offset(
        self,
        tensor: torch.Tensor,
        transcript: Transcript,
        label: bytes,
        row_offset: int,
        col_offset: int,
    ) -> List[SpotCheck]:
        """Sample spot check locations and values, storing global indices."""
        rows, cols = tensor.shape
        num_spots = self.config.spot_checks

        # Pre-convert to numpy for fast element access (avoids .item() overhead)
        arr = tensor.numpy() if tensor.is_cpu else tensor.cpu().numpy()

        spots = []
        for i in range(num_spots):
            # Derive local indices from transcript
            local_row = transcript.squeeze(label + b":row:" + str(i).encode()) % rows
            local_col = transcript.squeeze(label + b":col:" + str(i).encode()) % cols

            value = mod_p(int(arr[local_row, local_col]))

            # Store global indices
            spots.append(SpotCheck(
                row=row_offset + local_row,
                col=col_offset + local_col,
                value=value
            ))

        return spots

    def _sample_spots_with_merkle_proof(
        self,
        W_full: torch.Tensor,
        W_cols: torch.Tensor,
        merkle_tree: FlatWeightMerkle,
        transcript: Transcript,
        label: bytes,
        row_offset: int,
        col_offset: int,
        W_flat_cpu: Optional[torch.Tensor] = None,
        W_raw_bytes: Optional[bytes] = None,
    ) -> Tuple[List[SpotCheck], List[SpotCheckWithProof]]:
        """Sample W spot checks and generate flat chunk Merkle proofs.

        Samples from W_cols (column slice) but generates Merkle proofs
        against the full W matrix's FlatWeightMerkle tree.

        Uses a two-phase approach for efficiency:
        1. Sample all spot check positions and collect unique chunk indices
        2. Batch-fetch all Merkle proofs (single GPU->CPU transfer if on GPU)
        3. Assemble spot checks with their proofs

        Returns:
            (spot_W, spot_W_with_proofs) - both for backward compatibility
        """
        import time as _time
        rows, cols = W_cols.shape
        N_full = merkle_tree.num_cols
        num_spots = self.config.spot_checks
        chunk_size = merkle_tree.chunk_size

        # Pre-convert to numpy for fast element access (avoids .item() overhead)
        _t0 = _time.perf_counter()
        W_arr = W_cols.numpy() if W_cols.is_cpu else W_cols.cpu().numpy()
        _t_cpu = (_time.perf_counter() - _t0) * 1000

        # Use pre-flattened CPU tensor if provided, else flatten here.
        _t0 = _time.perf_counter()
        W_full_flat = W_flat_cpu if W_flat_cpu is not None else (
            W_full.contiguous().view(-1) if W_full is not None else None)
        _t_flat = (_time.perf_counter() - _t0) * 1000

        # Phase 1: Sample all positions, collect unique chunks
        _t0 = _time.perf_counter()
        spots = []
        spot_chunk_indices = []  # chunk_idx for each spot
        unique_chunks = []  # ordered unique chunk indices
        chunk_set = set()

        for i in range(num_spots):
            local_row = transcript.squeeze(label + b":row:" + str(i).encode()) % rows
            local_col = transcript.squeeze(label + b":col:" + str(i).encode()) % cols

            value = mod_p(int(W_arr[local_row, local_col]))
            global_row = row_offset + local_row
            global_col = col_offset + local_col

            spots.append(SpotCheck(row=global_row, col=global_col, value=value))

            flat_idx = global_row * N_full + global_col
            chunk_idx = flat_idx // chunk_size
            spot_chunk_indices.append(chunk_idx)

            if chunk_idx not in chunk_set:
                chunk_set.add(chunk_idx)
                unique_chunks.append(chunk_idx)
        _t_sample = (_time.perf_counter() - _t0) * 1000

        # Phase 2: Batch-fetch all unique Merkle proofs
        _t0 = _time.perf_counter()
        if hasattr(merkle_tree, 'get_proof_batch'):
            batch_results = merkle_tree.get_proof_batch(
                unique_chunks, W_tensor=W_full_flat,
                raw_bytes=W_raw_bytes)
            chunk_cache = {ci: (ld, mp) for ci, (mp, ld) in zip(unique_chunks, batch_results)}
        else:
            chunk_cache = {}
            for ci in unique_chunks:
                mp, ld = merkle_tree.get_proof(
                    ci, W_tensor=W_full_flat,
                    raw_bytes=W_raw_bytes)
                chunk_cache[ci] = (ld, mp)
        _t_batch = (_time.perf_counter() - _t0) * 1000

        # Phase 3: Assemble spot checks with proofs
        _t0 = _time.perf_counter()
        spots_with_proofs = []
        for i in range(num_spots):
            leaf_data, merkle_path = chunk_cache[spot_chunk_indices[i]]
            spots_with_proofs.append(SpotCheckWithProof(
                row=spots[i].row,
                col=spots[i].col,
                value=spots[i].value,
                merkle_path=merkle_path,
                leaf_data=leaf_data,
            ))
        _t_assemble = (_time.perf_counter() - _t0) * 1000

        # Store sub-timings for profiling
        if not hasattr(self, '_spot_w_timings'):
            self._spot_w_timings = []
        self._spot_w_timings.append({
            'w_cols_cpu': _t_cpu,
            'w_full_flat': _t_flat,
            'sample_loop': _t_sample,
            'batch_proof': _t_batch,
            'assemble': _t_assemble,
            'n_unique_chunks': len(unique_chunks),
        })

        return spots, spots_with_proofs


# Export as drop-in replacement
GEMMProver = GEMMProverFast
