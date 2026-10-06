"""
GEMM Prover for verifiable inference.

Proves Y = X @ W using sumcheck + spot checks.
"""

from typing import List, Tuple, Optional

import torch

from verathos.capacity.zk.config import Config, get_config
from verathos.capacity.zk.types import (
    GEMMProof,
    GEMMBlockProof,
    SumcheckProof,
    SpotCheck,
)
from verathos.capacity.zk.crypto.field import mod_p
from verathos.capacity.zk.crypto.merkle import build_block_merkle, BlockMerkleTree
from verathos.capacity.zk.crypto.transcript import Transcript
from verathos.capacity.zk.crypto.mle import build_mle_from_matrix
from verathos.capacity.zk.crypto.sumcheck import sumcheck_prove


class GEMMProver:
    """
    Proves GEMM operations using sumcheck + spot checks.

    For Y = X @ W where:
    - X: [M, K] input activations (INT8 or quantized)
    - W: [K, N] weights (INT8 or quantized)
    - Y: [M, N] output (INT32 accumulator or dequantized)
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize GEMM prover."""
        self.config = config or get_config()
        self.block_size = self.config.block_size

    def prove(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Y: torch.Tensor,
        challenged_blocks: List[Tuple[int, int]],
        transcript: Transcript,
    ) -> GEMMProof:
        """
        Generate proof for GEMM operation.

        Args:
            X: Input tensor [M, K]
            W: Weight tensor [K, N]
            Y: Output tensor [M, N]
            challenged_blocks: List of (block_row, block_col) to prove
            transcript: Fiat-Shamir transcript

        Returns:
            GEMMProof with proofs for all challenged blocks
        """
        if X.dim() != 2 or W.dim() != 2 or Y.dim() != 2:
            raise ValueError("All tensors must be 2D")

        M, K = X.shape
        K2, N = W.shape
        if K != K2:
            raise ValueError(f"Inner dimensions must match: X[{M},{K}] @ W[{K2},{N}]")

        # Build Merkle tree over Y blocks
        Y_merkle = build_block_merkle(Y, self.block_size)

        # Absorb Y root into transcript
        transcript.absorb_bytes(b"Y_root", Y_merkle.root)

        # Generate proof for each challenged block
        block_proofs = []
        for bi, bj in challenged_blocks:
            block_proof = self._prove_block(
                X=X,
                W=W,
                Y=Y,
                Y_merkle=Y_merkle,
                bi=bi,
                bj=bj,
                transcript=transcript.fork(f"block_{bi}_{bj}".encode()),
            )
            block_proofs.append(block_proof)

        return GEMMProof(
            output_root=Y_merkle.root,
            block_proofs=block_proofs,
        )

    def _prove_block(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Y: torch.Tensor,
        Y_merkle: BlockMerkleTree,
        bi: int,
        bj: int,
        transcript: Transcript,
    ) -> GEMMBlockProof:
        """Prove a single block of the GEMM output."""
        M, K = X.shape
        _, N = W.shape
        bs = self.block_size

        # Extract block boundaries
        row_start = bi * bs
        row_end = min(row_start + bs, M)
        col_start = bj * bs
        col_end = min(col_start + bs, N)

        # Extract the block
        Y_block = Y[row_start:row_end, col_start:col_end]
        X_rows = X[row_start:row_end, :]
        W_cols = W[:, col_start:col_end]

        # Get Merkle path and leaf hash
        merkle_path = Y_merkle.get_path(bi, bj)
        leaf_hash = Y_merkle.get_leaf(bi, bj)

        # Absorb block info into transcript
        transcript.absorb(b"bi", bi)
        transcript.absorb(b"bj", bj)
        transcript.absorb_bytes(b"leaf_hash", leaf_hash)

        # Generate sumcheck proof
        sumcheck_proof, challenges = self._sumcheck_for_block(
            X_rows=X_rows,
            W_cols=W_cols,
            Y_block=Y_block,
            transcript=transcript,
        )

        # Generate spot checks with global indices
        # For X: row is global (row_start + local_row), col is already global
        # For W: row is already global, col is global (col_start + local_col)
        spot_X = self._sample_spots_with_offset(
            X_rows, transcript, b"spot_X", row_offset=row_start, col_offset=0
        )
        spot_W = self._sample_spots_with_offset(
            W_cols, transcript, b"spot_W", row_offset=0, col_offset=col_start
        )

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
        )

    def _sumcheck_for_block(
        self,
        X_rows: torch.Tensor,  # [block_rows, K]
        W_cols: torch.Tensor,  # [K, block_cols]
        Y_block: torch.Tensor,  # [block_rows, block_cols]
        transcript: Transcript,
    ) -> Tuple[SumcheckProof, List[int]]:
        """
        Generate sumcheck proof for a block.

        We prove that Y_block[i,j] = sum_k X_rows[i,k] * W_cols[k,j]
        for a random (i,j) derived from transcript.
        """
        m, K = X_rows.shape
        _, n = W_cols.shape

        # Pad dimensions to powers of 2
        m_bits = max(1, (m - 1).bit_length())
        n_bits = max(1, (n - 1).bit_length())
        k_bits = max(1, (K - 1).bit_length())

        m_padded = 1 << m_bits
        n_padded = 1 << n_bits
        k_padded = 1 << k_bits

        # Convert to field elements
        X_field = self._tensor_to_field_2d(X_rows, m_padded, k_padded)
        W_field = self._tensor_to_field_2d(W_cols, k_padded, n_padded)
        Y_field = self._tensor_to_field_2d(Y_block, m_padded, n_padded)

        # Build MLEs
        X_mle = build_mle_from_matrix(X_field)
        W_mle = build_mle_from_matrix(W_field)
        Y_mle = build_mle_from_matrix(Y_field)

        # Derive random evaluation points for i and j
        r_i = transcript.squeeze_n(b"r_i", m_bits)
        r_j = transcript.squeeze_n(b"r_j", n_bits)

        # Compute claimed sum: Y_tilde(r_i, r_j)
        claimed_sum = Y_mle.evaluate(r_i + r_j)

        # Build sumcheck tables
        # T_A[k] = X_tilde(r_i, k) for k in {0,1}^k_bits
        # T_B[k] = W_tilde(k, r_j) for k in {0,1}^k_bits
        T_A = X_mle.partial_evaluate_first(r_i)
        T_B = W_mle.evaluate_partial(list(range(k_bits))).partial_evaluate_first(r_j)

        # Actually, we need to be more careful about the MLE structure
        # X_mle has variables for (row, col) = (m_bits, k_bits)
        # We want to fix row to r_i, leaving k_bits variables
        # Then T_A[k] = X_tilde(r_i, k)

        # Similarly for W_mle with variables (k_bits, n_bits)
        # We want to fix col to r_j, leaving k_bits variables
        # Then T_B[k] = W_tilde(k, r_j)

        # For simplicity, let's recompute the tables directly
        T_A = self._compute_T_A(X_field, r_i, k_bits)
        T_B = self._compute_T_B(W_field, r_j, k_bits)

        # Run sumcheck
        proof, challenges = sumcheck_prove(T_A, T_B, claimed_sum, transcript)

        return proof, challenges

    def _compute_T_A(
        self,
        X: List[List[int]],
        r_i: List[int],
        k_bits: int,
    ) -> List[int]:
        """
        Compute T_A[k] = X_tilde(r_i, k) for all k in {0,1}^k_bits.

        Using MLE formula:
        X_tilde(r_i, k) = sum_{i in {0,1}^m} X[i,k] * eq(r_i, i)
        """
        from verathos.capacity.zk.crypto.mle import build_eq_table
        from verathos.capacity.zk.crypto.field import add_f, mul_f

        m = len(X)
        k_size = 1 << k_bits

        # Build eq table for r_i
        eq_table = build_eq_table(r_i)

        # For each k, sum over i
        T_A = []
        for k in range(k_size):
            val = 0
            for i in range(min(m, len(eq_table))):
                if k < len(X[i]):
                    val = add_f(val, mul_f(X[i][k], eq_table[i]))
            T_A.append(val)

        return T_A

    def _compute_T_B(
        self,
        W: List[List[int]],
        r_j: List[int],
        k_bits: int,
    ) -> List[int]:
        """
        Compute T_B[k] = W_tilde(k, r_j) for all k in {0,1}^k_bits.

        W_tilde(k, r_j) = sum_{j in {0,1}^n} W[k,j] * eq(r_j, j)
        """
        from verathos.capacity.zk.crypto.mle import build_eq_table
        from verathos.capacity.zk.crypto.field import add_f, mul_f

        k_size = 1 << k_bits
        n = len(W[0]) if W else 0

        # Build eq table for r_j
        eq_table = build_eq_table(r_j)

        # For each k, sum over j
        T_B = []
        for k in range(k_size):
            val = 0
            if k < len(W):
                for j in range(min(n, len(eq_table))):
                    if j < len(W[k]):
                        val = add_f(val, mul_f(W[k][j], eq_table[j]))
            T_B.append(val)

        return T_B

    def _tensor_to_field_2d(
        self,
        tensor: torch.Tensor,
        padded_rows: int,
        padded_cols: int,
    ) -> List[List[int]]:
        """Convert 2D tensor to field elements with padding."""
        rows, cols = tensor.shape
        result = []

        for i in range(padded_rows):
            row = []
            for j in range(padded_cols):
                if i < rows and j < cols:
                    val = int(tensor[i, j].item())
                    row.append(mod_p(val))
                else:
                    row.append(0)
            result.append(row)

        return result

    def _sample_spots(
        self,
        tensor: torch.Tensor,
        transcript: Transcript,
        label: bytes,
    ) -> List[SpotCheck]:
        """Sample spot check locations and values."""
        return self._sample_spots_with_offset(tensor, transcript, label, 0, 0)

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

        spots = []
        for i in range(num_spots):
            # Derive local indices from transcript
            local_row = transcript.squeeze(label + b":row:" + str(i).encode()) % rows
            local_col = transcript.squeeze(label + b":col:" + str(i).encode()) % cols

            value = mod_p(int(tensor[local_row, local_col].item()))

            # Store global indices
            spots.append(SpotCheck(
                row=row_offset + local_row,
                col=col_offset + local_col,
                value=value
            ))

        return spots
