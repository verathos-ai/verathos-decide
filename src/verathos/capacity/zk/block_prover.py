"""Sumcheck proof that one committed output block of ``Y = X @ W`` is correct, over the Mersenne-61 field.

The proof is produced by the native extension ``zk.native._block_prover`` and checked by
``proof.workspace.verify_workspace_gemm_block_proof``: the transcript absorbs the block coordinates and its Merkle
leaf, the sumcheck reduces the block's MLE claim to one evaluation of each operand, and Fiat-Shamir spot checks open
entries of the X row band and the W column band.
"""

from __future__ import annotations

from typing import Any

from verathos.capacity.zk.transcript import Transcript
from verathos.capacity.zk.types import GEMMBlockProof, SpotCheck, SumcheckProof, SumcheckRound


def load_extension() -> Any:
    import torch  # noqa: F401  (the extension links against libtorch)

    try:
        from verathos.capacity.zk.native import _block_prover
    except ImportError as exc:
        raise RuntimeError(f"native block prover unavailable ({exc}); run verathos-build-native") from exc
    return _block_prover


def _bits(size: int) -> int:
    return max(1, (size - 1).bit_length())


def prove_block(*, x: Any, w: Any, y: Any, q_tree: Any, bi: int, bj: int, block_size: int, spot_checks: int,
                transcript: Transcript) -> GEMMBlockProof:
    """Prove output block ``(bi, bj)`` of ``y = x @ w`` (2-D integer tensors) against its leaf in ``q_tree``.

    ``transcript`` must already hold the context the verifier absorbs before the block (pass, round, round root).
    """
    native = load_extension()
    row_start, col_start = bi * block_size, bj * block_size
    x_rows = x[row_start:row_start + block_size, :]
    w_cols = w[:, col_start:col_start + block_size].contiguous()
    y_block = y[row_start:row_start + block_size, col_start:col_start + block_size]
    leaf_hash = q_tree.get_leaf(bi, bj)
    rounds, claimed_sum, final_a, final_b, _challenges, spot_x, spot_w, state = native.native_prove_block(
        x_rows, w_cols, y_block, transcript.get_state(), bi, bj, leaf_hash, spot_checks, row_start, col_start, 0, 128)
    transcript.set_state(state)
    return GEMMBlockProof(
        bi=bi,
        bj=bj,
        leaf_hash=leaf_hash,
        merkle_path=q_tree.get_path(bi, bj),
        sumcheck_proof=SumcheckProof(claimed_sum=claimed_sum, rounds=[SumcheckRound(*r) for r in rounds],
                                     final_A=final_a, final_B=final_b),
        spot_X=[SpotCheck(row=r, col=c, value=v) for r, c, v in spot_x],
        spot_W=[SpotCheck(row=r, col=c, value=v) for r, c, v, _ in spot_w],
        m_bits=_bits(int(x_rows.shape[0])),
        n_bits=_bits(int(w_cols.shape[1])),
        k_bits=len(rounds),
    )
