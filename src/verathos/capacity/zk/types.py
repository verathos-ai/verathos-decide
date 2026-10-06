"""Proof records of the Q block sumcheck."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpotCheck:
    row: int
    col: int
    value: int  # field element


@dataclass
class SumcheckRound:
    """Round polynomial g evaluated at 0, 1 and 2."""

    e0: int
    e1: int
    e2: int


@dataclass
class SumcheckProof:
    claimed_sum: int
    rounds: list[SumcheckRound]
    final_A: int
    final_B: int


@dataclass
class MerklePath:
    leaf_index: int
    siblings: list[tuple[bytes, bool]]  # (sibling hash, sibling is on the left)


@dataclass
class GEMMBlockProof:
    """Sumcheck proof for output block (bi, bj) with its Merkle opening and operand spot checks."""

    bi: int
    bj: int
    leaf_hash: bytes
    merkle_path: MerklePath
    sumcheck_proof: SumcheckProof
    spot_X: list[SpotCheck]
    spot_W: list[SpotCheck]
    m_bits: int = 0
    n_bits: int = 0
    k_bits: int = 0
