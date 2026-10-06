from __future__ import annotations

from dataclasses import replace

import pytest
import numpy as np

from verathos.zk.crypto.gemm_v2_batch import (
    GemmV2BatchProof,
    GemmV2BatchStatement,
    prove_gemm_v2_batch_reference,
    verify_gemm_v2_batch_sumcheck,
)
from verathos.zk.crypto.gemm_v2_reference import (
    GemmV2Statement,
    GemmV2SumcheckRound,
    GemmV2VerificationError,
    PALLAS_SCALAR_MODULUS,
)


def _statement(index: int) -> GemmV2Statement:
    return GemmV2Statement(
        outer_digest=bytes([index + 1]) * 32,
        operation_identity=b"operation-" + str(index).encode(),
        row_block=index,
        column_block=index,
        rows=1,
        inner=8,
        valid_inner=5 + index,
        columns=4,
        x_commitment=b"x" + bytes([index]) * 31,
        w_commitment=b"w" + bytes([index]) * 31,
        y_commitment=b"y" + bytes([index]) * 31,
    )


def _w(index: int) -> list[list[int]]:
    return [
        [((row + 1) * (column + index + 2)) % 11 - 5 for column in range(4)]
        for row in range(8)
    ]


def _witnesses():
    statements = tuple(_statement(index) for index in range(4))
    xs = []
    ws = []
    ys = []
    for index, statement in enumerate(statements):
        x = [[index + row - 2 for row in range(statement.valid_inner)] + [0] * (8 - statement.valid_inner)]
        w = _w(index)
        for row in range(statement.valid_inner, 8):
            w[row] = [0] * 4
        y = [[
            sum(x[0][inner] * w[inner][column] for inner in range(statement.valid_inner))
            for column in range(4)
        ]]
        xs.append(x)
        ws.append(w)
        ys.append(y)
    return GemmV2BatchStatement(statements), xs, ws, ys


def test_batch_sumcheck_roundtrip_and_canonical_encoding():
    statement, xs, ws, ys = _witnesses()
    proof = prove_gemm_v2_batch_reference(statement, xs, ws, ys)
    encoded = proof.canonical_bytes()
    decoded = GemmV2BatchProof.from_canonical_bytes(
        encoded,
        expected_blocks=4,
        expected_rounds=3,
    )
    claims = verify_gemm_v2_batch_sumcheck(statement, decoded)
    assert decoded == proof
    assert len(claims.inner_challenges) == 3
    assert len(claims.x_values) == 4
    assert claims.aggregation_coefficients[0] == 1


def test_native_batch_prover_matches_the_reference_transcript_exactly():
    from verathos.zk.crypto.pcs_v2 import prove_gemm_batch_sumcheck_i8

    statement, xs, ws, ys = _witnesses()
    expected = prove_gemm_v2_batch_reference(statement, xs, ws, ys)
    x_values = np.asarray(xs, dtype=np.int8)
    w_values = np.asarray(ws, dtype=np.int8)
    y_values = np.asarray(ys, dtype="<i8")
    encoded = prove_gemm_batch_sumcheck_i8(
        batch_digest=statement.digest(),
        x_values=x_values.tobytes(),
        w_values=w_values.tobytes(),
        y_values_i64_le=y_values.tobytes(),
        valid_inner=tuple(item.valid_inner for item in statement.statements),
        block_count=4,
        rows=1,
        inner=8,
        columns=4,
    )
    assert encoded == expected.canonical_bytes()
    verify_gemm_v2_batch_sumcheck(
        statement,
        GemmV2BatchProof.from_canonical_bytes(encoded),
    )


def test_batch_statement_order_and_every_claim_are_bound():
    statement, xs, ws, ys = _witnesses()
    proof = prove_gemm_v2_batch_reference(statement, xs, ws, ys)
    reordered = GemmV2BatchStatement(tuple(reversed(statement.statements)))
    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_batch_sumcheck(reordered, proof)

    changed = list(proof.claimed_y_evals)
    changed[2] = (changed[2] + 1) % PALLAS_SCALAR_MODULUS
    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_batch_sumcheck(
            statement,
            replace(proof, claimed_y_evals=tuple(changed)),
        )


def test_batch_rejects_tampered_round_and_terminal_claim():
    statement, xs, ws, ys = _witnesses()
    proof = prove_gemm_v2_batch_reference(statement, xs, ws, ys)
    rounds = list(proof.rounds)
    first = rounds[0]
    rounds[0] = GemmV2SumcheckRound(
        (first.e0 + 1) % PALLAS_SCALAR_MODULUS,
        first.e1,
        first.e2,
        first.e3,
    )
    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_batch_sumcheck(statement, replace(proof, rounds=tuple(rounds)))

    x_values = list(proof.final_x_evals)
    x_values[1] = (x_values[1] + 1) % PALLAS_SCALAR_MODULUS
    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_batch_sumcheck(
            statement,
            replace(proof, final_x_evals=tuple(x_values)),
        )


def test_batch_uses_maximum_padded_dimensions_for_heterogeneous_statements():
    first = _statement(0)
    second = replace(first, rows=2, inner=16, valid_inner=9, columns=8)
    batch = GemmV2BatchStatement((first, second))
    xs = (
        [[1, -2, 3, -4, 5, 0, 0, 0]],
        [
            [((row + column) % 7) - 3 for column in range(16)]
            for row in range(2)
        ],
    )
    ws = (
        [[((row + column) % 5) - 2 for column in range(4)] for row in range(8)],
        [[((2 * row + column) % 9) - 4 for column in range(8)] for row in range(16)],
    )
    for index, statement in enumerate(batch.statements):
        for data_row in xs[index]:
            for inner in range(statement.valid_inner, statement.inner):
                data_row[inner] = 0
        for row in range(statement.valid_inner, statement.inner):
            ws[index][row] = [0] * statement.columns
    ys = tuple(
        [
            [
                sum(
                    x[row][inner] * w[inner][column]
                    for inner in range(statement.valid_inner)
                )
                for column in range(statement.columns)
            ]
            for row in range(statement.rows)
        ]
        for statement, x, w in zip(batch.statements, xs, ws)
    )

    assert (batch.rows, batch.inner, batch.columns) == (2, 16, 8)
    assert (batch.row_bits, batch.inner_bits, batch.column_bits) == (1, 4, 3)
    proof = prove_gemm_v2_batch_reference(batch, xs, ws, ys)
    verify_gemm_v2_batch_sumcheck(batch, proof)

    from verathos.zk.crypto.pcs_v2 import prove_gemm_batch_sumcheck_i8

    x_values = np.zeros((2, 2, 16), dtype=np.int8)
    w_values = np.zeros((2, 16, 8), dtype=np.int8)
    y_values = np.zeros((2, 2, 8), dtype="<i8")
    for index, statement in enumerate(batch.statements):
        x_values[index, : statement.rows, : statement.inner] = xs[index]
        w_values[index, : statement.inner, : statement.columns] = ws[index]
        y_values[index, : statement.rows, : statement.columns] = ys[index]
    encoded = prove_gemm_batch_sumcheck_i8(
        batch_digest=batch.digest(),
        x_values=x_values.tobytes(),
        w_values=w_values.tobytes(),
        y_values_i64_le=y_values.tobytes(),
        valid_inner=tuple(item.valid_inner for item in batch.statements),
        block_count=2,
        rows=batch.rows,
        inner=batch.inner,
        columns=batch.columns,
    )
    assert encoded == proof.canonical_bytes()
    verify_gemm_v2_batch_sumcheck(
        batch,
        GemmV2BatchProof.from_canonical_bytes(encoded),
    )
