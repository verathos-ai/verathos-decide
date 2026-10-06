"""Focused native PCS tests for proof protocol v2."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import struct

import pytest

from verathos.zk.crypto.pcs_v2 import (
    ENCODING_PALLAS_SCALAR,
    ENCODING_SIGNED_I8,
    ENCODING_SIGNED_I64,
    MAX_BATCH_COMMITMENTS,
    MAX_COMBINE_TERMS,
    MAX_VALIDATE_COMMITMENTS,
    MAX_VECTOR_LEN,
    PALLAS_SCALAR_MODULUS,
    PCSFormatError,
    combine_commitments,
    commit,
    commit_i8_batch,
    encode_scalar,
    evaluate,
    evaluate_i8_matrix,
    evaluate_lean_projection_terminal_relation,
    fold_u31_linear_coefficients,
    native_library_path,
    proof_size,
    prove,
    prove_i64_linear_combination_linear,
    prove_linear,
    prove_i8_linear_combination,
    scalar_to_int,
    validate_commitments,
    verify,
    verify_linear,
)


def test_combine_term_cap_covers_hardened_qwen_batch():
    # Four selected layers with at most five registered projections, plus four
    # LM-head blocks, each opening sixteen W columns: 4*5*16 + 4*16 = 384.
    assert MAX_COMBINE_TERMS >= 384


OUTER_DIGEST = bytes(range(32))


def test_local_native_library_is_loaded() -> None:
    path = native_library_path()
    assert path.is_file()
    assert "verathos_pcs_v2" in path.name


def test_i8_non_power_of_two_roundtrip_is_exact_and_deterministic() -> None:
    values = [-3, 5, 7, -11, 13]
    point = [2, 3, 5]

    first = prove(values, point, OUTER_DIGEST, encoding="i8")
    second = prove(values, point, OUTER_DIGEST, encoding=ENCODING_SIGNED_I8)

    assert first == second
    assert first.vector_length == 5
    assert first.padded_length == 8
    assert len(first.commitment) == 32
    assert len(first.evaluation) == 32
    assert len(first.proof) == proof_size(len(values))
    assert commit(values, encoding="signed_i8") == first.commitment
    assert evaluate(values, point, encoding="int8") == first.evaluation
    assert verify(first, point, OUTER_DIGEST)

    explicitly_padded = values + [0, 0, 0]
    assert commit(explicitly_padded) == first.commitment
    padded_opening = prove(explicitly_padded, point, OUTER_DIGEST)
    assert padded_opening.vector_length == padded_opening.padded_length == 8
    assert verify(padded_opening, point, OUTER_DIGEST)


def test_signed_i64_and_msb_first_boolean_point_order() -> None:
    values = [-3, 7, 11]
    assert scalar_to_int(evaluate(values, [0, 1], encoding="i64")) == 7
    assert scalar_to_int(evaluate(values, [1, 0], encoding="i64")) == 11

    opening = prove(values, [9, 10], OUTER_DIGEST, encoding=ENCODING_SIGNED_I64)
    assert opening.encoding == ENCODING_SIGNED_I64
    assert verify(opening, [9, 10], OUTER_DIGEST)

    extrema = [-(1 << 63), (1 << 63) - 1]
    extrema_opening = prove(extrema, [4], OUTER_DIGEST, encoding="int64")
    assert verify(extrema_opening, [4], OUTER_DIGEST)


@pytest.mark.parametrize(
    ("values", "encoding"),
    (
        ([-3, 5, 7, -11, 13], "i8"),
        ([-(1 << 40), 7, 11, (1 << 41), -9], "i64"),
    ),
)
def test_arbitrary_linear_opening_is_exact_and_statement_bound(
    values,
    encoding,
) -> None:
    coefficients = tuple(index * 17 + 3 for index in range(8))
    opening = prove_linear(
        values,
        coefficients,
        OUTER_DIGEST,
        encoding=encoding,
    )
    expected = sum(
        value * coefficient
        for value, coefficient in zip(values, coefficients, strict=False)
    ) % PALLAS_SCALAR_MODULUS
    assert scalar_to_int(opening.evaluation) == expected
    assert opening.commitment == commit(values, encoding=encoding)
    assert verify_linear(opening, coefficients, OUTER_DIGEST)

    changed = list(coefficients)
    changed[0] += 1
    assert not verify_linear(opening, changed, OUTER_DIGEST)
    wrong_outer = bytes((OUTER_DIGEST[0] ^ 1,)) + OUTER_DIGEST[1:]
    assert not verify_linear(opening, coefficients, wrong_outer)
    wrong_evaluation = encode_scalar((expected + 1) % PALLAS_SCALAR_MODULUS)
    assert not verify_linear(
        replace(opening, evaluation=wrong_evaluation),
        coefficients,
        OUTER_DIGEST,
    )
    other_commitment = commit(
        [value + 1 for value in values],
        encoding=encoding,
    )
    assert not verify_linear(
        replace(opening, commitment=other_commitment),
        coefficients,
        OUTER_DIGEST,
    )
    other_encoding = "i64" if encoding == "i8" else "i8"
    assert not verify_linear(
        replace(opening, encoding=other_encoding),
        coefficients,
        OUTER_DIGEST,
    )


def test_arbitrary_linear_opening_rejects_wrong_shape_or_encoding() -> None:
    with pytest.raises(PCSFormatError, match="coefficient count"):
        prove_linear([1, 2, 3], [1, 2, 3], OUTER_DIGEST)
    with pytest.raises(PCSFormatError, match="support signed"):
        prove_linear(
            [1, 2],
            [3, 4],
            OUTER_DIGEST,
            encoding=ENCODING_PALLAS_SCALAR,
        )


def test_i64_linear_combination_linear_opening_matches_commitments() -> None:
    vectors = (
        (-3, 5, 7, -11),
        (13, -17, 19, 23),
        (29, 31, -37, 41),
    )
    combination = (3, 5, 7)
    linear = (11, 13, 17, 19)
    raw = b"".join(
        struct.pack("<4q", *vector) for vector in vectors
    )
    opening = prove_i64_linear_combination_linear(
        raw,
        vector_length=4,
        combination_coefficients=combination,
        linear_coefficients=linear,
        outer_digest=OUTER_DIGEST,
    )
    expected_commitment = combine_commitments(
        tuple(commit(vector, encoding="i64") for vector in vectors),
        combination,
    )
    expected_evaluation = sum(
        combination[term] * linear[index] * vectors[term][index]
        for term in range(len(vectors))
        for index in range(4)
    ) % PALLAS_SCALAR_MODULUS
    assert opening.commitment == expected_commitment
    assert scalar_to_int(opening.evaluation) == expected_evaluation
    assert verify_linear(opening, linear, OUTER_DIGEST)

    changed = list(linear)
    changed[2] += 1
    assert not verify_linear(opening, changed, OUTER_DIGEST)


def test_i64_linear_combination_linear_rejects_ragged_input() -> None:
    with pytest.raises(PCSFormatError, match="wrong length"):
        prove_i64_linear_combination_linear(
            b"\0" * 24,
            vector_length=4,
            combination_coefficients=(1,),
            linear_coefficients=(1, 2, 3, 4),
            outer_digest=OUTER_DIGEST,
        )


def test_packed_linear_coefficients_are_validated_by_native_backend() -> None:
    coefficients = tuple(index * 17 + 3 for index in range(8))
    packed = b"".join(
        int(value).to_bytes(32, "little") for value in coefficients
    )
    opening = prove_linear(
        [1, -2, 3, -4, 5],
        packed,
        OUTER_DIGEST,
        encoding="i8",
    )
    assert verify_linear(opening, packed, OUTER_DIGEST)

    noncanonical = PALLAS_SCALAR_MODULUS.to_bytes(32, "little") * 8
    with pytest.raises(PCSFormatError, match="scalar"):
        verify_linear(opening, noncanonical, OUTER_DIGEST)


def test_native_u31_linear_fold_matches_exact_scalar_arithmetic() -> None:
    claim_count = 3
    term_count = 5
    fold_count = 4
    padded = 16
    coefficient_values = tuple(
        (claim * 101 + fold * 17 + term * 3 + 1) & ((1 << 31) - 1)
        for claim in range(claim_count)
        for fold in range(fold_count)
        for term in range(term_count)
    )
    coefficients = b"".join(
        value.to_bytes(4, "little") for value in coefficient_values
    )
    betas = tuple(
        (index + 2) * (1 << 80) + 7
        for index in range(claim_count * fold_count)
    )
    packed = fold_u31_linear_coefficients(
        coefficients,
        betas,
        claim_count=claim_count,
        term_count=term_count,
        fold_count=fold_count,
        padded_length=padded,
    )
    decoded = tuple(
        int.from_bytes(packed[offset : offset + 32], "little")
        for offset in range(0, len(packed), 32)
    )
    expected = []
    for claim in range(claim_count):
        for term in range(term_count):
            expected.append(
                sum(
                    betas[claim * fold_count + fold]
                    * coefficient_values[
                        (claim * fold_count + fold) * term_count + term
                    ]
                    for fold in range(fold_count)
                )
                % PALLAS_SCALAR_MODULUS
            )
    expected.extend([0] * (padded - len(expected)))
    assert decoded == tuple(expected)

    malformed = bytearray(coefficients)
    malformed[:4] = (1 << 31).to_bytes(4, "little")
    with pytest.raises(PCSFormatError, match="31-bit"):
        fold_u31_linear_coefficients(
            bytes(malformed),
            betas,
            claim_count=claim_count,
            term_count=term_count,
            fold_count=fold_count,
            padded_length=padded,
        )


def test_i8_byte_input_uses_signed_twos_complement() -> None:
    raw = bytes((0xFF, 0x02, 0x80, 0x7F))
    signed = [-1, 2, -128, 127]
    assert commit(raw) == commit(signed)
    assert evaluate(raw, [3, 4]) == evaluate(signed, [3, 4])


def test_packed_i8_batch_matches_every_serial_commitment_byte() -> None:
    columns = (
        bytes((0x80, 0xEF, 0xFF, 0x00, 0x7F)),
        bytes((5, 4, 3, 2, 1)),
        bytes((0x9D, 88, 0xB3, 66, 0xC9)),
        bytes(5),
    )
    expected = tuple(commit(column, encoding="i8") for column in columns)

    assert commit_i8_batch(b"".join(columns), vector_length=5) == expected
    assert commit_i8_batch(memoryview(b"".join(columns)), vector_length=5) == expected


def test_packed_i8_batch_rejects_nonexact_or_unbounded_inputs() -> None:
    with pytest.raises(PCSFormatError, match="must be bytes"):
        commit_i8_batch([1, 2, 3], vector_length=3)  # type: ignore[arg-type]
    with pytest.raises(PCSFormatError, match="complete vectors"):
        commit_i8_batch(b"", vector_length=3)
    with pytest.raises(PCSFormatError, match="complete vectors"):
        commit_i8_batch(b"\x00" * 5, vector_length=3)
    with pytest.raises(PCSFormatError, match="batch count"):
        commit_i8_batch(
            b"\x00" * (MAX_BATCH_COMMITMENTS + 1),
            vector_length=1,
        )


def test_native_i8_matrix_evaluation_matches_flat_mle_order() -> None:
    values = bytes((0x80, 0xFF, 0, 1, 2, 3, 4, 0x7F))
    point = (5, 7, 11)
    assert evaluate_i8_matrix(
        values,
        rows=2,
        columns=4,
        point=point,
    ) == evaluate(values, point, encoding="i8")

    with pytest.raises(PCSFormatError, match="positive powers of two"):
        evaluate_i8_matrix(values, rows=2, columns=3, point=(5, 7))
    with pytest.raises(PCSFormatError, match="wrong length"):
        evaluate_i8_matrix(values[:-1], rows=2, columns=4, point=point)
    with pytest.raises(PCSFormatError, match="exactly 3"):
        evaluate_i8_matrix(values, rows=2, columns=4, point=(5, 7))


def test_native_lean_projection_terminal_relation_matches_reference() -> None:
    rows = (
        bytes((0x80, 0xFF, 3, 0x7F)),
        bytes((11, 0xF9, 0, 5)),
    )
    point = (17, 19)
    alphas = tuple(range(2, 10))
    terminals = tuple(range(23, 31))
    expected = 0
    for claim_index, row in enumerate(rows):
        x_evaluation = scalar_to_int(
            evaluate(row, point, encoding="i8")
        )
        fold_start = claim_index * 4
        fold_weight = sum(
            alphas[fold_start + fold] * terminals[fold_start + fold]
            for fold in range(4)
        ) % PALLAS_SCALAR_MODULUS
        expected = (
            expected + x_evaluation * fold_weight
        ) % PALLAS_SCALAR_MODULUS

    assert evaluate_lean_projection_terminal_relation(
        b"".join(rows),
        claim_count=2,
        width=4,
        point=point,
        alphas=alphas,
        terminal_fold_evaluations=terminals,
    ) == expected
    with pytest.raises(PCSFormatError, match="scalar count"):
        evaluate_lean_projection_terminal_relation(
            b"".join(rows),
            claim_count=2,
            width=4,
            point=point,
            alphas=alphas[:-1],
            terminal_fold_evaluations=terminals,
        )


def test_length_one_has_an_empty_mle_point() -> None:
    opening = prove([-9], [], OUTER_DIGEST)
    assert opening.padded_length == 1
    assert proof_size(1) == 34
    assert scalar_to_int(opening.evaluation) == PALLAS_SCALAR_MODULUS - 9
    assert verify(opening, b"", OUTER_DIGEST)


def test_every_bound_statement_component_is_enforced() -> None:
    values = [1, 2, 3, 4, 5]
    point = [6, 7, 8]
    opening = prove(values, point, OUTER_DIGEST)

    wrong_outer = bytes((OUTER_DIGEST[0] ^ 1,)) + OUTER_DIGEST[1:]
    assert not verify(opening, point, wrong_outer)

    wrong_point = [6, 7, 9]
    assert not verify(opening, wrong_point, OUTER_DIGEST)

    wrong_evaluation = encode_scalar(
        (scalar_to_int(opening.evaluation) + 1) % PALLAS_SCALAR_MODULUS
    )
    assert not verify(
        replace(opening, evaluation=wrong_evaluation), point, OUTER_DIGEST
    )

    other_commitment = commit([1, 2, 3, 4, 6])
    assert not verify(
        replace(opening, commitment=other_commitment), point, OUTER_DIGEST
    )

    tampered_scalar = encode_scalar(
        (scalar_to_int(opening.proof[-32:]) + 1) % PALLAS_SCALAR_MODULUS
    )
    assert not verify(
        replace(opening, proof=opening.proof[:-32] + tampered_scalar),
        point,
        OUTER_DIGEST,
    )

    assert not verify(
        replace(opening, encoding=ENCODING_SIGNED_I64), point, OUTER_DIGEST
    )
    assert not verify(
        replace(opening, vector_length=6, padded_length=8), point, OUTER_DIGEST
    )


def test_malformed_and_noncanonical_inputs_raise() -> None:
    point = [2, 3]
    opening = prove([1, 2, 3, 4], point, OUTER_DIGEST)

    with pytest.raises(PCSFormatError, match="IPA proof"):
        verify(replace(opening, proof=opening.proof[:-1]), point, OUTER_DIGEST)
    with pytest.raises(PCSFormatError, match="Pallas scalar"):
        verify(
            replace(
                opening,
                evaluation=PALLAS_SCALAR_MODULUS.to_bytes(32, "little"),
            ),
            point,
            OUTER_DIGEST,
        )
    with pytest.raises(PCSFormatError, match="Pallas scalar"):
        verify(
            opening,
            [PALLAS_SCALAR_MODULUS.to_bytes(32, "little"), 3],
            OUTER_DIGEST,
        )
    with pytest.raises(PCSFormatError, match="commitment"):
        verify(replace(opening, commitment=b"\xff" * 32), point, OUTER_DIGEST)
    with pytest.raises(PCSFormatError, match="padded"):
        verify(replace(opening, padded_length=16), point, OUTER_DIGEST)
    with pytest.raises(PCSFormatError, match="encoding"):
        verify(replace(opening, encoding=99), point, OUTER_DIGEST)


def test_caps_scalar_encoding_and_exact_point_lengths() -> None:
    with pytest.raises(PCSFormatError, match="vector length"):
        proof_size(0)
    with pytest.raises(PCSFormatError, match="vector length"):
        proof_size(MAX_VECTOR_LEN + 1)
    with pytest.raises(PCSFormatError, match="canonical field range"):
        encode_scalar(PALLAS_SCALAR_MODULUS)
    with pytest.raises(PCSFormatError, match="canonical field range"):
        encode_scalar(1 << 300)
    with pytest.raises(PCSFormatError, match="exactly 3"):
        prove([1, 2, 3, 4, 5], [1, 2], OUTER_DIGEST)
    with pytest.raises(PCSFormatError, match="signed 8-bit"):
        commit([128], encoding="i8")
    with pytest.raises(PCSFormatError, match="signed 64-bit"):
        commit([1 << 63], encoding="i64")

    assert proof_size(MAX_VECTOR_LEN) == 2 + 64 * 17 + 32


def test_cached_setup_verification_is_thread_safe() -> None:
    values = [1, -1, 2, -2, 3, -3, 4, -4]
    point = [2, 3, 4]
    opening = prove(values, point, OUTER_DIGEST)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: verify(opening, point, OUTER_DIGEST),
                range(8),
            )
        )
    assert results == [True] * 8


def test_canonical_pallas_field_vector_inputs_roundtrip() -> None:
    values = [1, PALLAS_SCALAR_MODULUS - 2, 3, 17, 99]
    encoded_values = [encode_scalar(value) for value in values]
    packed_values = b"".join(encoded_values)
    point = [5, 7, 11]

    from_integers = commit(values, encoding=ENCODING_PALLAS_SCALAR)
    assert commit(encoded_values, encoding="field") == from_integers
    assert commit(packed_values, encoding="pallas_scalar") == from_integers
    assert evaluate(values, point, encoding="fr") == evaluate(
        packed_values,
        point,
        encoding=ENCODING_PALLAS_SCALAR,
    )

    opening = prove(values, point, OUTER_DIGEST, encoding=ENCODING_PALLAS_SCALAR)
    assert opening.encoding == ENCODING_PALLAS_SCALAR
    assert opening.commitment == from_integers
    assert verify(opening, point, OUTER_DIGEST)

    noncanonical = PALLAS_SCALAR_MODULUS.to_bytes(32, "little") + packed_values[32:]
    with pytest.raises(PCSFormatError, match="canonical field range"):
        commit(noncanonical, encoding="field")
    with pytest.raises(PCSFormatError, match="32-byte scalars"):
        commit(packed_values[:-1], encoding="field")


def test_structured_commitment_combination_opens_folded_field_vector() -> None:
    rows = [
        [1, 3, 5, 7, 11, 13, 17, 19],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [23, 29, 31, 37, 41, 43, 47, 53],
    ]
    coefficients = [59, 61, 67]
    row_commitments = [commit(row, encoding="field") for row in rows]
    folded = [
        sum(coefficient * row[column] for coefficient, row in zip(coefficients, rows))
        % PALLAS_SCALAR_MODULUS
        for column in range(len(rows[0]))
    ]

    combined = combine_commitments(row_commitments, coefficients)
    assert combined == combine_commitments(
        b"".join(row_commitments),
        b"".join(encode_scalar(coefficient) for coefficient in coefficients),
    )
    assert combined == commit(folded, encoding=ENCODING_PALLAS_SCALAR)

    point = [71, 73, 79]
    opening = prove(folded, point, OUTER_DIGEST, encoding="field")
    derived_opening = replace(opening, commitment=combined)
    assert verify(derived_opening, point, OUTER_DIGEST)

    wrong_coefficients = [coefficients[0] + 1, *coefficients[1:]]
    wrong_combined = combine_commitments(row_commitments, wrong_coefficients)
    assert wrong_combined != combined
    assert not verify(replace(opening, commitment=wrong_combined), point, OUTER_DIGEST)
    assert not verify(
        replace(opening, commitment=row_commitments[0]),
        point,
        OUTER_DIGEST,
    )


def test_native_i8_linear_combination_matches_explicit_field_fold() -> None:
    rows = (
        bytes((0x80, 0xFF, 0, 1, 2, 3, 4, 0x7F)),
        bytes((7, 6, 5, 4, 3, 2, 1, 0)),
        bytes((0xF9, 8, 0xF7, 10, 0xF5, 12, 0xF3, 14)),
    )
    signed_rows = tuple(
        tuple(value if value < 128 else value - 256 for value in row) for row in rows
    )
    coefficients = (17, 19, 23)
    folded = tuple(
        sum(
            coefficient * row[index]
            for coefficient, row in zip(coefficients, signed_rows)
        )
        % PALLAS_SCALAR_MODULUS
        for index in range(8)
    )
    point = (29, 31, 37)
    opening = prove_i8_linear_combination(
        b"".join(rows),
        vector_length=8,
        coefficients=coefficients,
        point=point,
        outer_digest=OUTER_DIGEST,
    )
    expected_commitment = combine_commitments(
        tuple(commit(row, encoding="i8") for row in rows),
        coefficients,
    )
    assert opening.commitment == expected_commitment
    assert opening.commitment == commit(folded, encoding="field")
    assert opening.evaluation == evaluate(folded, point, encoding="field")
    assert verify(opening, point, OUTER_DIGEST)
    wrong_evaluation = encode_scalar(
        (scalar_to_int(opening.evaluation) + 1) % PALLAS_SCALAR_MODULUS
    )
    assert not verify(
        replace(opening, evaluation=wrong_evaluation),
        point,
        OUTER_DIGEST,
    )


def test_native_i8_linear_combination_accepts_hardened_qwen_term_count() -> None:
    term_count = 384
    rows = tuple(
        bytes(((term + column) % 255 for column in range(8)))
        for term in range(term_count)
    )
    coefficients = tuple(range(1, term_count + 1))
    opening = prove_i8_linear_combination(
        b"".join(rows),
        vector_length=8,
        coefficients=coefficients,
        point=(3, 5, 7),
        outer_digest=OUTER_DIGEST,
    )

    assert opening.commitment == combine_commitments(
        tuple(commit(row, encoding="i8") for row in rows),
        coefficients,
    )
    assert verify(opening, (3, 5, 7), OUTER_DIGEST)


def test_commitment_combination_rejects_malformed_or_unbounded_inputs() -> None:
    commitment = commit([1, 2], encoding="field")
    with pytest.raises(PCSFormatError, match="term count"):
        combine_commitments([], [])
    with pytest.raises(PCSFormatError, match="counts must match"):
        combine_commitments([commitment], [1, 2])
    with pytest.raises(PCSFormatError, match="term count"):
        combine_commitments([commitment] * (MAX_COMBINE_TERMS + 1), [1])
    with pytest.raises(PCSFormatError, match="canonical field range"):
        combine_commitments(
            [commitment],
            [PALLAS_SCALAR_MODULUS.to_bytes(32, "little")],
        )
    with pytest.raises(PCSFormatError, match="commitment"):
        combine_commitments([b"\xff" * 32], [1])


def test_direct_commitment_validation_accepts_canonical_and_rejects_malformed() -> None:
    commitments = [commit([index, index + 1]) for index in range(8)]
    assert validate_commitments(commitments) is None
    assert validate_commitments(b"".join(commitments)) is None
    with pytest.raises(PCSFormatError, match="commitment"):
        validate_commitments([*commitments, b"\xff" * 32])
    with pytest.raises(PCSFormatError, match="validation count"):
        validate_commitments(commitments * (MAX_VALIDATE_COMMITMENTS // 8 + 1))


def test_structured_combination_and_opening_concurrent_stress() -> None:
    rows = [[index + row_index for index in range(8)] for row_index in (1, 11, 23, 47)]
    coefficients = [5, 7, 11, 13]
    row_commitments = [commit(row, encoding="field") for row in rows]
    folded = [
        sum(coefficient * row[column] for coefficient, row in zip(coefficients, rows))
        % PALLAS_SCALAR_MODULUS
        for column in range(len(rows[0]))
    ]
    point = [17, 19, 23]
    opening = prove(folded, point, OUTER_DIGEST, encoding="field")

    def combine_and_verify(_: int) -> bool:
        combined = combine_commitments(row_commitments, coefficients)
        return combined == opening.commitment and verify(
            replace(opening, commitment=combined),
            point,
            OUTER_DIGEST,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(combine_and_verify, range(16)))
    assert results == [True] * 16
