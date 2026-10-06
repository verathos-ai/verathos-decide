import hashlib
import struct
from dataclasses import replace

import pytest

from verathos.zk.crypto.gemm_v2_reference import (
    GemmV2FormatError,
    GemmV2Proof,
    GemmV2Statement,
    GemmV2SumcheckRound,
    GemmV2VerificationError,
    PALLAS_SCALAR_MODULUS,
    derive_gemm_v2_index_challenges,
    prove_gemm_v2_reference,
    scalar_from_bytes,
    scalar_to_bytes,
    verify_gemm_v2_sumcheck,
)


def _digest(label: bytes) -> bytes:
    return hashlib.sha256(label).digest()


def _statement(**changes) -> GemmV2Statement:
    values = {
        "outer_digest": _digest(b"outer-context"),
        "operation_identity": b"layer=7;operation=mlp.down_proj;expert=none",
        "row_block": 3,
        "column_block": 5,
        "rows": 4,
        "inner": 8,
        "valid_inner": 8,
        "columns": 4,
        "x_commitment": _digest(b"x-band"),
        "w_commitment": _digest(b"w-band"),
        "y_commitment": _digest(b"y-block"),
    }
    values.update(changes)
    return GemmV2Statement(**values)


def _matrices():
    x = [
        [2, -1, 3, 0, 1, 4, -2, 5],
        [0, 2, -3, 1, 4, -1, 2, 3],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [-2, 3, 0, 4, 2, 1, -3, 2],
    ]
    w = [
        [1, 0, 2, -1],
        [2, -2, 1, 3],
        [-1, 4, 0, 2],
        [3, 1, -2, 0],
        [0, 2, 3, 1],
        [4, -1, 1, -2],
        [-2, 3, 2, 2],
        [1, 2, -1, 4],
    ]
    y = [
        [sum(x[row][k] * w[k][column] for k in range(8)) for column in range(4)]
        for row in range(4)
    ]
    return x, w, y


def _eq_weight(point, index):
    value = 1
    for point_index, challenge in enumerate(point):
        bit = len(point) - point_index - 1
        factor = challenge if index & (1 << bit) else 1 - challenge
        value = value * factor % PALLAS_SCALAR_MODULUS
    return value


def _matrix_mle(matrix, first_point, second_point):
    value = 0
    for first_index, row in enumerate(matrix):
        first_weight = _eq_weight(first_point, first_index)
        for second_index, entry in enumerate(row):
            second_weight = _eq_weight(second_point, second_index)
            value += entry * first_weight * second_weight
    return value % PALLAS_SCALAR_MODULUS


def test_honest_padded_block_proof_and_opening_points():
    statement = _statement()
    x, w, y = _matrices()

    proof = prove_gemm_v2_reference(statement, x, w, y)
    claims = verify_gemm_v2_sumcheck(statement, proof)

    assert len(proof.rounds) == 3
    assert len(claims.row_challenges) == 2
    assert len(claims.column_challenges) == 2
    assert len(claims.inner_challenges) == 3
    assert claims.x_k_point == claims.inner_challenges
    assert claims.w_k_point == claims.inner_challenges
    assert claims.validity_mask_value == 1
    assert claims.x_point == claims.row_challenges + claims.inner_challenges
    assert claims.w_point == claims.inner_challenges + claims.column_challenges
    assert claims.y_point == claims.row_challenges + claims.column_challenges
    assert claims.pcs_outer_digest == statement.digest()
    assert len(claims.pcs_outer_digest) == 32
    assert claims.x_value == proof.final_x_eval
    assert claims.w_value == proof.final_w_eval
    assert claims.y_value == proof.claimed_y_eval
    assert claims.x_value == _matrix_mle(
        x,
        claims.row_challenges,
        claims.inner_challenges,
    )
    assert claims.w_value == _matrix_mle(
        w,
        claims.inner_challenges,
        claims.column_challenges,
    )
    assert claims.y_value == _matrix_mle(
        y,
        claims.row_challenges,
        claims.column_challenges,
    )


def test_proof_encoding_round_trip_uses_canonical_little_endian_scalars():
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())

    encoded = proof.canonical_bytes()
    decoded = GemmV2Proof.from_canonical_bytes(
        encoded,
        expected_rounds=statement.inner_bits,
    )

    assert decoded == proof
    assert encoded[8:40] == scalar_to_bytes(proof.claimed_y_eval)
    assert scalar_from_bytes(scalar_to_bytes(PALLAS_SCALAR_MODULUS - 1)) == (
        PALLAS_SCALAR_MODULUS - 1
    )


def test_canonical_transcript_and_proof_vector():
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())
    claims = verify_gemm_v2_sumcheck(statement, proof)

    assert statement.digest().hex() == (
        "d00360c38d256db4e5fbeb357a4c810de3e6e498f632beb6f5afb9bbe08caf6c"
    )
    assert [scalar_to_bytes(value).hex() for value in claims.row_challenges] == [
        "b4b1071a6c1ff499b6f3685dd1874cca9614eb29d68c7322df3a433c7f1c2e0c",
        "0b9ce41c84f439bc680433b0837dd346e1d9454b76a179095969610368bc7828",
    ]
    assert [scalar_to_bytes(value).hex() for value in claims.column_challenges] == [
        "480311129aca67464812073d7637b7d43920003fc1fee2dd44cc92528d7f7501",
        "11a22fbc3a40e10f15332a27ff89933edce3ca57031a4fc3daf7c50c6b477600",
    ]
    assert [scalar_to_bytes(value).hex() for value in claims.inner_challenges] == [
        "07d6e730ed57a0bdfb52da2b31df7c7c2ade42921f9658f883f218e0dfd12437",
        "f2f57cd0a3ac172bbfa6384c61f8c7ead233c72c5d8849db70961a06d2aef10a",
        "6525223a8e86218f1e632f365f8afe4a96f467697f7aa745a9427767da1b5215",
    ]
    assert len(proof.canonical_bytes()) == 488
    assert hashlib.sha256(proof.canonical_bytes()).hexdigest() == (
        "ccd5dd38d271677990933ac8dfc14704bd40b521108d79eba57416ca3b53690e"
    )


def test_y_commitment_precedes_row_and_column_challenges():
    statement = _statement()
    original_points = derive_gemm_v2_index_challenges(statement)

    changed_points = derive_gemm_v2_index_challenges(
        replace(statement, y_commitment=_digest(b"different-y-block"))
    )

    assert changed_points != original_points


def test_statement_digest_binds_exact_pcs_context():
    statement = _statement()

    assert statement.digest() == _statement().digest()
    assert replace(statement, column_block=statement.column_block + 1).digest() != (
        statement.digest()
    )
    assert replace(statement, y_commitment=_digest(b"other-y")).digest() != (
        statement.digest()
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("outer_digest", _digest(b"other-outer-context")),
        ("operation_identity", b"layer=8;operation=mlp.down_proj;expert=none"),
        ("row_block", 4),
        ("column_block", 6),
        ("x_commitment", _digest(b"other-x-band")),
        ("w_commitment", _digest(b"other-w-band")),
        ("y_commitment", _digest(b"other-y-block")),
        ("valid_inner", 7),
    ),
)
def test_statement_context_tampering_invalidates_proof(field_name, replacement):
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())
    changed = replace(statement, **{field_name: replacement})

    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_sumcheck(changed, proof)


@pytest.mark.parametrize("target", ("round", "final_x", "final_w"))
def test_arithmetic_proof_tampering_is_rejected(target):
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())

    if target == "round":
        first = proof.rounds[0]
        changed_first = replace(first, e0=(first.e0 + 1) % PALLAS_SCALAR_MODULUS)
        changed = replace(proof, rounds=(changed_first, *proof.rounds[1:]))
    elif target == "final_x":
        changed = replace(
            proof,
            final_x_eval=(proof.final_x_eval + 1) % PALLAS_SCALAR_MODULUS,
        )
    else:
        changed = replace(
            proof,
            final_w_eval=(proof.final_w_eval + 1) % PALLAS_SCALAR_MODULUS,
        )

    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_sumcheck(statement, changed)


def test_incorrect_output_block_does_not_produce_an_acceptable_proof():
    statement = _statement()
    x, w, y = _matrices()
    y[2][1] += 1

    proof = prove_gemm_v2_reference(statement, x, w, y)

    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_sumcheck(statement, proof)


def test_round_count_is_derived_from_the_inner_dimension():
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())
    shortened = replace(proof, rounds=proof.rounds[:-1])

    with pytest.raises(GemmV2VerificationError, match="round count"):
        verify_gemm_v2_sumcheck(statement, shortened)
    with pytest.raises(GemmV2FormatError, match="round count"):
        GemmV2Proof.from_canonical_bytes(
            shortened.canonical_bytes(),
            expected_rounds=statement.inner_bits,
        )


def test_malformed_proof_encodings_are_rejected():
    statement = _statement()
    proof = prove_gemm_v2_reference(statement, *_matrices())
    encoded = proof.canonical_bytes()

    with pytest.raises(GemmV2FormatError, match="truncated|invalid length"):
        GemmV2Proof.from_canonical_bytes(encoded[:-1])
    with pytest.raises(GemmV2FormatError, match="invalid length"):
        GemmV2Proof.from_canonical_bytes(encoded + b"\x00")

    noncanonical = bytearray(encoded)
    noncanonical[8:40] = PALLAS_SCALAR_MODULUS.to_bytes(32, "little")
    with pytest.raises(GemmV2FormatError, match="not canonical"):
        GemmV2Proof.from_canonical_bytes(bytes(noncanonical))

    wrong_round_count = bytearray(encoded)
    struct.pack_into("<H", wrong_round_count, 6, 0)
    with pytest.raises(GemmV2FormatError, match="invalid length"):
        GemmV2Proof.from_canonical_bytes(bytes(wrong_round_count))


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    (
        ("outer_digest", b"short", "exactly 32"),
        ("operation_identity", b"", "non-empty"),
        ("rows", 3, "power of two"),
        ("inner", 0, "power of two"),
        ("valid_inner", 0, r"\[1, inner\]"),
        ("valid_inner", 9, r"\[1, inner\]"),
        ("columns", 6, "power of two"),
        ("row_block", -1, "uint64"),
        ("x_encoding_id", 2, "not supported"),
        ("x_encoding_id", True, "must be an integer"),
        ("w_encoding_id", 2, "not supported"),
        ("y_encoding_id", 1, "not supported"),
    ),
)
def test_malformed_statement_is_rejected(field_name, replacement, message):
    with pytest.raises(GemmV2FormatError, match=message):
        _statement(**{field_name: replacement})


def test_matrix_shapes_and_field_range_are_exact():
    statement = _statement()
    x, w, y = _matrices()

    with pytest.raises(GemmV2FormatError, match="row count"):
        prove_gemm_v2_reference(statement, x[:-1], w, y)
    x[0][0] = PALLAS_SCALAR_MODULUS
    with pytest.raises(GemmV2FormatError, match="field range"):
        prove_gemm_v2_reference(statement, x, w, y)


def test_scalar_and_round_constructors_reject_noncanonical_values():
    with pytest.raises(GemmV2FormatError, match="not a canonical"):
        scalar_to_bytes(PALLAS_SCALAR_MODULUS)
    with pytest.raises(GemmV2FormatError, match="not a canonical"):
        GemmV2SumcheckRound(0, 0, 0, PALLAS_SCALAR_MODULUS)


def test_non_power_of_two_logical_inner_uses_the_canonical_mask():
    statement = _statement(
        valid_inner=5,
        x_commitment=_digest(b"x-band-k5"),
        w_commitment=_digest(b"w-band-k5"),
        y_commitment=_digest(b"y-block-k5"),
    )
    x, w, _ = _matrices()
    y = [
        [sum(x[row][k] * w[k][column] for k in range(5)) for column in range(4)]
        for row in range(4)
    ]

    proof = prove_gemm_v2_reference(statement, x, w, y)
    claims = verify_gemm_v2_sumcheck(statement, proof)

    assert len(proof.rounds) == 3
    assert 0 <= claims.validity_mask_value < PALLAS_SCALAR_MODULUS


def test_validity_mask_matches_the_prefix_mle_for_every_logical_width():
    x, w, _ = _matrices()
    for valid_inner in range(1, 9):
        statement = _statement(
            valid_inner=valid_inner,
            x_commitment=_digest(b"x-mask" + bytes([valid_inner])),
            w_commitment=_digest(b"w-mask" + bytes([valid_inner])),
            y_commitment=_digest(b"y-mask" + bytes([valid_inner])),
        )
        y = [
            [
                sum(
                    x[row][k] * w[k][column]
                    for k in range(valid_inner)
                )
                for column in range(4)
            ]
            for row in range(4)
        ]
        proof = prove_gemm_v2_reference(statement, x, w, y)
        claims = verify_gemm_v2_sumcheck(statement, proof)
        expected_mask = sum(
            _eq_weight(claims.inner_challenges, index)
            for index in range(valid_inner)
        ) % PALLAS_SCALAR_MODULUS

        assert claims.validity_mask_value == expected_mask


def test_non_power_of_two_padding_does_not_change_the_gemm_relation():
    statement = _statement(
        valid_inner=5,
        x_commitment=_digest(b"x-band-arbitrary-padding"),
        w_commitment=_digest(b"w-band-arbitrary-padding"),
        y_commitment=_digest(b"y-block-arbitrary-padding"),
    )
    x, w, _ = _matrices()
    for row in range(4):
        x[row][5:] = [101 + row, -77 - row, 63 + row]
    w[5:] = [
        [90, -80, 70, -60],
        [-50, 40, -30, 20],
        [10, 11, 12, 13],
    ]
    y = [
        [sum(x[row][k] * w[k][column] for k in range(5)) for column in range(4)]
        for row in range(4)
    ]

    proof = prove_gemm_v2_reference(statement, x, w, y)

    verify_gemm_v2_sumcheck(statement, proof)


def test_non_power_of_two_valid_region_tampering_is_rejected():
    statement = _statement(
        valid_inner=5,
        x_commitment=_digest(b"x-band-k5-tamper"),
        w_commitment=_digest(b"w-band-k5-tamper"),
        y_commitment=_digest(b"y-block-k5-tamper"),
    )
    x, w, _ = _matrices()
    y = [
        [sum(x[row][k] * w[k][column] for k in range(5)) for column in range(4)]
        for row in range(4)
    ]
    y[1][2] += 1

    proof = prove_gemm_v2_reference(statement, x, w, y)

    with pytest.raises(GemmV2VerificationError):
        verify_gemm_v2_sumcheck(statement, proof)
