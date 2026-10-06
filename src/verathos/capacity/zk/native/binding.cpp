// GEMM block sumcheck prover over the Mersenne field p = 2^61 - 1 (CPU).
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "keccak256.h"

namespace py = pybind11;

// ======================================================================
//  Field arithmetic
// ======================================================================

static constexpr uint64_t MP = (1ULL << 61) - 1;

static inline uint64_t h_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r >= MP) r -= MP;
    return r;
}

static inline uint64_t h_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (MP - b + a);
}

static inline uint64_t h_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    uint64_t lo = (uint64_t)prod;
    uint64_t hi = (uint64_t)(prod >> 64);
    uint64_t lo_lo = lo & MP;
    uint64_t lo_hi = lo >> 61;
    uint64_t sum = lo_lo + lo_hi + (hi << 3);
    uint64_t r = (sum >> 61) + (sum & MP);
    if (r >= MP) r -= MP;
    return r;
}

// eq(r, x) for every x in {0,1}^m, LSB-first.
static std::vector<uint64_t> eq_vector(const std::vector<uint64_t>& r) {
    int m = (int)r.size();
    int n = 1 << m;
    std::vector<uint64_t> eq(n, 0);
    eq[0] = 1;
    int cur_size = 1;
    for (int i = 0; i < m; i++) {
        uint64_t ri = r[i];
        uint64_t one_minus_ri = h_sub(1, ri);
        // Expand from the back so entries are read before being overwritten.
        for (int j = cur_size - 1; j >= 0; j--) {
            eq[j + cur_size] = h_mul(eq[j], ri);
            eq[j]            = h_mul(eq[j], one_minus_ri);
        }
        cur_size <<= 1;
    }
    return eq;
}

// Sumcheck round polynomial g(0), g(1), g(2) of sum_b T_A(b) * T_B(b).
static std::tuple<uint64_t, uint64_t, uint64_t>
compute_round_evals(
    const std::vector<uint64_t>& T_A,
    const std::vector<uint64_t>& T_B
) {
    int half = (int)T_A.size() / 2;
    uint64_t e0 = 0, e1 = 0, e2 = 0;
    for (int b = 0; b < half; b++) {
        uint64_t a0 = T_A[2*b], a1 = T_A[2*b + 1];
        uint64_t b0 = T_B[2*b], b1 = T_B[2*b + 1];
        e0 = h_add(e0, h_mul(a0, b0));
        e1 = h_add(e1, h_mul(a1, b1));
        uint64_t a2 = h_sub(h_add(a1, a1), a0);
        uint64_t b2 = h_sub(h_add(b1, b1), b0);
        e2 = h_add(e2, h_mul(a2, b2));
    }
    return {e0, e1, e2};
}

// Bind the lowest variable of a table to the challenge.
static std::vector<uint64_t> fold_table(
    const std::vector<uint64_t>& table, uint64_t challenge
) {
    int half = (int)table.size() / 2;
    uint64_t one_minus_r = h_sub(1, challenge);
    std::vector<uint64_t> result(half);
    for (int b = 0; b < half; b++)
        result[b] = h_add(h_mul(table[2*b], one_minus_r),
                          h_mul(table[2*b + 1], challenge));
    return result;
}

// ======================================================================
//  Block proof
// ======================================================================

// ((v % p) + p) % p
static inline uint64_t mod_p_signed(int64_t v) {
    if (v >= 0 && static_cast<uint64_t>(v) < MP)
        return static_cast<uint64_t>(v);
    int64_t r = v % static_cast<int64_t>(MP);
    if (r < 0) r += static_cast<int64_t>(MP);
    return static_cast<uint64_t>(r);
}

// Row-major [padded_rows, padded_cols] field elements, zero-padded.
template<typename T>
static std::vector<uint64_t> tensor_to_field_padded_impl(
    const T* data, int rows, int cols,
    int padded_rows, int padded_cols
) {
    std::vector<uint64_t> out(
        static_cast<size_t>(padded_rows) * static_cast<size_t>(padded_cols), 0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[static_cast<size_t>(i) * padded_cols + j] =
                mod_p_signed(static_cast<int64_t>(data[static_cast<size_t>(i) * cols + j]));
        }
    }
    return out;
}

// int8, int32 or int64 (any other dtype is read as int64).
static std::vector<uint64_t> tensor_to_field_dispatch(
    const at::Tensor& t, int rows, int cols,
    int padded_rows, int padded_cols
) {
    if (t.scalar_type() == at::kChar)
        return tensor_to_field_padded_impl(t.data_ptr<int8_t>(), rows, cols, padded_rows, padded_cols);
    else if (t.scalar_type() == at::kInt)
        return tensor_to_field_padded_impl(t.data_ptr<int32_t>(), rows, cols, padded_rows, padded_cols);
    else
        return tensor_to_field_padded_impl(t.data_ptr<int64_t>(), rows, cols, padded_rows, padded_cols);
}

// Field value of one element of a contiguous int8/int32/int64 tensor.
static inline uint64_t tensor_get_field_value(
    const at::Tensor& t, int64_t row, int64_t col, int stride
) {
    if (t.scalar_type() == at::kChar)
        return mod_p_signed(static_cast<int64_t>(t.data_ptr<int8_t>()[row * stride + col]));
    else if (t.scalar_type() == at::kInt)
        return mod_p_signed(static_cast<int64_t>(t.data_ptr<int32_t>()[row * stride + col]));
    else
        return mod_p_signed(t.data_ptr<int64_t>()[row * stride + col]);
}

// MLE evaluation by folding buf in place, one point coordinate per pass.
static uint64_t mle_evaluate_vec(
    std::vector<uint64_t>& buf, const std::vector<uint64_t>& point
) {
    size_t size = buf.size();
    for (auto ri : point) {
        size_t half = size >> 1;
        uint64_t one_minus_ri = h_sub(1, ri);
        for (size_t b = 0; b < half; b++) {
            buf[b] = h_add(h_mul(buf[2*b], one_minus_ri),
                           h_mul(buf[2*b + 1], ri));
        }
        size = half;
    }
    return buf[0];
}

// Compute T_A[k] = sum_i X_field[i,k] * eq(r_i, i)
// X_field is row-major [m_padded, k_padded]
static std::vector<uint64_t> compute_T_A_vec(
    const std::vector<uint64_t>& X_field, int m_padded, int k_padded,
    const std::vector<uint64_t>& r_i
) {
    auto eq = eq_vector(r_i);
    int eq_len = std::min(m_padded, static_cast<int>(eq.size()));
    std::vector<uint64_t> T_A(k_padded, 0);

    for (int k = 0; k < k_padded; k++) {
        uint64_t val = 0;
        for (int i = 0; i < eq_len; i++) {
            val = h_add(val, h_mul(X_field[static_cast<size_t>(i) * k_padded + k], eq[i]));
        }
        T_A[k] = val;
    }
    return T_A;
}

// Compute T_B[k] = sum_j W_field[k,j] * eq(r_j, j)
// W_field is row-major [k_padded, n_padded]
static std::vector<uint64_t> compute_T_B_vec(
    const std::vector<uint64_t>& W_field, int k_padded, int n_padded,
    const std::vector<uint64_t>& r_j
) {
    auto eq = eq_vector(r_j);
    int eq_len = std::min(n_padded, static_cast<int>(eq.size()));
    std::vector<uint64_t> T_B(k_padded, 0);

    for (int k = 0; k < k_padded; k++) {
        uint64_t val = 0;
        for (int j = 0; j < eq_len; j++) {
            val = h_add(val, h_mul(W_field[static_cast<size_t>(k) * n_padded + j], eq[j]));
        }
        T_B[k] = val;
    }
    return T_B;
}

// Proves Y = X @ W for one output block: absorbs the block position and Y leaf
// hash, claims Y's MLE at (r_i, r_j), runs the k-dimension sumcheck and samples
// X/W spot checks, all from the forked transcript state.
//
// Returns: (rounds, claimed_sum, final_A, final_B, challenges,
//           spot_X, spot_W, new_transcript_state)
static std::tuple<
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>,     // rounds
    uint64_t,                                                    // claimed_sum
    uint64_t,                                                    // final_A
    uint64_t,                                                    // final_B
    std::vector<uint64_t>,                                       // challenges
    std::vector<std::tuple<int64_t, int64_t, uint64_t>>,         // spot_X
    std::vector<std::tuple<int64_t, int64_t, uint64_t, int64_t>>,// spot_W
    py::bytes                                                    // transcript state
> native_prove_block(
    at::Tensor X_rows,           // [m, K] int8/int32/int64 (any device)
    at::Tensor W_cols,           // [K, n] int8/int32/int64 (any device)
    at::Tensor Y_block,          // [m, n] int8/int32/int64 (any device)
    py::bytes transcript_state_bytes,  // 32-byte forked transcript state
    int bi, int bj,
    py::bytes leaf_hash_bytes,   // 32-byte Y Merkle leaf hash
    int num_spots,
    int row_offset, int col_offset,
    int W_num_cols_full, int W_chunk_size
) {
    // --- Parse inputs (GIL held) ---
    std::string state_str(transcript_state_bytes);
    TORCH_CHECK(state_str.size() == 32,
        "Transcript state must be 32 bytes, got ", state_str.size());

    std::string leaf_str(leaf_hash_bytes);
    TORCH_CHECK(leaf_str.size() == 32,
        "Leaf hash must be 32 bytes, got ", leaf_str.size());

    TORCH_CHECK(X_rows.dim() == 2 && W_cols.dim() == 2 && Y_block.dim() == 2,
        "X, W, Y must be 2D tensors");

    // Keep the input dtypes; conversion to field elements reads them directly.
    auto X_cpu = X_rows.to(at::kCPU).contiguous();
    auto W_cpu = W_cols.to(at::kCPU).contiguous();
    auto Y_cpu = Y_block.to(at::kCPU).contiguous();

    int m = static_cast<int>(X_cpu.size(0));
    int K = static_cast<int>(X_cpu.size(1));
    int n = static_cast<int>(W_cpu.size(1));

    // --- All computation (release GIL) ---
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rounds;
    uint64_t claimed_sum = 0, final_A_val = 0, final_B_val = 0;
    std::vector<uint64_t> challenges;
    std::vector<std::tuple<int64_t, int64_t, uint64_t>> spot_X;
    std::vector<std::tuple<int64_t, int64_t, uint64_t, int64_t>> spot_W;
    std::array<uint8_t, 32> final_state;

    {
        py::gil_scoped_release release;

        // Dimension bits (power-of-2 padding)
        auto bit_length = [](int v) -> int {
            if (v <= 1) return 1;
            int bits = 0;
            int tmp = v - 1;
            while (tmp > 0) { bits++; tmp >>= 1; }
            return std::max(1, bits);
        };
        int m_bits = bit_length(m);
        int n_bits = bit_length(n);
        int k_bits = bit_length(K);
        int m_padded = 1 << m_bits;
        int n_padded = 1 << n_bits;
        int k_padded = 1 << k_bits;

        // ---- 1. Initialize transcript ----
        std::array<uint8_t, 32> init_state;
        std::memcpy(init_state.data(), state_str.data(), 32);
        verathos::NativeTranscript transcript(init_state);

        // ---- 2. Absorb block info ----
        transcript.absorb("bi", static_cast<uint64_t>(bi));
        transcript.absorb("bj", static_cast<uint64_t>(bj));
        transcript.absorb_bytes(
            reinterpret_cast<const uint8_t*>("leaf_hash"), 9,
            reinterpret_cast<const uint8_t*>(leaf_str.data()), 32);

        // ---- 3. Field conversion with padding ----
        auto X_field = tensor_to_field_dispatch(X_cpu, m, K, m_padded, k_padded);
        auto W_field = tensor_to_field_dispatch(W_cpu, K, n, k_padded, n_padded);
        auto Y_field = tensor_to_field_dispatch(Y_cpu, m, n, m_padded, n_padded);

        // ---- 4. Build Y MLE (column-major flatten) ----
        // evals[row + col * m_padded] = Y_field[row, col]
        std::vector<uint64_t> Y_mle(
            static_cast<size_t>(m_padded) * static_cast<size_t>(n_padded), 0);
        for (int col = 0; col < n_padded; col++) {
            for (int row = 0; row < m_padded; row++) {
                Y_mle[row + static_cast<size_t>(col) * m_padded] =
                    Y_field[static_cast<size_t>(row) * n_padded + col];
            }
        }
        Y_field.clear();  // Free

        // ---- 5. Squeeze r_i, r_j and evaluate MLE ----
        auto r_i = transcript.squeeze_n("r_i", m_bits);
        auto r_j = transcript.squeeze_n("r_j", n_bits);

        // Point is r_i || r_j.
        std::vector<uint64_t> combined_point;
        combined_point.reserve(m_bits + n_bits);
        combined_point.insert(combined_point.end(), r_i.begin(), r_i.end());
        combined_point.insert(combined_point.end(), r_j.begin(), r_j.end());
        claimed_sum = mle_evaluate_vec(Y_mle, combined_point);
        Y_mle.clear();  // Free

        // ---- 6. Compute T_A and T_B ----
        auto T_A = compute_T_A_vec(X_field, m_padded, k_padded, r_i);
        X_field.clear();  // Free

        auto T_B = compute_T_B_vec(W_field, k_padded, n_padded, r_j);

        // ---- 7. Sumcheck loop ----
        rounds.reserve(k_bits);
        challenges.reserve(k_bits);

        for (int round_idx = 0; round_idx < k_bits; round_idx++) {
            auto [e0, e1, e2] = compute_round_evals(T_A, T_B);
            rounds.emplace_back(e0, e1, e2);

            transcript.absorb("round", static_cast<uint64_t>(round_idx));
            transcript.absorb("e0", e0);
            transcript.absorb("e1", e1);
            transcript.absorb("e2", e2);

            uint64_t r = transcript.squeeze("challenge");
            challenges.push_back(r);

            T_A = fold_table(T_A, r);
            T_B = fold_table(T_B, r);
        }

        final_A_val = T_A.empty() ? 0 : T_A[0];
        final_B_val = T_B.empty() ? 0 : T_B[0];

        // ---- 8. Spot X ----
        spot_X.reserve(num_spots);
        for (int i = 0; i < num_spots; i++) {
            std::string row_label = "spot_X:row:" + std::to_string(i);
            std::string col_label = "spot_X:col:" + std::to_string(i);

            int64_t local_row = static_cast<int64_t>(
                transcript.squeeze(row_label) % static_cast<uint64_t>(m));
            int64_t local_col = static_cast<int64_t>(
                transcript.squeeze(col_label) % static_cast<uint64_t>(K));

            uint64_t value = tensor_get_field_value(X_cpu, local_row, local_col, K);
            spot_X.emplace_back(
                row_offset + local_row,
                0 + local_col,  // col_offset for X is always 0
                value);
        }

        // ---- 9. Spot W ----
        spot_W.reserve(num_spots);
        for (int i = 0; i < num_spots; i++) {
            std::string row_label = "spot_W:row:" + std::to_string(i);
            std::string col_label = "spot_W:col:" + std::to_string(i);

            int64_t local_row = static_cast<int64_t>(
                transcript.squeeze(row_label) % static_cast<uint64_t>(K));
            int64_t local_col = static_cast<int64_t>(
                transcript.squeeze(col_label) % static_cast<uint64_t>(n));

            uint64_t value = tensor_get_field_value(W_cpu, local_row, local_col, n);

            int64_t global_row = 0 + local_row;  // row_offset for W = 0
            int64_t global_col = col_offset + local_col;

            int64_t chunk_idx = -1;
            if (W_num_cols_full > 0 && W_chunk_size > 0) {
                int64_t flat_idx = global_row * W_num_cols_full + global_col;
                chunk_idx = flat_idx / W_chunk_size;
            }

            spot_W.emplace_back(global_row, global_col, value, chunk_idx);
        }

        W_field.clear();  // Free

        final_state = transcript.get_state();
    }
    // GIL re-acquired

    py::bytes state_out(
        reinterpret_cast<const char*>(final_state.data()), 32);

    return {rounds, claimed_sum, final_A_val, final_B_val,
            challenges, spot_X, spot_W, state_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GEMM block sumcheck prover";

    m.def("native_prove_block", &native_prove_block,
          py::arg("X_rows"), py::arg("W_cols"), py::arg("Y_block"),
          py::arg("transcript_state"),
          py::arg("bi"), py::arg("bj"), py::arg("leaf_hash"),
          py::arg("num_spots"),
          py::arg("row_offset"), py::arg("col_offset"),
          py::arg("W_num_cols_full"), py::arg("W_chunk_size"),
          "Prove one GEMM output block: field conversion, Y MLE claim, sumcheck, spot checks");
}
