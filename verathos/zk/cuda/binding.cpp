// binding.cpp — PyBind11 module for zkllm_native
//
// Native CPU operations: fast field arithmetic, sumcheck tables, MLE, eq table
// Optional CUDA operations: GPU-accelerated sumcheck
//
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <cstdint>
#include <vector>

#include "keccak256.h"

#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace py = pybind11;

// ======================================================================
//  CUDA function declarations (defined in sumcheck_kernels.cu)
// ======================================================================

#ifdef WITH_CUDA
at::Tensor cuda_fold_table(at::Tensor table, uint64_t r, int device_index);
std::tuple<uint64_t, uint64_t, uint64_t> cuda_compute_round_evals(
    at::Tensor T_A, at::Tensor T_B, int device_index);
at::Tensor cuda_build_eq_table(at::Tensor point, int device_index);
at::Tensor cuda_field_mul_batch(at::Tensor a, at::Tensor b, int device_index);
at::Tensor cuda_int64_matmul(at::Tensor X, at::Tensor W, int device_index);
at::Tensor cuda_int8_matmul(at::Tensor X, at::Tensor W, int device_index);
at::Tensor cuda_field_sum_axis0(at::Tensor input, int device_index);
at::Tensor cuda_field_sum_axis1(at::Tensor input, int device_index);
at::Tensor cuda_compute_T_fixing_row(at::Tensor M, at::Tensor eq, int device_index);
at::Tensor cuda_compute_T_fixing_col(at::Tensor M, at::Tensor eq, int device_index);
at::Tensor cuda_tensor_to_field(at::Tensor input, int64_t padded_rows, int64_t padded_cols, int device_index);
// Defined in blake3_merkle.cu
at::Tensor cuda_blake3_merkle_leaves(at::Tensor W, int32_t chunk_size, int device_index);
at::Tensor cuda_blake3_activation_merkle_leaves(at::Tensor W, int32_t chunk_size, int device_index);
at::Tensor cuda_blake3_activation_row_roots(
    at::Tensor W,
    int32_t words_per_row,
    int device_index,
    int32_t words_per_leaf);
void cuda_blake3_runtime_row_roots_into(
    at::Tensor destination,
    at::Tensor source,
    int32_t lane_bytes,
    int device_index);
void cuda_blake3_paged_cache_block_roots_into(
    at::Tensor destination,
    at::Tensor cache,
    at::Tensor pages,
    int32_t cache_index,
    bool pair_first,
    int32_t lane_bytes,
    int device_index);
void cuda_blake3_runtime_row_roots_retain_into(
    at::Tensor destination,
    at::Tensor source,
    at::Tensor retained_values,
    at::Tensor retained_hashes,
    at::Tensor retention_slots,
    int32_t retention_stage,
    int32_t lane_bytes,
    int device_index);
void cuda_blake3_runtime_staged_row_roots_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    int32_t row_count,
    int device_index);
void cuda_blake3_runtime_staged_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    int32_t row_count,
    int device_index);
void cuda_blake3_runtime_staged_row_roots_lanes_history_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    at::Tensor destination_indices,
    int32_t row_count,
    int device_index);
void cuda_blake3_runtime_tensor_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources_first,
    std::vector<at::Tensor> sources_second,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    int32_t row_count,
    int device_index);
void cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    std::vector<at::Tensor> source_indices,
    at::Tensor destination_indices,
    int32_t row_count,
    int device_index);
void cuda_blake3_runtime_tensor_indexed_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    at::Tensor source_indices,
    int32_t row_count,
    int device_index);
void cuda_copy_runtime_tensor_indexed_rows_into(
    std::vector<at::Tensor> destinations,
    std::vector<at::Tensor> sources,
    at::Tensor source_indices,
    at::Tensor staged_indices,
    int32_t row_count,
    int device_index);
std::tuple<at::Tensor, at::Tensor>
cuda_blake3_prehashed_update_peaks(
    at::Tensor row_roots,
    int64_t base_count,
    at::Tensor initial_peaks,
    at::Tensor prior_deferred,
    int device_index);
void cuda_blake3_prehashed_update_peaks_inplace(
    at::Tensor row_roots,
    int64_t base_count,
    at::Tensor peaks,
    at::Tensor deferred,
    int device_index);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cuda_blake3_prehashed_update_peaks_ranges(
    at::Tensor row_roots,
    int64_t base_count,
    at::Tensor initial_peaks,
    at::Tensor prior_deferred,
    int64_t range_start,
    int64_t range_base_count,
    int64_t range_row_start,
    at::Tensor range_hashes,
    at::Tensor range_levels,
    at::Tensor range_indices,
    at::Tensor range_lengths,
    int device_index);
std::tuple<at::Tensor, at::Tensor> cuda_blake3_activation_update_peaks(
    at::Tensor W,
    int32_t words_per_row,
    int32_t rows_per_stage,
    int64_t base_count,
    at::Tensor initial_peaks,
    int device_index,
    int32_t words_per_leaf);
std::tuple<at::Tensor, at::Tensor>
cuda_blake3_activation_update_peaks_heterogeneous(
    at::Tensor W,
    at::Tensor stage_chunk_offsets,
    at::Tensor chunks_per_row,
    int32_t rows_per_stage,
    int64_t base_count,
    at::Tensor initial_peaks,
    at::Tensor prior_deferred,
    int device_index);
#endif

// ======================================================================
//  Mersenne-prime field arithmetic  (p = 2^61 - 1)
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

// ======================================================================
//  eq_vector - builds eq table for sumcheck
// ======================================================================

static std::vector<uint64_t> native_eq_vector(const std::vector<uint64_t>& r) {
    int m = (int)r.size();
    int n = 1 << m;
    std::vector<uint64_t> eq(n, 0);
    eq[0] = 1;
    int cur_size = 1;
    for (int i = 0; i < m; i++) {
        uint64_t ri = r[i];
        uint64_t one_minus_ri = h_sub(1, ri);
        // Expand from back to avoid overwriting (LSB-first ordering)
        for (int j = cur_size - 1; j >= 0; j--) {
            eq[j + cur_size] = h_mul(eq[j], ri);
            eq[j]            = h_mul(eq[j], one_minus_ri);
        }
        cur_size <<= 1;
    }
    return eq;
}

// Numpy-compatible version
static py::array_t<uint64_t> native_build_eq_table(py::array_t<uint64_t> point_np) {
    auto point = point_np.unchecked<1>();
    int m = (int)point.shape(0);
    int n = 1 << m;

    auto result = py::array_t<uint64_t>(n);
    auto eq = result.mutable_unchecked<1>();

    eq(0) = 1;
    int cur_size = 1;
    for (int i = 0; i < m; i++) {
        uint64_t ri = point(i);
        uint64_t one_minus_ri = h_sub(1, ri);
        for (int j = cur_size - 1; j >= 0; j--) {
            eq(j + cur_size) = h_mul(eq(j), ri);
            eq(j)            = h_mul(eq(j), one_minus_ri);
        }
        cur_size <<= 1;
    }
    return result;
}

// ======================================================================
//  Sumcheck round operations
// ======================================================================

static std::tuple<uint64_t, uint64_t, uint64_t>
native_compute_round_evals(
    py::array_t<uint64_t> T_A_np,
    py::array_t<uint64_t> T_B_np
) {
    auto T_A = T_A_np.unchecked<1>();
    auto T_B = T_B_np.unchecked<1>();
    int half = (int)T_A.shape(0) / 2;

    uint64_t e0 = 0, e1 = 0, e2 = 0;

    for (int b = 0; b < half; b++) {
        uint64_t a0 = T_A(2*b), a1 = T_A(2*b + 1);
        uint64_t b0 = T_B(2*b), b1 = T_B(2*b + 1);
        e0 = h_add(e0, h_mul(a0, b0));
        e1 = h_add(e1, h_mul(a1, b1));
        uint64_t a2 = h_sub(h_add(a1, a1), a0);
        uint64_t b2 = h_sub(h_add(b1, b1), b0);
        e2 = h_add(e2, h_mul(a2, b2));
    }
    return {e0, e1, e2};
}

// Vector version for compatibility
static std::tuple<uint64_t, uint64_t, uint64_t>
native_compute_round_evals_vec(
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

// ======================================================================
//  Table folding (update table)
// ======================================================================

static py::array_t<uint64_t> native_fold_table(
    py::array_t<uint64_t> table_np,
    uint64_t challenge
) {
    auto table = table_np.unchecked<1>();
    int half = (int)table.shape(0) / 2;
    uint64_t one_minus_r = h_sub(1, challenge);

    auto result = py::array_t<uint64_t>(half);
    auto out = result.mutable_unchecked<1>();

    for (int b = 0; b < half; b++) {
        out(b) = h_add(h_mul(table(2*b), one_minus_r),
                       h_mul(table(2*b + 1), challenge));
    }
    return result;
}

// Vector version
static std::vector<uint64_t> native_update_table(
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
//  MLE evaluation
// ======================================================================

static uint64_t native_mle_evaluate(
    py::array_t<uint64_t> evals_np,
    py::array_t<uint64_t> point_np
) {
    auto evals = evals_np.unchecked<1>();
    auto point = point_np.unchecked<1>();
    int m = (int)point.shape(0);
    int size = (int)evals.shape(0);

    // Copy to mutable buffer
    std::vector<uint64_t> buf(size);
    for (int i = 0; i < size; i++) buf[i] = evals(i);

    for (int i = 0; i < m; i++) {
        int half = size >> 1;
        uint64_t ri = point(i);
        uint64_t one_minus_ri = h_sub(1, ri);
        for (int b = 0; b < half; b++)
            buf[b] = h_add(h_mul(buf[2*b], one_minus_ri), h_mul(buf[2*b + 1], ri));
        size = half;
    }
    return buf[0];
}

// ======================================================================
//  T_A and T_B computation for GEMM sumcheck
// ======================================================================

static py::array_t<uint64_t> native_compute_T_A(
    py::array_t<uint64_t> X_np,  // [m, k_padded]
    py::array_t<uint64_t> r_i_np,
    int k_bits
) {
    auto X = X_np.unchecked<2>();
    auto r_i = r_i_np.unchecked<1>();
    int m = (int)X.shape(0);
    int k_padded = (int)X.shape(1);
    int k_size = 1 << k_bits;

    // Build eq table for r_i
    std::vector<uint64_t> r_vec(r_i.shape(0));
    for (size_t i = 0; i < r_vec.size(); i++) r_vec[i] = r_i(i);
    auto eq_table = native_eq_vector(r_vec);

    auto result = py::array_t<uint64_t>(k_size);
    auto T_A = result.mutable_unchecked<1>();

    int eq_len = std::min(m, (int)eq_table.size());

    for (int k = 0; k < k_size; k++) {
        uint64_t val = 0;
        if (k < k_padded) {
            for (int i = 0; i < eq_len; i++) {
                val = h_add(val, h_mul(X(i, k), eq_table[i]));
            }
        }
        T_A(k) = val;
    }
    return result;
}

static py::array_t<uint64_t> native_compute_T_B(
    py::array_t<uint64_t> W_np,  // [k_padded, n]
    py::array_t<uint64_t> r_j_np,
    int k_bits
) {
    auto W = W_np.unchecked<2>();
    auto r_j = r_j_np.unchecked<1>();
    int k_padded = (int)W.shape(0);
    int n = (int)W.shape(1);
    int k_size = 1 << k_bits;

    // Build eq table for r_j
    std::vector<uint64_t> r_vec(r_j.shape(0));
    for (size_t i = 0; i < r_vec.size(); i++) r_vec[i] = r_j(i);
    auto eq_table = native_eq_vector(r_vec);

    auto result = py::array_t<uint64_t>(k_size);
    auto T_B = result.mutable_unchecked<1>();

    int eq_len = std::min(n, (int)eq_table.size());

    for (int k = 0; k < k_size; k++) {
        uint64_t val = 0;
        if (k < k_padded) {
            for (int j = 0; j < eq_len; j++) {
                val = h_add(val, h_mul(W(k, j), eq_table[j]));
            }
        }
        T_B(k) = val;
    }
    return result;
}

// ======================================================================
//  Batch field operations
// ======================================================================

static py::array_t<uint64_t> native_mul_vec(
    py::array_t<uint64_t> a_np,
    py::array_t<uint64_t> b_np
) {
    auto a = a_np.unchecked<1>();
    auto b = b_np.unchecked<1>();
    int n = (int)a.shape(0);

    auto result = py::array_t<uint64_t>(n);
    auto out = result.mutable_unchecked<1>();

    for (int i = 0; i < n; i++) {
        out(i) = h_mul(a(i), b(i));
    }
    return result;
}

static py::array_t<uint64_t> native_add_vec(
    py::array_t<uint64_t> a_np,
    py::array_t<uint64_t> b_np
) {
    auto a = a_np.unchecked<1>();
    auto b = b_np.unchecked<1>();
    int n = (int)a.shape(0);

    auto result = py::array_t<uint64_t>(n);
    auto out = result.mutable_unchecked<1>();

    for (int i = 0; i < n; i++) {
        out(i) = h_add(a(i), b(i));
    }
    return result;
}

static uint64_t native_dot_product(
    py::array_t<uint64_t> a_np,
    py::array_t<uint64_t> b_np
) {
    auto a = a_np.unchecked<1>();
    auto b = b_np.unchecked<1>();
    int n = (int)a.shape(0);

    uint64_t result = 0;
    for (int i = 0; i < n; i++) {
        result = h_add(result, h_mul(a(i), b(i)));
    }
    return result;
}

// ======================================================================
//  Complete CPU block proof (native_prove_block)
//
//  Replaces the entire _prove_block_fast() Python method with one C++
//  call.  Handles: transcript absorb/squeeze, field conversion,
//  MLE evaluation, T_A/T_B, sumcheck loop, spot check sampling.
//  Pure CPU — no CUDA kernels — optimal for small MoE matrices.
// ======================================================================

// Convert signed int64 to Mersenne field element: ((v % P) + P) % P
static inline uint64_t mod_p_signed(int64_t v) {
    // Fast path for small positive values (common after int8 quantization)
    if (v >= 0 && static_cast<uint64_t>(v) < MP)
        return static_cast<uint64_t>(v);
    // General case: v mod P with sign handling
    int64_t r = v % static_cast<int64_t>(MP);
    if (r < 0) r += static_cast<int64_t>(MP);
    return static_cast<uint64_t>(r);
}

// Convert tensor to field elements with power-of-2 padding.
// Returns flat row-major vector of size padded_rows * padded_cols.
// Templated to accept int8/int32/int64 data directly.
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

// Legacy overload for int64_t* callers (X, Y which are already int64).
static std::vector<uint64_t> tensor_to_field_padded(
    const int64_t* data, int rows, int cols,
    int padded_rows, int padded_cols
) {
    return tensor_to_field_padded_impl(data, rows, cols, padded_rows, padded_cols);
}

// Dispatch tensor_to_field_padded based on tensor dtype.
// Accepts int8 (kChar), int32 (kInt), or int64 (kLong) tensors.
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

// Extract a single element from a tensor and convert to field element.
// Supports int8/int32/int64 dtypes without requiring a full int64 copy.
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

// MLE evaluate via iterative folding on a flat vector.
// Modifies buf in-place and returns the final scalar.
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
    auto eq = native_eq_vector(r_i);
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
    auto eq = native_eq_vector(r_j);
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

// Complete block proof: one C++ call replaces _prove_block_fast().
//
// All transcript, field conversion, MLE eval, sumcheck, and spot checks
// are done in C++.  Only W Merkle proof extraction stays in Python.
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
    at::Tensor X_rows,           // [m, K] int64 (CPU or GPU)
    at::Tensor W_cols,           // [K, n] int64 (CPU or GPU)
    at::Tensor Y_block,          // [m, n] int64 (CPU or GPU)
    py::bytes transcript_state_bytes,  // 32-byte forked transcript state
    int bi, int bj,
    py::bytes leaf_hash_bytes,   // 32-byte Y_merkle leaf hash
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

    // Move to CPU, keeping original dtype (avoids int8→int64 allocation for W).
    // X and Y are already int64 from matmul; W may be int8 from quantization.
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
        zkllm::NativeTranscript transcript(init_state);

        // ---- 2. Absorb block info ----
        transcript.absorb("bi", static_cast<uint64_t>(bi));
        transcript.absorb("bj", static_cast<uint64_t>(bj));
        transcript.absorb_bytes(
            reinterpret_cast<const uint8_t*>("leaf_hash"), 9,
            reinterpret_cast<const uint8_t*>(leaf_str.data()), 32);

        // ---- 3. Field conversion with padding ----
        // Uses dtype dispatch — avoids int8→int64 copy for W.
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

        // Evaluate: fold with r_i then r_j (matches Python Y_mle.evaluate(r_i + r_j))
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
        // Keep W_field for spot W value extraction below

        // ---- 7. Sumcheck loop ----
        rounds.reserve(k_bits);
        challenges.reserve(k_bits);

        for (int round_idx = 0; round_idx < k_bits; round_idx++) {
            auto [e0, e1, e2] = native_compute_round_evals_vec(T_A, T_B);
            rounds.emplace_back(e0, e1, e2);

            transcript.absorb("round", static_cast<uint64_t>(round_idx));
            transcript.absorb("e0", e0);
            transcript.absorb("e1", e1);
            transcript.absorb("e2", e2);

            uint64_t r = transcript.squeeze("challenge");
            challenges.push_back(r);

            T_A = native_update_table(T_A, r);
            T_B = native_update_table(T_B, r);
        }

        final_A_val = T_A.empty() ? 0 : T_A[0];
        final_B_val = T_B.empty() ? 0 : T_B[0];

        // ---- 8. Spot X ----
        spot_X.reserve(num_spots);
        for (int i = 0; i < num_spots; i++) {
            // Build labels: "spot_X:row:" + str(i), "spot_X:col:" + str(i)
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

// ======================================================================
//  Fused GPU sumcheck block prover
// ======================================================================

#ifdef WITH_CUDA

// Forward declarations for CUDA functions (defined in sumcheck_kernels.cu)
// (Already declared at the top of this file)

// Lagrange interpolation for degree-2 polynomial at point r
// g(0)=e0, g(1)=e1, g(2)=e2 → g(r)
// Used to recompute claimed_sum verification (not needed for prover,
// but included here for completeness — verifier uses this).
static uint64_t lagrange_interp_deg2(uint64_t e0, uint64_t e1, uint64_t e2, uint64_t r) {
    // g(r) = e0 * (1-r)(2-r)/2 + e1 * r(2-r)/(-1) + e2 * r(r-1)/2
    // Simplification: g(r) = e0 + r * (-3*e0 + 4*e1 - e2) / 2 + r^2 * (e0 - 2*e1 + e2) / 2
    // In Mersenne field, division by 2 = multiplication by inverse of 2.
    // inv(2) mod P = (P+1)/2
    static const uint64_t inv2 = (MP + 1) / 2;

    // a = e2 - 2*e1 + e0 (coefficient of r^2)
    uint64_t two_e1 = h_add(e1, e1);
    uint64_t a = h_add(h_sub(e2, two_e1), e0);

    // b = 4*e1 - 3*e0 - e2 (coefficient of r, before /2)
    uint64_t four_e1 = h_add(two_e1, two_e1);
    uint64_t three_e0 = h_add(h_add(e0, e0), e0);
    uint64_t b = h_sub(h_sub(four_e1, three_e0), e2);

    // g(r) = e0 + r * (b + r * a) / 2
    uint64_t inner = h_add(h_mul(r, a), b);
    return h_add(e0, h_mul(r, h_mul(inner, inv2)));
}

// Fused GPU sumcheck block prover.
//
// Performs the entire block proof on GPU with C++ Fiat-Shamir transcript:
//   1. Convert X, W, Y to Mersenne field elements (cuda_tensor_to_field)
//   2. Squeeze r_i, r_j from transcript (C++ Keccak256)
//   3. Evaluate Y MLE at (r_i, r_j) via iterative GPU folding
//   4. Compute T_A, T_B tables (cuda_build_eq_table + cuda_compute_T_fixing_*)
//   5. Run k_bits rounds of sumcheck (cuda_compute_round_evals + cuda_fold_table)
//   6. Return proof data + updated transcript state
//
// Returns: (rounds, claimed_sum, final_A, final_B, challenges, transcript_state)
static std::tuple<
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>,  // rounds
    uint64_t,                                                // claimed_sum
    uint64_t,                                                // final_A
    uint64_t,                                                // final_B
    std::vector<uint64_t>,                                   // challenges
    py::bytes                                                // new transcript state
> cuda_prove_sumcheck_block(
    at::Tensor X_int,                // [m, K] int64 on GPU
    at::Tensor W_int,                // [K, n] int64 on GPU
    at::Tensor Y_int,                // [m, n] int64 on GPU
    py::bytes transcript_state_bytes,// 32-byte current transcript state
    int m_bits, int n_bits, int k_bits,
    int device_index
) {
    // --- Parse transcript state (GIL held) ---
    std::string state_str(transcript_state_bytes);
    TORCH_CHECK(state_str.size() == 32,
        "Transcript state must be 32 bytes, got ", state_str.size());

    std::array<uint8_t, 32> init_state;
    std::memcpy(init_state.data(), state_str.data(), 32);

    // --- Pre-validate inputs ---
    TORCH_CHECK(X_int.dim() == 2 && W_int.dim() == 2 && Y_int.dim() == 2,
        "X, W, Y must be 2D tensors");
    TORCH_CHECK(X_int.dtype() == at::kLong && W_int.dtype() == at::kLong
        && Y_int.dtype() == at::kLong, "X, W, Y must be int64");
    TORCH_CHECK(X_int.is_cuda() && W_int.is_cuda() && Y_int.is_cuda(),
        "X, W, Y must be on CUDA");

    // --- Compute (release GIL) ---
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rounds;
    uint64_t claimed_sum = 0, final_A = 0, final_B = 0;
    std::vector<uint64_t> challenges;
    std::array<uint8_t, 32> final_state;

    {
        py::gil_scoped_release release;

        c10::cuda::CUDAGuard guard(device_index);

        int m_padded = 1 << m_bits;
        int n_padded = 1 << n_bits;
        int k_padded = 1 << k_bits;

        // Initialize C++ transcript
        zkllm::NativeTranscript transcript(init_state);

        // ---- Step 1: Convert X, W, Y to Mersenne field on GPU ----
        auto X_field = cuda_tensor_to_field(
            X_int.contiguous(), m_padded, k_padded, device_index);
        auto W_field = cuda_tensor_to_field(
            W_int.contiguous(), k_padded, n_padded, device_index);
        auto Y_field = cuda_tensor_to_field(
            Y_int.contiguous(), m_padded, n_padded, device_index);

        // ---- Step 2: Squeeze r_i, r_j from transcript ----
        auto r_i = transcript.squeeze_n("r_i", m_bits);
        auto r_j = transcript.squeeze_n("r_j", n_bits);

        // ---- Step 3: Evaluate Y MLE at (r_i || r_j) via GPU folding ----
        // MLE uses LSB-first bit ordering with column-major layout:
        //   evals[row + col * padded_rows] = matrix[row, col]
        // PyTorch reshape(-1) is row-major (C-order), so we transpose
        // first to get Fortran-order flattening (matches numpy ravel('F')).
        auto Y_flat = Y_field.t().contiguous().reshape({-1});

        // Fold with each element of the combined point [r_i || r_j]
        // (matches Python's Y_mle.evaluate(r_i + r_j))
        for (auto v : r_i) {
            Y_flat = cuda_fold_table(Y_flat, v, device_index);
        }
        for (auto v : r_j) {
            Y_flat = cuda_fold_table(Y_flat, v, device_index);
        }

        // Read claimed_sum from GPU
        claimed_sum = static_cast<uint64_t>(Y_flat.cpu().item<int64_t>());

        // Free Y tensors
        Y_field = at::Tensor();
        Y_flat = at::Tensor();

        // ---- Step 4: Compute T_A and T_B ----

        // T_A[k] = sum_i X_field[i,k] * eq(r_i, i)
        auto r_i_tensor = at::empty(
            {static_cast<int64_t>(r_i.size())},
            at::TensorOptions().dtype(at::kLong));
        std::memcpy(r_i_tensor.data_ptr<int64_t>(), r_i.data(),
                     r_i.size() * sizeof(uint64_t));
        r_i_tensor = r_i_tensor.cuda();

        auto eq_i = cuda_build_eq_table(r_i_tensor, device_index);
        auto T_A = cuda_compute_T_fixing_row(X_field, eq_i, device_index);

        // T_B[k] = sum_j W_field[k,j] * eq(r_j, j)
        auto r_j_tensor = at::empty(
            {static_cast<int64_t>(r_j.size())},
            at::TensorOptions().dtype(at::kLong));
        std::memcpy(r_j_tensor.data_ptr<int64_t>(), r_j.data(),
                     r_j.size() * sizeof(uint64_t));
        r_j_tensor = r_j_tensor.cuda();

        auto eq_j = cuda_build_eq_table(r_j_tensor, device_index);
        auto T_B = cuda_compute_T_fixing_col(W_field, eq_j, device_index);

        // Free intermediate tensors
        X_field = at::Tensor();
        W_field = at::Tensor();
        eq_i = at::Tensor();
        eq_j = at::Tensor();
        r_i_tensor = at::Tensor();
        r_j_tensor = at::Tensor();

        // ---- Step 5: Sumcheck loop (k_bits rounds) ----
        rounds.reserve(k_bits);
        challenges.reserve(k_bits);

        for (int round_idx = 0; round_idx < k_bits; round_idx++) {
            // Compute (e0, e1, e2) on GPU (includes cudaDeviceSynchronize)
            auto [e0, e1, e2] = cuda_compute_round_evals(
                T_A, T_B, device_index);

            rounds.emplace_back(e0, e1, e2);

            // Absorb into transcript (C++ Keccak, no GIL needed)
            transcript.absorb("round", static_cast<uint64_t>(round_idx));
            transcript.absorb("e0", e0);
            transcript.absorb("e1", e1);
            transcript.absorb("e2", e2);

            // Squeeze challenge
            uint64_t r = transcript.squeeze("challenge");
            challenges.push_back(r);

            // Fold tables for next round (GPU async)
            T_A = cuda_fold_table(T_A, r, device_index);
            T_B = cuda_fold_table(T_B, r, device_index);
        }

        // ---- Step 6: Extract final values ----
        final_A = (T_A.size(0) > 0)
            ? static_cast<uint64_t>(T_A.cpu().item<int64_t>()) : 0;
        final_B = (T_B.size(0) > 0)
            ? static_cast<uint64_t>(T_B.cpu().item<int64_t>()) : 0;

        final_state = transcript.get_state();
    }
    // GIL re-acquired here

    py::bytes state_out(
        reinterpret_cast<const char*>(final_state.data()), 32);

    return {rounds, claimed_sum, final_A, final_B, challenges, state_out};
}

#endif  // WITH_CUDA

// ======================================================================
//  PyBind11 module
// ======================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "VeriLLM native extension for fast field arithmetic and sumcheck";

    // Field arithmetic
    m.def("h_add", &h_add, "Field addition");
    m.def("h_sub", &h_sub, "Field subtraction");
    m.def("h_mul", &h_mul, "Field multiplication");

    // Eq table
    m.def("native_eq_vector", &native_eq_vector, py::arg("r"),
          "Build eq table from vector");
    m.def("native_build_eq_table", &native_build_eq_table, py::arg("point"),
          "Build eq table from numpy array");

    // Sumcheck operations
    m.def("native_compute_round_evals", &native_compute_round_evals,
          py::arg("T_A"), py::arg("T_B"),
          "Compute sumcheck round evaluations (e0, e1, e2)");
    m.def("native_compute_round_evals_vec", &native_compute_round_evals_vec,
          py::arg("T_A"), py::arg("T_B"),
          "Compute sumcheck round evaluations from vectors");

    // Table folding
    m.def("native_fold_table", &native_fold_table,
          py::arg("table"), py::arg("challenge"),
          "Fold sumcheck table with challenge");
    m.def("native_update_table", &native_update_table,
          py::arg("table"), py::arg("challenge"),
          "Update sumcheck table (vector version)");

    // MLE evaluation
    m.def("native_mle_evaluate", &native_mle_evaluate,
          py::arg("evals"), py::arg("point"),
          "Evaluate MLE at point");

    // T_A/T_B computation
    m.def("native_compute_T_A", &native_compute_T_A,
          py::arg("X"), py::arg("r_i"), py::arg("k_bits"),
          "Compute T_A table for GEMM sumcheck");
    m.def("native_compute_T_B", &native_compute_T_B,
          py::arg("W"), py::arg("r_j"), py::arg("k_bits"),
          "Compute T_B table for GEMM sumcheck");

    // Batch operations
    m.def("native_mul_vec", &native_mul_vec,
          py::arg("a"), py::arg("b"),
          "Vectorized field multiplication");
    m.def("native_add_vec", &native_add_vec,
          py::arg("a"), py::arg("b"),
          "Vectorized field addition");
    m.def("native_dot_product", &native_dot_product,
          py::arg("a"), py::arg("b"),
          "Field dot product");

    // Complete block proof (CPU C++)
    m.def("native_prove_block", &native_prove_block,
          py::arg("X_rows"), py::arg("W_cols"), py::arg("Y_block"),
          py::arg("transcript_state"),
          py::arg("bi"), py::arg("bj"), py::arg("leaf_hash"),
          py::arg("num_spots"),
          py::arg("row_offset"), py::arg("col_offset"),
          py::arg("W_num_cols_full"), py::arg("W_chunk_size"),
          "Complete block proof in C++: field conv + MLE + sumcheck + spots");

#ifdef WITH_CUDA
    // CUDA-accelerated operations
    m.attr("HAS_CUDA") = true;

    m.def("cuda_fold_table", &cuda_fold_table,
          py::arg("table"), py::arg("r"), py::arg("device_index") = 0,
          "GPU-accelerated table folding");
    m.def("cuda_compute_round_evals", &cuda_compute_round_evals,
          py::arg("T_A"), py::arg("T_B"), py::arg("device_index") = 0,
          "GPU-accelerated sumcheck round evaluation");
    m.def("cuda_build_eq_table", &cuda_build_eq_table,
          py::arg("point"), py::arg("device_index") = 0,
          "GPU-accelerated eq table construction");
    m.def("cuda_field_mul_batch", &cuda_field_mul_batch,
          py::arg("a"), py::arg("b"), py::arg("device_index") = 0,
          "GPU-accelerated batch field multiplication");
    m.def("cuda_int64_matmul", &cuda_int64_matmul,
          py::arg("X"), py::arg("W"), py::arg("device_index") = 0,
          "GPU-accelerated exact int64 matrix multiplication");
    m.def("cuda_int8_matmul", &cuda_int8_matmul,
          py::arg("X"), py::arg("W"), py::arg("device_index") = 0,
          "GPU-accelerated INT8 matrix multiplication with int32 output");
    m.def("cuda_field_sum_axis0", &cuda_field_sum_axis0,
          py::arg("input"), py::arg("device_index") = 0,
          "GPU-accelerated 2D field sum along axis 0 (row reduction)");
    m.def("cuda_field_sum_axis1", &cuda_field_sum_axis1,
          py::arg("input"), py::arg("device_index") = 0,
          "GPU-accelerated 2D field sum along axis 1 (column reduction)");
    m.def("cuda_compute_T_fixing_row", &cuda_compute_T_fixing_row,
          py::arg("M"), py::arg("eq"), py::arg("device_index") = 0,
          "Fused T table: T[j] = sum_i M[i,j] * eq[i] on GPU");
    m.def("cuda_compute_T_fixing_col", &cuda_compute_T_fixing_col,
          py::arg("M"), py::arg("eq"), py::arg("device_index") = 0,
          "Fused T table: T[i] = sum_j M[i,j] * eq[j] on GPU");
    m.def("cuda_tensor_to_field", &cuda_tensor_to_field,
          py::arg("input"), py::arg("padded_rows"), py::arg("padded_cols"),
          py::arg("device_index") = 0,
          "GPU-accelerated tensor to Mersenne field conversion with padding");
    m.def("cuda_blake3_merkle_leaves", &cuda_blake3_merkle_leaves,
          py::arg("W"), py::arg("chunk_size") = 128, py::arg("device_index") = 0,
          "GPU-accelerated BLAKE3 Merkle leaf hashing for int8 weight tensors");
    m.def("cuda_blake3_activation_merkle_leaves", &cuda_blake3_activation_merkle_leaves,
          py::arg("W"), py::arg("chunk_size") = 256, py::arg("device_index") = 0,
          "GPU-accelerated BLAKE3 activation Merkle leaf hashing for int64 tensors");
    m.def("cuda_blake3_activation_row_roots", &cuda_blake3_activation_row_roots,
          py::arg("W"), py::arg("words_per_row"),
          py::arg("device_index") = 0, py::arg("words_per_leaf") = 256,
          "GPU-accelerated BLAKE3 roots for padded fixed-width activation rows");
    m.def("cuda_blake3_runtime_row_roots_into",
          &cuda_blake3_runtime_row_roots_into,
          py::arg("destination"), py::arg("source"),
          py::arg("lane_bytes") = 2048, py::arg("device_index") = 0,
          "GPU BLAKE3 roots from native runtime rows into a device buffer");
    m.def("cuda_blake3_paged_cache_block_roots_into",
          &cuda_blake3_paged_cache_block_roots_into,
          py::arg("destination"), py::arg("cache"), py::arg("pages"),
          py::arg("cache_index"), py::arg("pair_first"),
          py::arg("lane_bytes") = 2048, py::arg("device_index") = 0,
          "GPU BLAKE3 logical block roots directly from paged K/V cache");
    m.def("cuda_blake3_runtime_row_roots_retain_into",
          &cuda_blake3_runtime_row_roots_retain_into,
          py::arg("destination"), py::arg("source"),
          py::arg("retained_values"), py::arg("retained_hashes"),
          py::arg("retention_slots"), py::arg("retention_stage"),
          py::arg("lane_bytes") = 2048, py::arg("device_index") = 0,
          "GPU BLAKE3 roots plus bounded post-nonce retained lanes");
    m.def("cuda_blake3_runtime_staged_row_roots_into",
          &cuda_blake3_runtime_staged_row_roots_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("staging"), py::arg("row_widths"),
          py::arg("row_count"), py::arg("device_index") = 0,
          "Batched GPU BLAKE3 roots from bounded staged runtime rows");
    m.def("cuda_blake3_runtime_staged_row_roots_lanes_into",
          &cuda_blake3_runtime_staged_row_roots_lanes_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("staging"), py::arg("row_widths"),
          py::arg("lane_widths"), py::arg("row_count"),
          py::arg("device_index") = 0,
          "Batched GPU BLAKE3 roots from mixed-lane staged runtime rows");
    m.def("cuda_blake3_runtime_staged_row_roots_lanes_history_into",
          &cuda_blake3_runtime_staged_row_roots_lanes_history_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("staging"), py::arg("row_widths"),
          py::arg("lane_widths"), py::arg("destination_indices"),
          py::arg("row_count"), py::arg("device_index") = 0,
          "Batched GPU BLAKE3 roots into persistent runtime history");
    m.def("cuda_blake3_runtime_tensor_row_roots_lanes_into",
          &cuda_blake3_runtime_tensor_row_roots_lanes_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("sources_first"), py::arg("sources_second"),
          py::arg("row_widths"), py::arg("lane_widths"),
          py::arg("row_count"), py::arg("device_index") = 0,
          "Batched GPU BLAKE3 roots directly from runtime tensors");
    m.def(
          "cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into",
          &cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("sources"), py::arg("row_widths"),
          py::arg("lane_widths"), py::arg("source_indices"),
          py::arg("destination_indices"), py::arg("row_count"),
          py::arg("device_index") = 0,
          "GPU BLAKE3 roots from indexed runtime tensors into history");
    m.def("cuda_blake3_runtime_tensor_indexed_row_roots_lanes_into",
          &cuda_blake3_runtime_tensor_indexed_row_roots_lanes_into,
          py::arg("destination"), py::arg("scratch"),
          py::arg("sources"), py::arg("row_widths"),
          py::arg("lane_widths"), py::arg("source_indices"),
          py::arg("row_count"), py::arg("device_index") = 0,
          "GPU BLAKE3 roots from active indexed runtime tensor rows");
    m.def("cuda_copy_runtime_tensor_indexed_rows_into",
          &cuda_copy_runtime_tensor_indexed_rows_into,
          py::arg("destinations"), py::arg("sources"),
          py::arg("source_indices"), py::arg("staged_indices"),
          py::arg("row_count"), py::arg("device_index") = 0,
          "Copy active indexed runtime tensor rows into compact staging");
    m.def("cuda_blake3_prehashed_update_peaks",
          &cuda_blake3_prehashed_update_peaks,
          py::arg("row_roots"), py::arg("base_count"),
          py::arg("initial_peaks"), py::arg("prior_deferred"),
          py::arg("device_index") = 0,
          "GPU outer-frontier update from prehashed runtime row roots");
    m.def("cuda_blake3_prehashed_update_peaks_inplace",
          &cuda_blake3_prehashed_update_peaks_inplace,
          py::arg("row_roots"), py::arg("base_count"),
          py::arg("peaks"), py::arg("deferred"),
          py::arg("device_index") = 0,
          "In-place GPU frontier update from strided runtime row roots");
    m.def("cuda_blake3_prehashed_update_peaks_ranges",
          &cuda_blake3_prehashed_update_peaks_ranges,
          py::arg("row_roots"), py::arg("base_count"),
          py::arg("initial_peaks"), py::arg("prior_deferred"),
          py::arg("range_start"), py::arg("range_base_count"),
          py::arg("range_row_start"), py::arg("range_hashes"),
          py::arg("range_levels"), py::arg("range_indices"),
          py::arg("range_lengths"), py::arg("device_index") = 0,
          "GPU combined prefix and dyadic range update from row roots");
    m.def("cuda_blake3_activation_update_peaks",
          &cuda_blake3_activation_update_peaks,
          py::arg("W"), py::arg("words_per_row"),
          py::arg("rows_per_stage"), py::arg("base_count"),
          py::arg("initial_peaks"), py::arg("device_index") = 0,
          py::arg("words_per_leaf") = 256,
          "GPU-accelerated streaming Merkle frontier update for activation rows");
    m.def("cuda_blake3_activation_update_peaks_heterogeneous",
          &cuda_blake3_activation_update_peaks_heterogeneous,
          py::arg("W"), py::arg("stage_chunk_offsets"),
          py::arg("chunks_per_row"), py::arg("rows_per_stage"),
          py::arg("base_count"), py::arg("initial_peaks"),
          py::arg("prior_deferred"),
          py::arg("device_index") = 0,
          "GPU-accelerated heterogeneous activation frontier update");
    m.def("cuda_prove_sumcheck_block", &cuda_prove_sumcheck_block,
          py::arg("X_int"), py::arg("W_int"), py::arg("Y_int"),
          py::arg("transcript_state"),
          py::arg("m_bits"), py::arg("n_bits"), py::arg("k_bits"),
          py::arg("device_index") = 0,
          "Fused GPU sumcheck block prover: field conversion + MLE eval + "
          "T_A/T_B + sumcheck loop in one call");
#else
    m.attr("HAS_CUDA") = false;
#endif
}
