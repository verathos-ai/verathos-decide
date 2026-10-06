/*
 * sumcheck_kernels.cu — VeriLLM CUDA kernels for fast sumcheck
 *
 * Components:
 *   1. Mersenne-prime field arithmetic (p = 2^61 - 1)
 *   2. Parallel table folding
 *   3. Parallel sumcheck round evaluation (reduction)
 *   4. Parallel eq table construction
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

/* ================================================================== */
/*  Constants                                                          */
/* ================================================================== */

#define MERSENNE_P      ((uint64_t)((1ULL << 61) - 1))
#define BLOCK_THREADS   256
#define WARP_SIZE       32

/* ================================================================== */
/*  Error checking macro                                               */
/* ================================================================== */

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    TORCH_CHECK(_e == cudaSuccess, #expr " failed: ", cudaGetErrorString(_e)); \
} while(0)

/* ================================================================== */
/*  Device: Mersenne-prime field arithmetic                            */
/* ================================================================== */

__device__ __forceinline__ uint64_t field_add(uint64_t a, uint64_t b) {
    uint64_t r = a + b;
    if (r >= MERSENNE_P) r -= MERSENNE_P;
    return r;
}

__device__ __forceinline__ uint64_t field_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (MERSENNE_P - b + a);
}

__device__ __forceinline__ uint64_t field_mul(uint64_t a, uint64_t b) {
    // Use __umul64hi for high 64 bits of 128-bit product
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);

    // Mersenne reduction: r = (lo mod 2^61) + (lo >> 61) + (hi * 8)
    // Note: hi * 8 = hi << 3, and since hi < 2^61, hi*8 < 2^64
    uint64_t lo_lo = lo & MERSENNE_P;
    uint64_t lo_hi = lo >> 61;
    uint64_t sum = lo_lo + lo_hi + (hi << 3);

    // Final reduction
    uint64_t r = (sum >> 61) + (sum & MERSENNE_P);
    if (r >= MERSENNE_P) r -= MERSENNE_P;
    return r;
}

/* ================================================================== */
/*  Kernel: Parallel table folding                                     */
/*  new[i] = table[2i] * (1-r) + table[2i+1] * r                       */
/* ================================================================== */

__global__ void fold_table_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ table,
    uint64_t r,
    uint64_t one_minus_r,
    int64_t new_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= new_size) return;

    uint64_t t0 = table[2 * idx];
    uint64_t t1 = table[2 * idx + 1];

    // out[i] = t0 * (1-r) + t1 * r
    uint64_t term0 = field_mul(t0, one_minus_r);
    uint64_t term1 = field_mul(t1, r);
    out[idx] = field_add(term0, term1);
}

/* ================================================================== */
/*  Kernel: Build eq table (parallel doubling)                         */
/*  For variable j: new[i] = old[i] * (1-xj), new[i + old_size] = old[i] * xj */
/* ================================================================== */

__global__ void eq_table_double_kernel(
    uint64_t* __restrict__ table,
    uint64_t xj,
    uint64_t one_minus_xj,
    int64_t old_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= old_size) return;

    uint64_t v = table[idx];
    uint64_t v0 = field_mul(v, one_minus_xj);  // bit j = 0
    uint64_t v1 = field_mul(v, xj);            // bit j = 1

    // Write both values
    table[idx] = v0;
    table[idx + old_size] = v1;
}

/* ================================================================== */
/*  Kernel: Compute sumcheck round evals (parallel reduction)          */
/*  Computes e0 = sum T_A[2i]*T_B[2i], e1 = sum T_A[2i+1]*T_B[2i+1]   */
/*  e2 = sum (2*T_A[2i+1]-T_A[2i]) * (2*T_B[2i+1]-T_B[2i])            */
/* ================================================================== */

// Warp-level reduction
__device__ __forceinline__ uint64_t warp_reduce_add(uint64_t val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        uint64_t other = __shfl_down_sync(0xffffffff, val, offset);
        val = field_add(val, other);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ uint64_t block_reduce_add(uint64_t val) {
    __shared__ uint64_t shared[WARP_SIZE];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp reduction
    val = warp_reduce_add(val);

    // Write warp results to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces all warp results
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_add(val);

    return val;
}

// Kernel for computing e0, e1, e2 partial sums per block
__global__ void compute_round_evals_kernel(
    uint64_t* __restrict__ e0_out,
    uint64_t* __restrict__ e1_out,
    uint64_t* __restrict__ e2_out,
    const uint64_t* __restrict__ T_A,
    const uint64_t* __restrict__ T_B,
    int64_t half_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t local_e0 = 0, local_e1 = 0, local_e2 = 0;

    if (idx < half_size) {
        uint64_t a0 = T_A[2 * idx];
        uint64_t a1 = T_A[2 * idx + 1];
        uint64_t b0 = T_B[2 * idx];
        uint64_t b1 = T_B[2 * idx + 1];

        // e0 contribution: a0 * b0
        local_e0 = field_mul(a0, b0);

        // e1 contribution: a1 * b1
        local_e1 = field_mul(a1, b1);

        // e2 contribution: (2*a1 - a0) * (2*b1 - b0)
        uint64_t a2 = field_sub(field_add(a1, a1), a0);
        uint64_t b2 = field_sub(field_add(b1, b1), b0);
        local_e2 = field_mul(a2, b2);
    }

    // Block-level reduction
    uint64_t block_e0 = block_reduce_add(local_e0);
    uint64_t block_e1 = block_reduce_add(local_e1);
    uint64_t block_e2 = block_reduce_add(local_e2);

    // Thread 0 writes block result
    if (threadIdx.x == 0) {
        e0_out[blockIdx.x] = block_e0;
        e1_out[blockIdx.x] = block_e1;
        e2_out[blockIdx.x] = block_e2;
    }
}

// Final reduction kernel for block results
__global__ void final_reduce_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ partial,
    int64_t num_blocks
) {
    __shared__ uint64_t shared[BLOCK_THREADS];

    int tid = threadIdx.x;
    uint64_t sum = 0;

    // Each thread accumulates multiple partial results
    for (int64_t i = tid; i < num_blocks; i += blockDim.x) {
        sum = field_add(sum, partial[i]);
    }

    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = field_add(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = shared[0];
    }
}

/* ================================================================== */
/*  Kernel: Batch field multiplication                                 */
/* ================================================================== */

__global__ void field_mul_batch_kernel(
    uint64_t* __restrict__ out,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = field_mul(a[idx], b[idx]);
}

/* ================================================================== */
/*  Kernel: Dot product (reduction)                                    */
/* ================================================================== */

__global__ void dot_product_kernel(
    uint64_t* __restrict__ result,
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    int64_t n
) {
    __shared__ uint64_t shared[BLOCK_THREADS];

    int tid = threadIdx.x;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Compute local product and reduce
    uint64_t sum = 0;
    while (idx < n) {
        sum = field_add(sum, field_mul(a[idx], b[idx]));
        idx += gridDim.x * blockDim.x;
    }

    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = field_add(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd((unsigned long long*)result, shared[0]);
    }
}

/* ================================================================== */
/*  Host-callable wrapper functions                                    */
/* ================================================================== */

// Fold table on GPU
at::Tensor cuda_fold_table(at::Tensor table, uint64_t r, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    int64_t old_size = table.size(0);
    int64_t new_size = old_size / 2;

    TORCH_CHECK(table.dtype() == at::kLong, "Table must be int64");
    TORCH_CHECK(table.is_cuda(), "Table must be on CUDA");

    auto result = at::zeros({new_size}, table.options());

    if (new_size == 0) return result;

    uint64_t one_minus_r = (r > 0) ? (MERSENNE_P - r + 1) : 1;
    if (one_minus_r >= MERSENNE_P) one_minus_r -= MERSENNE_P;

    int num_blocks = (new_size + BLOCK_THREADS - 1) / BLOCK_THREADS;

    fold_table_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)result.data_ptr<int64_t>(),
        (const uint64_t*)table.data_ptr<int64_t>(),
        r,
        one_minus_r,
        new_size
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}

// Compute round evals on GPU
std::tuple<uint64_t, uint64_t, uint64_t> cuda_compute_round_evals(
    at::Tensor T_A, at::Tensor T_B, int device_index
) {
    c10::cuda::CUDAGuard guard(device_index);

    int64_t size = T_A.size(0);
    int64_t half_size = size / 2;

    TORCH_CHECK(T_A.dtype() == at::kLong && T_B.dtype() == at::kLong, "Must be int64");
    TORCH_CHECK(T_A.is_cuda() && T_B.is_cuda(), "Must be on CUDA");
    TORCH_CHECK(T_A.size(0) == T_B.size(0), "Size mismatch");

    if (half_size == 0) {
        return std::make_tuple(0ULL, 0ULL, 0ULL);
    }

    int num_blocks = (half_size + BLOCK_THREADS - 1) / BLOCK_THREADS;

    // Allocate partial result arrays
    auto e0_partial = at::zeros({num_blocks}, T_A.options());
    auto e1_partial = at::zeros({num_blocks}, T_A.options());
    auto e2_partial = at::zeros({num_blocks}, T_A.options());

    // Phase 1: Compute block-level partial sums
    compute_round_evals_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)e0_partial.data_ptr<int64_t>(),
        (uint64_t*)e1_partial.data_ptr<int64_t>(),
        (uint64_t*)e2_partial.data_ptr<int64_t>(),
        (const uint64_t*)T_A.data_ptr<int64_t>(),
        (const uint64_t*)T_B.data_ptr<int64_t>(),
        half_size
    );

    // Phase 2: Final reduction
    auto e0_result = at::zeros({1}, T_A.options());
    auto e1_result = at::zeros({1}, T_A.options());
    auto e2_result = at::zeros({1}, T_A.options());

    final_reduce_kernel<<<1, BLOCK_THREADS>>>(
        (uint64_t*)e0_result.data_ptr<int64_t>(),
        (const uint64_t*)e0_partial.data_ptr<int64_t>(),
        num_blocks
    );
    final_reduce_kernel<<<1, BLOCK_THREADS>>>(
        (uint64_t*)e1_result.data_ptr<int64_t>(),
        (const uint64_t*)e1_partial.data_ptr<int64_t>(),
        num_blocks
    );
    final_reduce_kernel<<<1, BLOCK_THREADS>>>(
        (uint64_t*)e2_result.data_ptr<int64_t>(),
        (const uint64_t*)e2_partial.data_ptr<int64_t>(),
        num_blocks
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    uint64_t e0 = e0_result.cpu().item<int64_t>();
    uint64_t e1 = e1_result.cpu().item<int64_t>();
    uint64_t e2 = e2_result.cpu().item<int64_t>();

    return std::make_tuple(e0, e1, e2);
}

// Build eq table on GPU
at::Tensor cuda_build_eq_table(at::Tensor point, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    int n = point.size(0);
    int table_size = 1 << n;

    TORCH_CHECK(point.dtype() == at::kLong, "Point must be int64");

    // Move point to CPU for iteration
    auto point_cpu = point.cpu();
    auto point_ptr = point_cpu.data_ptr<int64_t>();

    // Allocate table on GPU
    auto table = at::zeros({table_size}, point.options().device(at::kCUDA, device_index));

    // Initialize first element to 1
    table[0] = 1;

    int cur_size = 1;
    for (int j = 0; j < n; j++) {
        uint64_t xj = (uint64_t)point_ptr[j];
        uint64_t one_minus_xj = (xj > 0) ? (MERSENNE_P - xj + 1) : 1;
        if (one_minus_xj >= MERSENNE_P) one_minus_xj -= MERSENNE_P;

        int num_blocks = (cur_size + BLOCK_THREADS - 1) / BLOCK_THREADS;

        eq_table_double_kernel<<<num_blocks, BLOCK_THREADS>>>(
            (uint64_t*)table.data_ptr<int64_t>(),
            xj,
            one_minus_xj,
            cur_size
        );

        cur_size <<= 1;
    }

    CUDA_CHECK(cudaGetLastError());
    return table;
}

/* ================================================================== */
/*  Kernel: Exact int64 matrix multiplication (tiled GEMM)             */
/*  Y[M,N] = X[M,K] @ W[K,N] using native int64 arithmetic             */
/*  No float conversion needed - exact results guaranteed              */
/* ================================================================== */

#define TILE_SIZE 16

__global__ void int64_matmul_kernel(
    int64_t* __restrict__ Y,       // [M, N] output
    const int64_t* __restrict__ X, // [M, K] input
    const int64_t* __restrict__ W, // [K, N] weights
    int64_t M, int64_t K, int64_t N
) {
    // Shared memory tiles
    __shared__ int64_t Xs[TILE_SIZE][TILE_SIZE];
    __shared__ int64_t Ws[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int64_t acc = 0;

    // Loop over tiles along K dimension
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_SIZE;

        // Load X tile (coalesced along K)
        int x_k = k_base + threadIdx.x;
        if (row < M && x_k < K) {
            Xs[threadIdx.y][threadIdx.x] = X[row * K + x_k];
        } else {
            Xs[threadIdx.y][threadIdx.x] = 0;
        }

        // Load W tile (coalesced along N)
        int w_k = k_base + threadIdx.y;
        if (w_k < K && col < N) {
            Ws[threadIdx.y][threadIdx.x] = W[w_k * N + col];
        } else {
            Ws[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial products for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Xs[threadIdx.y][k] * Ws[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        Y[row * N + col] = acc;
    }
}

// Host wrapper for int64 matmul
at::Tensor cuda_int64_matmul(at::Tensor X, at::Tensor W, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = W.size(1);

    TORCH_CHECK(X.dtype() == at::kLong && W.dtype() == at::kLong, "Must be int64");
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "Must be on CUDA");
    TORCH_CHECK(X.size(1) == W.size(0), "K dimension mismatch");

    auto Y = at::zeros({M, N}, X.options());

    if (M == 0 || K == 0 || N == 0) return Y;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    int64_matmul_kernel<<<grid, block>>>(
        Y.data_ptr<int64_t>(),
        X.data_ptr<int64_t>(),
        W.data_ptr<int64_t>(),
        M, K, N
    );

    CUDA_CHECK(cudaGetLastError());
    return Y;
}

/* ================================================================== */
/*  Kernel: INT8 native matrix multiplication with int32 accumulator  */
/*  Y[M,N] = X[M,K] @ W[K,N] using int8 inputs, int32 output          */
/*  Optimized for bitsandbytes INT8 weights (already on GPU!)         */
/*  int32 accumulator: max K=4096, max sum = 4096*127*127 = 66M < 2B  */
/* ================================================================== */

#define TILE_SIZE_INT8 16

__global__ void int8_matmul_kernel(
    int32_t* __restrict__ Y,       // [M, N] output
    const int8_t* __restrict__ X,  // [M, K] input
    const int8_t* __restrict__ W,  // [K, N] weights
    int64_t M, int64_t K, int64_t N
) {
    // Shared memory tiles for int8 (4x smaller than int64 tiles!)
    __shared__ int8_t Xs[TILE_SIZE_INT8][TILE_SIZE_INT8];
    __shared__ int8_t Ws[TILE_SIZE_INT8][TILE_SIZE_INT8];

    int row = blockIdx.y * TILE_SIZE_INT8 + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE_INT8 + threadIdx.x;

    int32_t acc = 0;  // int32 sufficient for K up to 131072

    // Loop over tiles along K dimension
    int num_tiles = (K + TILE_SIZE_INT8 - 1) / TILE_SIZE_INT8;
    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_SIZE_INT8;

        // Load X tile (coalesced along K)
        int x_k = k_base + threadIdx.x;
        if (row < M && x_k < K) {
            Xs[threadIdx.y][threadIdx.x] = X[row * K + x_k];
        } else {
            Xs[threadIdx.y][threadIdx.x] = 0;
        }

        // Load W tile (coalesced along N)
        int w_k = k_base + threadIdx.y;
        if (w_k < K && col < N) {
            Ws[threadIdx.y][threadIdx.x] = W[w_k * N + col];
        } else {
            Ws[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Accumulate with int32 (int8 * int8 -> int32)
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_INT8; k++) {
            acc += (int32_t)Xs[threadIdx.y][k] * (int32_t)Ws[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        Y[row * N + col] = acc;
    }
}

// Host wrapper for int8 matmul
at::Tensor cuda_int8_matmul(at::Tensor X, at::Tensor W, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    int64_t M = X.size(0);
    int64_t K = X.size(1);
    int64_t N = W.size(1);

    TORCH_CHECK(X.dtype() == at::kChar && W.dtype() == at::kChar,
                "Inputs must be int8 (torch.int8)");
    TORCH_CHECK(X.is_cuda() && W.is_cuda(), "Must be on CUDA");
    TORCH_CHECK(X.size(1) == W.size(0), "K dimension mismatch");

    // Output is int32 (sufficient for K up to 131072)
    auto Y = at::zeros({M, N}, X.options().dtype(at::kInt));

    if (M == 0 || K == 0 || N == 0) return Y;

    dim3 block(TILE_SIZE_INT8, TILE_SIZE_INT8);
    dim3 grid((N + TILE_SIZE_INT8 - 1) / TILE_SIZE_INT8,
              (M + TILE_SIZE_INT8 - 1) / TILE_SIZE_INT8);

    int8_matmul_kernel<<<grid, block, 0, stream.stream()>>>(
        Y.data_ptr<int32_t>(),
        X.data_ptr<int8_t>(),
        W.data_ptr<int8_t>(),
        M, K, N
    );

    CUDA_CHECK(cudaGetLastError());
    return Y;
}

// Batch field multiplication on GPU
at::Tensor cuda_field_mul_batch(at::Tensor a, at::Tensor b, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    int64_t n = a.size(0);
    TORCH_CHECK(a.size(0) == b.size(0), "Size mismatch");
    TORCH_CHECK(a.dtype() == at::kLong && b.dtype() == at::kLong, "Must be int64");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Must be on CUDA");

    auto result = at::zeros_like(a);

    if (n == 0) return result;

    int num_blocks = (n + BLOCK_THREADS - 1) / BLOCK_THREADS;

    field_mul_batch_kernel<<<num_blocks, BLOCK_THREADS, 0, stream.stream()>>>(
        (uint64_t*)result.data_ptr<int64_t>(),
        (const uint64_t*)a.data_ptr<int64_t>(),
        (const uint64_t*)b.data_ptr<int64_t>(),
        n
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}

/* ================================================================== */
/*  Kernel: Fused T table computation (build eq + mul + reduce)        */
/*  T[j] = sum_i M[i,j] * eq(r, i) mod P                               */
/*  Computes entire T table in one kernel without CPU roundtrips       */
/* ================================================================== */

__global__ void compute_T_fixing_row_kernel(
    uint64_t* __restrict__ T,          // [num_cols] output
    const uint64_t* __restrict__ M,    // [num_rows, num_cols] matrix
    const uint64_t* __restrict__ eq,   // [num_rows] precomputed eq table
    int64_t num_rows,
    int64_t num_cols
) {
    int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_cols) return;

    uint64_t acc = 0;

    // T[j] = sum_i M[i,j] * eq[i]
    for (int64_t row = 0; row < num_rows; row++) {
        uint64_t m_val = M[row * num_cols + col];
        uint64_t eq_val = eq[row];
        acc = field_add(acc, field_mul(m_val, eq_val));
    }

    T[col] = acc;
}

__global__ void compute_T_fixing_col_kernel(
    uint64_t* __restrict__ T,          // [num_rows] output
    const uint64_t* __restrict__ M,    // [num_rows, num_cols] matrix
    const uint64_t* __restrict__ eq,   // [num_cols] precomputed eq table
    int64_t num_rows,
    int64_t num_cols
) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint64_t acc = 0;

    // T[i] = sum_j M[i,j] * eq[j]
    for (int64_t col = 0; col < num_cols; col++) {
        uint64_t m_val = M[row * num_cols + col];
        uint64_t eq_val = eq[col];
        acc = field_add(acc, field_mul(m_val, eq_val));
    }

    T[row] = acc;
}

// Host wrapper: Compute T[j] = sum_i M[i,j] * eq[i] entirely on GPU
at::Tensor cuda_compute_T_fixing_row(at::Tensor M, at::Tensor eq, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    TORCH_CHECK(M.dim() == 2, "M must be 2D");
    TORCH_CHECK(eq.dim() == 1, "eq must be 1D");
    TORCH_CHECK(M.dtype() == at::kLong && eq.dtype() == at::kLong, "Must be int64");
    TORCH_CHECK(M.is_cuda() && eq.is_cuda(), "Must be on CUDA");

    int64_t num_rows = M.size(0);
    int64_t num_cols = M.size(1);
    int64_t eq_len = eq.size(0);

    // Use min(num_rows, eq_len) for actual computation
    int64_t effective_rows = (num_rows < eq_len) ? num_rows : eq_len;

    auto T = at::zeros({num_cols}, M.options());

    if (num_cols == 0 || effective_rows == 0) return T;

    int num_blocks = (num_cols + BLOCK_THREADS - 1) / BLOCK_THREADS;

    compute_T_fixing_row_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)T.data_ptr<int64_t>(),
        (const uint64_t*)M.data_ptr<int64_t>(),
        (const uint64_t*)eq.data_ptr<int64_t>(),
        effective_rows,
        num_cols
    );

    CUDA_CHECK(cudaGetLastError());
    return T;
}

// Host wrapper: Compute T[i] = sum_j M[i,j] * eq[j] entirely on GPU
at::Tensor cuda_compute_T_fixing_col(at::Tensor M, at::Tensor eq, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    TORCH_CHECK(M.dim() == 2, "M must be 2D");
    TORCH_CHECK(eq.dim() == 1, "eq must be 1D");
    TORCH_CHECK(M.dtype() == at::kLong && eq.dtype() == at::kLong, "Must be int64");
    TORCH_CHECK(M.is_cuda() && eq.is_cuda(), "Must be on CUDA");

    int64_t num_rows = M.size(0);
    int64_t num_cols = M.size(1);
    int64_t eq_len = eq.size(0);

    // Use min(num_cols, eq_len) for actual computation
    int64_t effective_cols = (num_cols < eq_len) ? num_cols : eq_len;

    auto T = at::zeros({num_rows}, M.options());

    if (num_rows == 0 || effective_cols == 0) return T;

    int num_blocks = (num_rows + BLOCK_THREADS - 1) / BLOCK_THREADS;

    compute_T_fixing_col_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)T.data_ptr<int64_t>(),
        (const uint64_t*)M.data_ptr<int64_t>(),
        (const uint64_t*)eq.data_ptr<int64_t>(),
        num_rows,
        effective_cols
    );

    CUDA_CHECK(cudaGetLastError());
    return T;
}

/* ================================================================== */
/*  Kernel: 2D field sum along axis 0                                  */
/*  T[j] = sum_i input[i,j] mod P for all j                            */
/*  Each thread handles one column, sums across all rows               */
/* ================================================================== */

__global__ void field_sum_axis0_kernel(
    uint64_t* __restrict__ out,        // [num_cols] output
    const uint64_t* __restrict__ in,   // [num_rows, num_cols] input (row-major)
    int64_t num_rows,
    int64_t num_cols
) {
    int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_cols) return;

    uint64_t acc = 0;

    // Each thread sums one column across all rows
    for (int64_t row = 0; row < num_rows; row++) {
        acc = field_add(acc, in[row * num_cols + col]);
    }

    out[col] = acc;
}

/* ================================================================== */
/*  Kernel: 2D field sum along axis 1                                  */
/*  T[i] = sum_j input[i,j] mod P for all i                            */
/*  Each thread handles one row, sums across all columns               */
/* ================================================================== */

__global__ void field_sum_axis1_kernel(
    uint64_t* __restrict__ out,        // [num_rows] output
    const uint64_t* __restrict__ in,   // [num_rows, num_cols] input (row-major)
    int64_t num_rows,
    int64_t num_cols
) {
    int64_t row = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint64_t acc = 0;

    // Each thread sums one row across all columns
    for (int64_t col = 0; col < num_cols; col++) {
        acc = field_add(acc, in[row * num_cols + col]);
    }

    out[row] = acc;
}

// Host wrapper: Sum 2D tensor along axis 0 in Mersenne field
at::Tensor cuda_field_sum_axis0(at::Tensor input, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.dtype() == at::kLong, "Input must be int64");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");

    int64_t num_rows = input.size(0);
    int64_t num_cols = input.size(1);

    auto result = at::zeros({num_cols}, input.options());

    if (num_cols == 0 || num_rows == 0) return result;

    int num_blocks = (num_cols + BLOCK_THREADS - 1) / BLOCK_THREADS;

    field_sum_axis0_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)result.data_ptr<int64_t>(),
        (const uint64_t*)input.data_ptr<int64_t>(),
        num_rows,
        num_cols
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}

/* ================================================================== */
/*  Kernel: Tensor → field element conversion with padding              */
/*  out[row,col] = in[row,col] < 0 ? in[row,col]+P : in[row,col]       */
/*  Padded region filled with zeros.                                    */
/* ================================================================== */

__global__ void tensor_to_field_padded_kernel(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ in,
    int64_t P,
    int64_t src_rows, int64_t src_cols,
    int64_t dst_cols, int64_t total
) {
    // 64-bit indexing: a padded LM-head block (e.g. 8192 x 262144 = 2^31
    // elements for a 154880-column vocab) overflows 32-bit index math.
    // With int params, (int)total wrapped negative and every thread
    // returned at the bounds check, silently leaving the output zeroed.
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t row = idx / dst_cols;
    int64_t col = idx % dst_cols;

    if (row < src_rows && col < src_cols) {
        int64_t val = in[row * src_cols + col];
        out[idx] = val < 0 ? val + P : val;
    } else {
        out[idx] = 0;
    }
}

// Host wrapper: Convert tensor to Mersenne field elements on GPU with padding
at::Tensor cuda_tensor_to_field(at::Tensor input, int64_t padded_rows, int64_t padded_cols, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.dtype() == at::kLong, "Input must be int64");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");

    int64_t src_rows = input.size(0);
    int64_t src_cols = input.size(1);
    int64_t total = padded_rows * padded_cols;

    auto result = at::zeros({padded_rows, padded_cols}, input.options());

    if (total == 0) return result;

    int64_t num_blocks = (total + BLOCK_THREADS - 1) / BLOCK_THREADS;
    TORCH_CHECK(num_blocks <= INT32_MAX,
        "tensor_to_field grid too large: ", num_blocks, " blocks");

    tensor_to_field_padded_kernel<<<(unsigned int)num_blocks, BLOCK_THREADS>>>(
        result.data_ptr<int64_t>(),
        input.data_ptr<int64_t>(),
        (int64_t)MERSENNE_P,
        src_rows, src_cols,
        padded_cols, total
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}

// Host wrapper: Sum 2D tensor along axis 1 in Mersenne field
at::Tensor cuda_field_sum_axis1(at::Tensor input, int device_index) {
    c10::cuda::CUDAGuard guard(device_index);

    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.dtype() == at::kLong, "Input must be int64");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");

    int64_t num_rows = input.size(0);
    int64_t num_cols = input.size(1);

    auto result = at::zeros({num_rows}, input.options());

    if (num_rows == 0 || num_cols == 0) return result;

    int num_blocks = (num_rows + BLOCK_THREADS - 1) / BLOCK_THREADS;

    field_sum_axis1_kernel<<<num_blocks, BLOCK_THREADS>>>(
        (uint64_t*)result.data_ptr<int64_t>(),
        (const uint64_t*)input.data_ptr<int64_t>(),
        num_rows,
        num_cols
    );

    CUDA_CHECK(cudaGetLastError());
    return result;
}
