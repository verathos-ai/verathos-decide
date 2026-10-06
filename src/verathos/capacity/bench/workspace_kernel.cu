// Capacity workspace (int8 GEMM with A/B transition mixing) and FP64 identity
// workloads, with SHA-256 Merkle commitments over square blocks.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <tuple>

#define CUDA_OK(expr) do { \
    cudaError_t _e = (expr); \
    TORCH_CHECK(_e == cudaSuccess, #expr " failed: ", cudaGetErrorString(_e)); \
} while (0)

#define CUBLAS_OK(expr) do { \
    cublasStatus_t _s = (expr); \
    TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS, #expr " failed: ", (int)_s); \
} while (0)

static constexpr int kReduceBlockCap = 2048;

static inline cublasHandle_t get_handle(int device_index) {
    static std::array<cublasHandle_t, 32> handles{};
    static std::array<bool, 32> initialized{};
    TORCH_CHECK(device_index >= 0 && device_index < (int)handles.size(), "unsupported device index");
    if (!initialized[device_index]) {
        c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
        CUBLAS_OK(cublasCreate(&handles[device_index]));
        CUBLAS_OK(cublasSetMathMode(handles[device_index], CUBLAS_TENSOR_OP_MATH));
        initialized[device_index] = true;
    }
    return handles[device_index];
}


__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__global__ void fill_i8_pair_kernel(
    int8_t* __restrict__ a,
    int8_t* __restrict__ b,
    int64_t total,
    uint64_t seed,
    int64_t pass_index,
    int round_index
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint64_t x = seed;
    x ^= (uint64_t)pass_index * 0xD1B54A32D192ED03ULL;
    x ^= (uint64_t)round_index * 0xABC98388FB8FAC03ULL;
    x ^= (uint64_t)idx;
    uint64_t ra = splitmix64(x ^ 0x8CB92BA72F3D8DD7ULL);
    uint64_t rb = splitmix64(x ^ 0x195D7C4F8E2D2D13ULL);
    a[idx] = (int8_t)((int)(ra & 0xFFU) - 128);
    b[idx] = (int8_t)((int)(rb & 0xFFU) - 128);
}

__global__ void mix_i8_pair_transition_kernel(
    const int8_t* __restrict__ src_a,
    const int8_t* __restrict__ src_b,
    int8_t* __restrict__ dst_a,
    int8_t* __restrict__ dst_b,
    int64_t total,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int mix_index,
    int transition_fanout
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint64_t basis = seed;
    basis ^= (uint64_t)pass_index * 0xD1B54A32D192ED03ULL;
    basis ^= (uint64_t)round_index * 0xABC98388FB8FAC03ULL;
    basis ^= (uint64_t)mix_index * 0x9E3779B97F4A7C15ULL;

    uint8_t out_a = (uint8_t)src_a[idx];
    uint8_t out_b = (uint8_t)src_b[idx];
    int fanout = transition_fanout > 1 ? transition_fanout : 1;
    for (int lane = 0; lane < fanout; ++lane) {
        uint64_t lane_basis = basis ^ ((uint64_t)(lane + 1) * 0xD6E8FEB86659FD93ULL);
        uint64_t pa = splitmix64(lane_basis ^ ((uint64_t)idx * 0xBF58476D1CE4E5B9ULL));
        uint64_t pb = splitmix64(lane_basis ^ ((uint64_t)idx * 0x94D049BB133111EBULL));
        int64_t j = (int64_t)(pa % (uint64_t)total);
        int64_t k = (int64_t)(pb % (uint64_t)total);
        uint8_t aj = (uint8_t)src_a[j];
        uint8_t bk = (uint8_t)src_b[k];
        uint8_t ma = (uint8_t)splitmix64(lane_basis ^ ((uint64_t)idx << 1) ^ 0xA5A5A5A5A5A5A5A5ULL);
        uint8_t mb = (uint8_t)splitmix64(lane_basis ^ ((uint64_t)idx << 2) ^ 0x5A5A5A5A5A5A5A5AULL);
        out_a = (uint8_t)(out_a + (uint8_t)(bk ^ ma));
        out_b = (uint8_t)(out_b + (uint8_t)(aj ^ mb));
    }
    dst_a[idx] = (int8_t)out_a;
    dst_b[idx] = (int8_t)out_b;
}

__global__ void copy_i8_pair_kernel(
    const int8_t* __restrict__ src_a,
    const int8_t* __restrict__ src_b,
    int8_t* __restrict__ dst_a,
    int8_t* __restrict__ dst_b,
    int64_t total
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst_a[idx] = src_a[idx];
    dst_b[idx] = src_b[idx];
}

__global__ void reduce_q_kernel(
    const int32_t* __restrict__ q,
    int64_t total,
    uint64_t* __restrict__ partial,
    uint64_t seed,
    int64_t pass_index,
    int round_index
) {
    extern __shared__ uint64_t smem[];
    uint64_t acc = seed ^ ((uint64_t)pass_index << 32) ^ (uint64_t)round_index;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    while (idx < total) {
        uint64_t v = (uint64_t)(uint32_t)q[idx];
        acc ^= splitmix64(v + (uint64_t)idx * 0x9E3779B97F4A7C15ULL);
        idx += stride;
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            smem[threadIdx.x] ^= splitmix64(smem[threadIdx.x + offset] + (uint64_t)offset);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = smem[0];
}

__global__ void reduce_partial_kernel(
    const uint64_t* __restrict__ partial,
    int64_t partial_count,
    int64_t* __restrict__ root,
    uint64_t seed,
    int64_t pass_index,
    int round_index
) {
    uint64_t acc0 = seed ^ 0x243F6A8885A308D3ULL;
    uint64_t acc1 = seed ^ 0x13198A2E03707344ULL;
    uint64_t acc2 = seed ^ 0xA4093822299F31D0ULL;
    uint64_t acc3 = seed ^ 0x082EFA98EC4E6C89ULL;
    for (int64_t i = threadIdx.x; i < partial_count; i += blockDim.x) {
        uint64_t v = partial[i];
        acc0 ^= splitmix64(v + (uint64_t)i);
        acc1 += splitmix64(v ^ ((uint64_t)i << 1));
        acc2 ^= splitmix64(v + (uint64_t)pass_index);
        acc3 += splitmix64(v ^ (uint64_t)round_index);
    }
    extern __shared__ uint64_t smem[];
    uint64_t* s0 = smem;
    uint64_t* s1 = smem + blockDim.x;
    uint64_t* s2 = smem + 2 * blockDim.x;
    uint64_t* s3 = smem + 3 * blockDim.x;
    s0[threadIdx.x] = acc0;
    s1[threadIdx.x] = acc1;
    s2[threadIdx.x] = acc2;
    s3[threadIdx.x] = acc3;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            s0[threadIdx.x] ^= splitmix64(s0[threadIdx.x + offset]);
            s1[threadIdx.x] += splitmix64(s1[threadIdx.x + offset]);
            s2[threadIdx.x] ^= splitmix64(s2[threadIdx.x + offset]);
            s3[threadIdx.x] += splitmix64(s3[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        root[0] = (int64_t)s0[0];
        root[1] = (int64_t)s1[0];
        root[2] = (int64_t)s2[0];
        root[3] = (int64_t)s3[0];
    }
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sha_sig0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t sha_sig1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t sha_sigma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha_sigma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

__constant__ uint32_t K256[64] = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,
    0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,
    0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,
    0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,
    0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,
    0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,
    0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,
    0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,
    0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
};

// Domain tags, zero-padded to 32 bytes. Q tags hash their string length;
// state and FP64 tags hash all 32 bytes.
__constant__ uint8_t D_Q_BLOCK_TAG_GPU[32];
__constant__ uint8_t D_Q_LEAF_TAG_GPU[32];
__constant__ uint8_t D_Q_NODE_TAG_GPU[32];
__constant__ uint8_t D_STATE_PAIR_BLOCK_TAG_GPU[32];
__constant__ uint8_t D_STATE_PAIR_LEAF_TAG_GPU[32];
__constant__ uint8_t D_STATE_PAIR_NODE_TAG_GPU[32];
__constant__ uint8_t D_FP64_C_BLOCK_TAG_GPU[32];
__constant__ uint8_t D_FP64_C_LEAF_TAG_GPU[32];
__constant__ uint8_t D_FP64_C_NODE_TAG_GPU[32];

struct Sha256Ctx {
    uint32_t h[8];
    uint64_t total_len;
    uint8_t buf[64];
    uint32_t buf_len;
};

__device__ __forceinline__ void sha256_init(Sha256Ctx& ctx) {
    ctx.h[0] = 0x6a09e667U; ctx.h[1] = 0xbb67ae85U;
    ctx.h[2] = 0x3c6ef372U; ctx.h[3] = 0xa54ff53aU;
    ctx.h[4] = 0x510e527fU; ctx.h[5] = 0x9b05688cU;
    ctx.h[6] = 0x1f83d9abU; ctx.h[7] = 0x5be0cd19U;
    ctx.total_len = 0;
    ctx.buf_len = 0;
}

__device__ void sha256_compress(Sha256Ctx& ctx, const uint8_t block[64]) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t(block[4 * i]) << 24) |
               (uint32_t(block[4 * i + 1]) << 16) |
               (uint32_t(block[4 * i + 2]) << 8) |
               (uint32_t(block[4 * i + 3]));
    }
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        w[i] = sha_sigma1(w[i - 2]) + w[i - 7] + sha_sigma0(w[i - 15]) + w[i - 16];
    }

    uint32_t a = ctx.h[0], b = ctx.h[1], c = ctx.h[2], d = ctx.h[3];
    uint32_t e = ctx.h[4], f = ctx.h[5], g = ctx.h[6], hh = ctx.h[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t t1 = hh + sha_sig1(e) + sha_ch(e, f, g) + K256[i] + w[i];
        uint32_t t2 = sha_sig0(a) + sha_maj(a, b, c);
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    ctx.h[0] += a; ctx.h[1] += b; ctx.h[2] += c; ctx.h[3] += d;
    ctx.h[4] += e; ctx.h[5] += f; ctx.h[6] += g; ctx.h[7] += hh;
}

__device__ void sha256_update(Sha256Ctx& ctx, const uint8_t* data, uint64_t len) {
    ctx.total_len += len;
    if (ctx.buf_len) {
        uint32_t need = 64 - ctx.buf_len;
        uint32_t take = (len < need) ? (uint32_t)len : need;
        for (uint32_t i = 0; i < take; ++i) {
            ctx.buf[ctx.buf_len + i] = data[i];
        }
        ctx.buf_len += take;
        data += take;
        len -= take;
        if (ctx.buf_len == 64) {
            sha256_compress(ctx, ctx.buf);
            ctx.buf_len = 0;
        }
    }
    while (len >= 64) {
        sha256_compress(ctx, data);
        data += 64;
        len -= 64;
    }
    if (len) {
        for (uint32_t i = 0; i < len; ++i) {
            ctx.buf[i] = data[i];
        }
        ctx.buf_len = (uint32_t)len;
    }
}

__device__ void sha256_final(Sha256Ctx& ctx, uint8_t out[32]) {
    uint64_t bit_len = ctx.total_len * 8ULL;
    ctx.buf[ctx.buf_len++] = 0x80;
    if (ctx.buf_len > 56) {
        while (ctx.buf_len < 64) {
            ctx.buf[ctx.buf_len++] = 0;
        }
        sha256_compress(ctx, ctx.buf);
        ctx.buf_len = 0;
    }
    while (ctx.buf_len < 56) {
        ctx.buf[ctx.buf_len++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        ctx.buf[ctx.buf_len++] = (uint8_t)((bit_len >> (i * 8)) & 0xFFU);
    }
    sha256_compress(ctx, ctx.buf);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[4 * i] = (uint8_t)((ctx.h[i] >> 24) & 0xFFU);
        out[4 * i + 1] = (uint8_t)((ctx.h[i] >> 16) & 0xFFU);
        out[4 * i + 2] = (uint8_t)((ctx.h[i] >> 8) & 0xFFU);
        out[4 * i + 3] = (uint8_t)(ctx.h[i] & 0xFFU);
    }
}

__global__ void q_i32_block_hash_kernel(
    const int32_t* __restrict__ q,
    int n,
    int block_size,
    uint8_t* __restrict__ block_hashes
) {
    int blocks_per_row = n / block_size;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = blocks_per_row * blocks_per_row;
    if (block_idx >= total_blocks) return;

    int bi = block_idx / blocks_per_row;
    int bj = block_idx % blocks_per_row;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_Q_BLOCK_TAG_GPU, 19);
    for (int r = 0; r < block_size; ++r) {
        int row = bi * block_size + r;
        const uint8_t* row_ptr = reinterpret_cast<const uint8_t*>(
            q + (size_t)row * n + (size_t)bj * block_size);
        sha256_update(ctx, row_ptr, (uint64_t)block_size * sizeof(int32_t));
    }
    sha256_final(ctx, block_hashes + (size_t)block_idx * 32);
}

__global__ void state_i8_pair_block_hash_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int n,
    int block_size,
    uint8_t* __restrict__ block_hashes
) {
    int blocks_per_row = n / block_size;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = blocks_per_row * blocks_per_row;
    if (block_idx >= total_blocks) return;

    int bi = block_idx / blocks_per_row;
    int bj = block_idx % blocks_per_row;
    Sha256Ctx ctx;
    sha256_init(ctx);
    // Use the full fixed-width tag to avoid device-side string-length work.
    sha256_update(ctx, D_STATE_PAIR_BLOCK_TAG_GPU, 32);
    for (int r = 0; r < block_size; ++r) {
        int row = bi * block_size + r;
        const uint8_t* a_row = reinterpret_cast<const uint8_t*>(
            a + (size_t)row * n + (size_t)bj * block_size);
        const uint8_t* b_row = reinterpret_cast<const uint8_t*>(
            b + (size_t)row * n + (size_t)bj * block_size);
        sha256_update(ctx, a_row, (uint64_t)block_size);
        sha256_update(ctx, b_row, (uint64_t)block_size);
    }
    sha256_final(ctx, block_hashes + (size_t)block_idx * 32);
}

__global__ void fill_fp64_pair_kernel(
    double* __restrict__ a,
    double* __restrict__ b,
    int64_t total,
    uint64_t seed,
    int64_t pass_index,
    int round_index
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint64_t x = seed;
    x ^= (uint64_t)pass_index * 0xD1B54A32D192ED03ULL;
    x ^= (uint64_t)round_index * 0xABC98388FB8FAC03ULL;
    x ^= (uint64_t)idx;
    uint64_t ra = splitmix64(x ^ 0xDDBCA9D6B4A7A1D9ULL);
    uint64_t rb = splitmix64(x ^ 0xA1F0E27C5C314DD1ULL);
    a[idx] = (ra & 1ULL) ? 1.0 : -1.0;
    b[idx] = (rb & 1ULL) ? 1.0 : -1.0;
}

__global__ void fp64_c_block_hash_kernel(
    const double* __restrict__ c,
    int n,
    int block_size,
    uint8_t* __restrict__ block_hashes
) {
    int blocks_per_row = n / block_size;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = blocks_per_row * blocks_per_row;
    if (block_idx >= total_blocks) return;

    int bi = block_idx / blocks_per_row;
    int bj = block_idx % blocks_per_row;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_FP64_C_BLOCK_TAG_GPU, 32);
    for (int r = 0; r < block_size; ++r) {
        int row = bi * block_size + r;
        const uint8_t* row_ptr = reinterpret_cast<const uint8_t*>(
            c + (size_t)row * n + (size_t)bj * block_size);
        sha256_update(ctx, row_ptr, (uint64_t)block_size * sizeof(double));
    }
    sha256_final(ctx, block_hashes + (size_t)block_idx * 32);
}

__global__ void q_merkle_leaf_kernel(
    const uint8_t* __restrict__ block_hashes,
    int num_blocks,
    uint8_t* __restrict__ leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_Q_LEAF_TAG_GPU, 18);
    sha256_update(ctx, block_hashes + (size_t)idx * 32, 32);
    sha256_final(ctx, leaves + (size_t)idx * 32);
}

__global__ void state_pair_merkle_leaf_kernel(
    const uint8_t* __restrict__ block_hashes,
    int num_blocks,
    uint8_t* __restrict__ leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_STATE_PAIR_LEAF_TAG_GPU, 32);
    sha256_update(ctx, block_hashes + (size_t)idx * 32, 32);
    sha256_final(ctx, leaves + (size_t)idx * 32);
}

__global__ void fp64_c_merkle_leaf_kernel(
    const uint8_t* __restrict__ block_hashes,
    int num_blocks,
    uint8_t* __restrict__ leaves
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_FP64_C_LEAF_TAG_GPU, 32);
    sha256_update(ctx, block_hashes + (size_t)idx * 32, 32);
    sha256_final(ctx, leaves + (size_t)idx * 32);
}

__global__ void q_merkle_level_kernel(
    const uint8_t* __restrict__ in_hashes,
    int in_count,
    uint8_t* __restrict__ out_hashes
) {
    int out_count = (in_count + 1) >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_count) return;
    int left_idx = idx << 1;
    int right_idx = left_idx + 1;
    if (right_idx >= in_count) right_idx = left_idx;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_Q_NODE_TAG_GPU, 18);
    sha256_update(ctx, in_hashes + (size_t)left_idx * 32, 32);
    sha256_update(ctx, in_hashes + (size_t)right_idx * 32, 32);
    sha256_final(ctx, out_hashes + (size_t)idx * 32);
}

__global__ void state_pair_merkle_level_kernel(
    const uint8_t* __restrict__ in_hashes,
    int in_count,
    uint8_t* __restrict__ out_hashes
) {
    int out_count = (in_count + 1) >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_count) return;
    int left_idx = idx << 1;
    int right_idx = left_idx + 1;
    if (right_idx >= in_count) right_idx = left_idx;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_STATE_PAIR_NODE_TAG_GPU, 32);
    sha256_update(ctx, in_hashes + (size_t)left_idx * 32, 32);
    sha256_update(ctx, in_hashes + (size_t)right_idx * 32, 32);
    sha256_final(ctx, out_hashes + (size_t)idx * 32);
}

__global__ void fp64_c_merkle_level_kernel(
    const uint8_t* __restrict__ in_hashes,
    int in_count,
    uint8_t* __restrict__ out_hashes
) {
    int out_count = (in_count + 1) >> 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_count) return;
    int left_idx = idx << 1;
    int right_idx = left_idx + 1;
    if (right_idx >= in_count) right_idx = left_idx;
    Sha256Ctx ctx;
    sha256_init(ctx);
    sha256_update(ctx, D_FP64_C_NODE_TAG_GPU, 32);
    sha256_update(ctx, in_hashes + (size_t)left_idx * 32, 32);
    sha256_update(ctx, in_hashes + (size_t)right_idx * 32, 32);
    sha256_final(ctx, out_hashes + (size_t)idx * 32);
}

static inline size_t merkle_total_nodes(size_t leaves) {
    size_t total = 0;
    for (;;) {
        total += leaves;
        if (leaves <= 1) break;
        leaves = (leaves + 1) >> 1;
    }
    return total;
}

static void run_round_gemm_only(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    cublasHandle_t handle,
    cudaStream_t stream,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    cublasGemmAlgo_t gemm_algo
) {
    int64_t n = a.size(0);
    int64_t total = n * n;
    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    const int32_t alpha = 1;
    const int32_t beta = 0;

    fill_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total, seed, pass_index, round_index);
    CUDA_OK(cudaGetLastError());

    // Row-major trick: cuBLAS column-major B*A stores row-major A*B in q.
    CUBLAS_OK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)n, (int)n,
        &alpha,
        b.data_ptr<int8_t>(), CUDA_R_8I, (int)n,
        a.data_ptr<int8_t>(), CUDA_R_8I, (int)n,
        &beta,
        q.data_ptr<int32_t>(), CUDA_R_32I, (int)n,
        CUBLAS_COMPUTE_32I,
        gemm_algo
    ));
}

static void run_fp64_identity_round_only(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cublasHandle_t handle,
    cudaStream_t stream,
    uint64_t seed,
    int64_t pass_index,
    int round_index
) {
    int64_t n = a.size(0);
    int64_t total = n * n;
    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    const double alpha = 1.0;
    const double beta = 0.0;

    fill_fp64_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
        a.data_ptr<double>(), b.data_ptr<double>(), total, seed, pass_index, round_index);
    CUDA_OK(cudaGetLastError());

    // Row-major trick: cuBLAS column-major B*A stores row-major A*B in c.
    CUBLAS_OK(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)n, (int)n,
        &alpha,
        b.data_ptr<double>(), (int)n,
        a.data_ptr<double>(), (int)n,
        &beta,
        c.data_ptr<double>(), (int)n
    ));
}

double run_workspace_round(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int device_index
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && q.is_cuda() && partial.is_cuda() && root.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "a/b must be int8");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(partial.dtype() == torch::kInt64, "partial must be int64");
    TORCH_CHECK(root.dtype() == torch::kInt64, "root must be int64");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && q.dim() == 2, "a/b/q must be matrices");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && q.is_contiguous(), "a/b/q must be contiguous");
    TORCH_CHECK(partial.is_contiguous() && root.is_contiguous(), "partial/root must be contiguous");
    int64_t n = a.size(0);
    TORCH_CHECK(a.size(1) == n && b.size(0) == n && b.size(1) == n, "a/b must be NxN");
    TORCH_CHECK(q.size(0) == n && q.size(1) == n, "q must be NxN");
    TORCH_CHECK(root.numel() >= 4, "root must have at least 4 int64 words");
    TORCH_CHECK(partial.numel() > 0, "partial tensor must not be empty");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    int64_t total = n * n;
    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    int blocks_reduce = std::min<int64_t>(partial.numel(), kReduceBlockCap);
    const int32_t alpha = 1;
    const int32_t beta = 0;

    fill_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total, seed, pass_index, round_index);
    CUDA_OK(cudaGetLastError());

    CUBLAS_OK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n, (int)n, (int)n,
        &alpha,
        b.data_ptr<int8_t>(), CUDA_R_8I, (int)n,
        a.data_ptr<int8_t>(), CUDA_R_8I, (int)n,
        &beta,
        q.data_ptr<int32_t>(), CUDA_R_32I, (int)n,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    reduce_q_kernel<<<blocks_reduce, threads, threads * sizeof(uint64_t), stream>>>(
        q.data_ptr<int32_t>(),
        total,
        reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>()),
        seed,
        pass_index,
        round_index
    );
    CUDA_OK(cudaGetLastError());
    reduce_partial_kernel<<<1, threads, 4 * threads * sizeof(uint64_t), stream>>>(
        reinterpret_cast<uint64_t*>(partial.data_ptr<int64_t>()),
        blocks_reduce,
        root.data_ptr<int64_t>(),
        seed,
        pass_index,
        round_index
    );
    CUDA_OK(cudaGetLastError());

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

static void upload_merkle_tags(cudaStream_t stream) {
    uint8_t block_tag[32] = {0};
    uint8_t leaf_tag[32] = {0};
    uint8_t node_tag[32] = {0};
    uint8_t state_block_tag[32] = {0};
    uint8_t state_leaf_tag[32] = {0};
    uint8_t state_node_tag[32] = {0};
    uint8_t fp64_block_tag[32] = {0};
    uint8_t fp64_leaf_tag[32] = {0};
    uint8_t fp64_node_tag[32] = {0};
    const char* block = "VERATHOS_Q_BLOCK_V1";
    const char* leaf = "VERATHOS_Q_LEAF_V1";
    const char* node = "VERATHOS_Q_NODE_V1";
    const char* state_block = "VERATHOS_AB_STATE_BLOCK_V1";
    const char* state_leaf = "VERATHOS_AB_STATE_LEAF_V1";
    const char* state_node = "VERATHOS_AB_STATE_NODE_V1";
    const char* fp64_block = "VERATHOS_FP64_C_BLOCK_V1";
    const char* fp64_leaf = "VERATHOS_FP64_C_LEAF_V1";
    const char* fp64_node = "VERATHOS_FP64_C_NODE_V1";
    memcpy(block_tag, block, strlen(block));
    memcpy(leaf_tag, leaf, strlen(leaf));
    memcpy(node_tag, node, strlen(node));
    memcpy(state_block_tag, state_block, strlen(state_block));
    memcpy(state_leaf_tag, state_leaf, strlen(state_leaf));
    memcpy(state_node_tag, state_node, strlen(state_node));
    memcpy(fp64_block_tag, fp64_block, strlen(fp64_block));
    memcpy(fp64_leaf_tag, fp64_leaf, strlen(fp64_leaf));
    memcpy(fp64_node_tag, fp64_node, strlen(fp64_node));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_Q_BLOCK_TAG_GPU, block_tag, sizeof(block_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_Q_LEAF_TAG_GPU, leaf_tag, sizeof(leaf_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_Q_NODE_TAG_GPU, node_tag, sizeof(node_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_STATE_PAIR_BLOCK_TAG_GPU, state_block_tag, sizeof(state_block_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_STATE_PAIR_LEAF_TAG_GPU, state_leaf_tag, sizeof(state_leaf_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_STATE_PAIR_NODE_TAG_GPU, state_node_tag, sizeof(state_node_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_FP64_C_BLOCK_TAG_GPU, fp64_block_tag, sizeof(fp64_block_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_FP64_C_LEAF_TAG_GPU, fp64_leaf_tag, sizeof(fp64_leaf_tag), 0, cudaMemcpyHostToDevice, stream));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        D_FP64_C_NODE_TAG_GPU, fp64_node_tag, sizeof(fp64_node_tag), 0, cudaMemcpyHostToDevice, stream));
}

std::tuple<at::Tensor, at::Tensor> q_merkle_commit_i32(
    at::Tensor q,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(q.dim() == 2, "q must be 2D");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    int64_t n64 = q.size(0);
    TORCH_CHECK(q.size(1) == n64, "q must be square");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "q dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "q dimension too large");

    int n = (int)n64;
    int bs = (int)block_size;
    int blocks_per_row = n / bs;
    int total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    upload_merkle_tags(stream);

    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    auto dev_cpu = torch::Device(torch::kCPU);
    at::Tensor flat_gpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    int hash_threads = 16;
    int hash_blocks = (total_blocks + hash_threads - 1) / hash_threads;
    q_i32_block_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
        q.data_ptr<int32_t>(), n, bs, block_hashes.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    int leaf_threads = 256;
    int leaf_blocks = (total_blocks + leaf_threads - 1) / leaf_threads;
    q_merkle_leaf_kernel<<<leaf_blocks, leaf_threads, 0, stream>>>(
        block_hashes.data_ptr<uint8_t>(), total_blocks, flat_gpu.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    size_t cur_count = (size_t)total_blocks;
    size_t read_offset = 0;
    size_t write_offset = cur_count * 32ULL;
    while (cur_count > 1) {
        size_t parent_count = (cur_count + 1) >> 1;
        int threads = (parent_count < 256) ? (int)parent_count : 256;
        if (threads <= 0) threads = 1;
        int blocks = (int)((parent_count + threads - 1) / threads);
        q_merkle_level_kernel<<<blocks, threads, 0, stream>>>(
            flat_gpu.data_ptr<uint8_t>() + read_offset,
            (int)cur_count,
            flat_gpu.data_ptr<uint8_t>() + write_offset);
        CUDA_OK(cudaGetLastError());
        read_offset = write_offset;
        write_offset += parent_count * 32ULL;
        cur_count = parent_count;
    }

    at::Tensor root_cpu = torch::empty({32}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        root_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToHost,
        stream));
    at::Tensor flat_cpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        flat_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>(),
        total_nodes * 32ULL,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return {root_cpu, flat_cpu};
}

std::tuple<at::Tensor, at::Tensor> fp64_c_merkle_commit_f64(
    at::Tensor c,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(c.dim() == 2, "c must be 2D");
    TORCH_CHECK(c.dtype() == torch::kFloat64, "c must be float64");
    TORCH_CHECK(c.is_cuda(), "c must be CUDA");
    TORCH_CHECK(c.is_contiguous(), "c must be contiguous");
    int64_t n64 = c.size(0);
    TORCH_CHECK(c.size(1) == n64, "c must be square");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "c dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "c dimension too large");

    int n = (int)n64;
    int bs = (int)block_size;
    int blocks_per_row = n / bs;
    int total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    upload_merkle_tags(stream);

    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    auto dev_cpu = torch::Device(torch::kCPU);
    at::Tensor flat_gpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    int hash_threads = 16;
    int hash_blocks = (total_blocks + hash_threads - 1) / hash_threads;
    fp64_c_block_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
        c.data_ptr<double>(), n, bs, block_hashes.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    int leaf_threads = 256;
    int leaf_blocks = (total_blocks + leaf_threads - 1) / leaf_threads;
    fp64_c_merkle_leaf_kernel<<<leaf_blocks, leaf_threads, 0, stream>>>(
        block_hashes.data_ptr<uint8_t>(), total_blocks, flat_gpu.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    size_t cur_count = (size_t)total_blocks;
    size_t read_offset = 0;
    size_t write_offset = cur_count * 32ULL;
    while (cur_count > 1) {
        size_t parent_count = (cur_count + 1) >> 1;
        int threads = (parent_count < 256) ? (int)parent_count : 256;
        if (threads <= 0) threads = 1;
        int blocks = (int)((parent_count + threads - 1) / threads);
        fp64_c_merkle_level_kernel<<<blocks, threads, 0, stream>>>(
            flat_gpu.data_ptr<uint8_t>() + read_offset,
            (int)cur_count,
            flat_gpu.data_ptr<uint8_t>() + write_offset);
        CUDA_OK(cudaGetLastError());
        read_offset = write_offset;
        write_offset += parent_count * 32ULL;
        cur_count = parent_count;
    }

    at::Tensor root_cpu = torch::empty({32}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        root_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToHost,
        stream));
    at::Tensor flat_cpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        flat_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>(),
        total_nodes * 32ULL,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return {root_cpu, flat_cpu};
}

static void fp64_c_merkle_root_store_scratch(
    at::Tensor c,
    int64_t block_size,
    at::Tensor out_roots,
    int64_t root_index,
    cudaStream_t stream,
    at::Tensor flat_gpu,
    at::Tensor block_hashes
) {
    TORCH_CHECK(c.dim() == 2, "c must be 2D");
    TORCH_CHECK(c.dtype() == torch::kFloat64, "c must be float64");
    TORCH_CHECK(c.is_cuda(), "c must be CUDA");
    TORCH_CHECK(c.is_contiguous(), "c must be contiguous");
    TORCH_CHECK(out_roots.is_cuda(), "out_roots must be CUDA");
    TORCH_CHECK(out_roots.dtype() == torch::kUInt8, "out_roots must be uint8");
    TORCH_CHECK(out_roots.dim() == 2 && out_roots.size(1) >= 32,
                "out_roots must be [num_roots, >=32]");
    TORCH_CHECK(root_index >= 0 && root_index < out_roots.size(0),
                "root_index out of range");
    TORCH_CHECK(flat_gpu.is_cuda() && flat_gpu.dtype() == torch::kUInt8 && flat_gpu.is_contiguous(),
                "flat_gpu must be contiguous CUDA uint8 scratch");
    TORCH_CHECK(block_hashes.is_cuda() && block_hashes.dtype() == torch::kUInt8 && block_hashes.is_contiguous(),
                "block_hashes must be contiguous CUDA uint8 scratch");
    int64_t n64 = c.size(0);
    TORCH_CHECK(c.size(1) == n64, "c must be square");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "c dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "c dimension too large");

    int n = (int)n64;
    int bs = (int)block_size;
    int blocks_per_row = n / bs;
    int total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    TORCH_CHECK(flat_gpu.numel() >= (int64_t)(total_nodes * 32ULL),
                "flat_gpu scratch too small");
    TORCH_CHECK(block_hashes.numel() >= (int64_t)(total_blocks * 32ULL),
                "block_hashes scratch too small");

    int hash_threads = 16;
    int hash_blocks = (total_blocks + hash_threads - 1) / hash_threads;
    fp64_c_block_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
        c.data_ptr<double>(), n, bs, block_hashes.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    int leaf_threads = 256;
    int leaf_blocks = (total_blocks + leaf_threads - 1) / leaf_threads;
    fp64_c_merkle_leaf_kernel<<<leaf_blocks, leaf_threads, 0, stream>>>(
        block_hashes.data_ptr<uint8_t>(), total_blocks, flat_gpu.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    size_t cur_count = (size_t)total_blocks;
    size_t read_offset = 0;
    size_t write_offset = cur_count * 32ULL;
    while (cur_count > 1) {
        size_t parent_count = (cur_count + 1) >> 1;
        int threads = (parent_count < 256) ? (int)parent_count : 256;
        if (threads <= 0) threads = 1;
        int blocks = (int)((parent_count + threads - 1) / threads);
        fp64_c_merkle_level_kernel<<<blocks, threads, 0, stream>>>(
            flat_gpu.data_ptr<uint8_t>() + read_offset,
            (int)cur_count,
            flat_gpu.data_ptr<uint8_t>() + write_offset);
        CUDA_OK(cudaGetLastError());
        read_offset = write_offset;
        write_offset += parent_count * 32ULL;
        cur_count = parent_count;
    }

    CUDA_OK(cudaMemcpyAsync(
        out_roots.data_ptr<uint8_t>() + (size_t)root_index * out_roots.stride(0),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToDevice,
        stream));
}

static void q_merkle_root_i32_store_scratch(
    at::Tensor q,
    int64_t block_size,
    at::Tensor out_roots,
    int64_t root_index,
    cudaStream_t stream,
    at::Tensor flat_gpu,
    at::Tensor block_hashes
) {
    TORCH_CHECK(q.dim() == 2, "q must be 2D");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(out_roots.is_cuda(), "out_roots must be CUDA");
    TORCH_CHECK(out_roots.dtype() == torch::kUInt8, "out_roots must be uint8");
    TORCH_CHECK(out_roots.dim() == 2 && out_roots.size(1) >= 32,
                "out_roots must be [num_roots, >=32]");
    TORCH_CHECK(root_index >= 0 && root_index < out_roots.size(0),
                "root_index out of range");
    TORCH_CHECK(flat_gpu.is_cuda() && flat_gpu.dtype() == torch::kUInt8 && flat_gpu.is_contiguous(),
                "flat_gpu must be contiguous CUDA uint8 scratch");
    TORCH_CHECK(block_hashes.is_cuda() && block_hashes.dtype() == torch::kUInt8 && block_hashes.is_contiguous(),
                "block_hashes must be contiguous CUDA uint8 scratch");
    int64_t n64 = q.size(0);
    TORCH_CHECK(q.size(1) == n64, "q must be square");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "q dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "q dimension too large");

    int n = (int)n64;
    int bs = (int)block_size;
    int blocks_per_row = n / bs;
    int total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    TORCH_CHECK(flat_gpu.numel() >= (int64_t)(total_nodes * 32ULL),
                "flat_gpu scratch too small");
    TORCH_CHECK(block_hashes.numel() >= (int64_t)(total_blocks * 32ULL),
                "block_hashes scratch too small");

    int hash_threads = 16;
    int hash_blocks = (total_blocks + hash_threads - 1) / hash_threads;
    q_i32_block_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
        q.data_ptr<int32_t>(), n, bs, block_hashes.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    int leaf_threads = 256;
    int leaf_blocks = (total_blocks + leaf_threads - 1) / leaf_threads;
    q_merkle_leaf_kernel<<<leaf_blocks, leaf_threads, 0, stream>>>(
        block_hashes.data_ptr<uint8_t>(), total_blocks, flat_gpu.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    size_t cur_count = (size_t)total_blocks;
    size_t read_offset = 0;
    size_t write_offset = cur_count * 32ULL;
    while (cur_count > 1) {
        size_t parent_count = (cur_count + 1) >> 1;
        int threads = (parent_count < 256) ? (int)parent_count : 256;
        if (threads <= 0) threads = 1;
        int blocks = (int)((parent_count + threads - 1) / threads);
        q_merkle_level_kernel<<<blocks, threads, 0, stream>>>(
            flat_gpu.data_ptr<uint8_t>() + read_offset,
            (int)cur_count,
            flat_gpu.data_ptr<uint8_t>() + write_offset);
        CUDA_OK(cudaGetLastError());
        read_offset = write_offset;
        write_offset += parent_count * 32ULL;
        cur_count = parent_count;
    }

    CUDA_OK(cudaMemcpyAsync(
        out_roots.data_ptr<uint8_t>() + (size_t)root_index * out_roots.stride(0),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToDevice,
        stream));
}

static void state_i8_pair_merkle_root_scratch(
    const int8_t* a,
    const int8_t* b,
    int64_t n64,
    int64_t block_size,
    cudaStream_t stream,
    at::Tensor flat_gpu,
    at::Tensor block_hashes
) {
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "state dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "state dimension too large");
    TORCH_CHECK(flat_gpu.is_cuda() && flat_gpu.dtype() == torch::kUInt8 && flat_gpu.is_contiguous(),
                "flat_gpu must be contiguous CUDA uint8 scratch");
    TORCH_CHECK(block_hashes.is_cuda() && block_hashes.dtype() == torch::kUInt8 && block_hashes.is_contiguous(),
                "block_hashes must be contiguous CUDA uint8 scratch");

    int n = (int)n64;
    int bs = (int)block_size;
    int blocks_per_row = n / bs;
    int total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    TORCH_CHECK(flat_gpu.numel() >= (int64_t)(total_nodes * 32ULL),
                "flat_gpu scratch too small");
    TORCH_CHECK(block_hashes.numel() >= (int64_t)(total_blocks * 32ULL),
                "block_hashes scratch too small");

    int hash_threads = 16;
    int hash_blocks = (total_blocks + hash_threads - 1) / hash_threads;
    state_i8_pair_block_hash_kernel<<<hash_blocks, hash_threads, 0, stream>>>(
        a, b, n, bs, block_hashes.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    int leaf_threads = 256;
    int leaf_blocks = (total_blocks + leaf_threads - 1) / leaf_threads;
    state_pair_merkle_leaf_kernel<<<leaf_blocks, leaf_threads, 0, stream>>>(
        block_hashes.data_ptr<uint8_t>(), total_blocks, flat_gpu.data_ptr<uint8_t>());
    CUDA_OK(cudaGetLastError());

    size_t cur_count = (size_t)total_blocks;
    size_t read_offset = 0;
    size_t write_offset = cur_count * 32ULL;
    while (cur_count > 1) {
        size_t parent_count = (cur_count + 1) >> 1;
        int threads = (parent_count < 256) ? (int)parent_count : 256;
        if (threads <= 0) threads = 1;
        int blocks = (int)((parent_count + threads - 1) / threads);
        state_pair_merkle_level_kernel<<<blocks, threads, 0, stream>>>(
            flat_gpu.data_ptr<uint8_t>() + read_offset,
            (int)cur_count,
            flat_gpu.data_ptr<uint8_t>() + write_offset);
        CUDA_OK(cudaGetLastError());
        read_offset = write_offset;
        write_offset += parent_count * 32ULL;
        cur_count = parent_count;
    }
}

static void state_i8_pair_merkle_root_store_scratch(
    const int8_t* a,
    const int8_t* b,
    int64_t n64,
    int64_t block_size,
    at::Tensor out_roots,
    int64_t root_index,
    cudaStream_t stream,
    at::Tensor flat_gpu,
    at::Tensor block_hashes
) {
    TORCH_CHECK(out_roots.is_cuda(), "out_roots must be CUDA");
    TORCH_CHECK(out_roots.dtype() == torch::kUInt8, "out_roots must be uint8");
    TORCH_CHECK(out_roots.dim() == 2 && out_roots.size(1) >= 32,
                "out_roots must be [num_roots, >=32]");
    TORCH_CHECK(root_index >= 0 && root_index < out_roots.size(0),
                "root_index out of range");
    state_i8_pair_merkle_root_scratch(
        a, b, n64, block_size, stream, flat_gpu, block_hashes);
    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    CUDA_OK(cudaMemcpyAsync(
        out_roots.data_ptr<uint8_t>() + (size_t)root_index * out_roots.stride(0),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToDevice,
        stream));
}

std::tuple<at::Tensor, at::Tensor> state_merkle_commit_i8(
    at::Tensor a,
    at::Tensor b,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "state tensors must be 2D");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8,
                "state tensors must be int8");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "state tensors must be CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "state tensors must be contiguous");
    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "state tensors must be same-sized square matrices");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(n64 % block_size == 0, "state dimension must be divisible by block_size");
    TORCH_CHECK(n64 <= INT32_MAX, "state dimension too large");

    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    upload_merkle_tags(stream);

    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    auto dev_cpu = torch::Device(torch::kCPU);
    at::Tensor flat_gpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));
    state_i8_pair_merkle_root_scratch(
        a.data_ptr<int8_t>(),
        b.data_ptr<int8_t>(),
        n64,
        block_size,
        stream,
        flat_gpu,
        block_hashes);

    at::Tensor root_cpu = torch::empty({32}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        root_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>() + (total_nodes - 1) * 32ULL,
        32,
        cudaMemcpyDeviceToHost,
        stream));
    at::Tensor flat_cpu = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cpu));
    CUDA_OK(cudaMemcpyAsync(
        flat_cpu.data_ptr<uint8_t>(),
        flat_gpu.data_ptr<uint8_t>(),
        total_nodes * 32ULL,
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    return {root_cpu, flat_cpu};
}

double run_workspace_pass_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(rounds > 0, "rounds must be > 0");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && q.is_cuda() && partial.is_cuda() && root.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "a/b must be int8");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(partial.dtype() == torch::kInt64, "partial must be int64");
    TORCH_CHECK(root.dtype() == torch::kInt64, "root must be int64");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && q.dim() == 2, "a/b/q must be matrices");
    TORCH_CHECK(root.numel() >= 4, "root must have at least 4 int64 words");
    TORCH_CHECK(partial.numel() > 0, "partial tensor must not be empty");
    TORCH_CHECK(pass_index >= 0, "pass_index must be non-negative");
    TORCH_CHECK(pass_index * rounds + rounds <= round_roots.size(0),
                "round_roots tensor too small for requested pass");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));
    upload_merkle_tags(stream);

    int64_t n64 = q.size(0);
    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    at::Tensor merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    for (int r = 0; r < rounds; ++r) {
        run_round_gemm_only(
            a, b, q, handle, stream, seed, pass_index, r,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        q_merkle_root_i32_store_scratch(
            q, block_size, round_roots, pass_index * rounds + r, stream,
            merkle_flat, merkle_block_hashes);
    }

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

double run_workspace_pass_transition_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor state_roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int rounds,
    int64_t block_size,
    int transition_mix_rounds,
    int transition_fanout,
    int device_index
) {
    TORCH_CHECK(transition_mix_rounds > 0, "transition_mix_rounds must be > 0");
    TORCH_CHECK(transition_fanout > 0, "transition_fanout must be > 0");
    TORCH_CHECK(rounds > 0, "rounds must be > 0");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && q.is_cuda() && partial.is_cuda() && root.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "a/b must be int8");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(partial.dtype() == torch::kInt64, "partial must be int64");
    TORCH_CHECK(root.dtype() == torch::kInt64, "root must be int64");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && q.dim() == 2, "a/b/q must be matrices");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && q.is_contiguous(), "a/b/q must be contiguous");
    TORCH_CHECK(state_roots.is_cuda() && state_roots.dtype() == torch::kUInt8,
                "state_roots must be CUDA uint8");
    TORCH_CHECK(round_roots.is_cuda() && round_roots.dtype() == torch::kUInt8,
                "round_roots must be CUDA uint8");
    TORCH_CHECK(pass_index >= 0, "pass_index must be non-negative");
    TORCH_CHECK(pass_index * rounds + rounds <= round_roots.size(0),
                "round_roots tensor too small for requested pass");
    TORCH_CHECK((pass_index * rounds + rounds) * (transition_mix_rounds + 1) <= state_roots.size(0),
                "state_roots tensor too small for requested pass");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));
    upload_merkle_tags(stream);

    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "a/b must be NxN");
    TORCH_CHECK(q.size(0) == n64 && q.size(1) == n64, "q must be NxN");
    int64_t total = n64 * n64;
    TORCH_CHECK(q.numel() * (int64_t)sizeof(int32_t) >= total * 2,
                "q scratch is too small for transition ping-pong buffers");
    TORCH_CHECK(n64 % block_size == 0, "matrix dimension must be divisible by block_size");

    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    int8_t* scratch = reinterpret_cast<int8_t*>(q.data_ptr<int32_t>());
    int8_t* scratch_a = scratch;
    int8_t* scratch_b = scratch + total;
    const int32_t alpha = 1;
    const int32_t beta = 0;

    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    at::Tensor state_merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor state_merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor q_merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor q_merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    for (int r = 0; r < rounds; ++r) {
        fill_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
            a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total, seed, pass_index, r);
        CUDA_OK(cudaGetLastError());

        const int8_t* cur_a = a.data_ptr<int8_t>();
        const int8_t* cur_b = b.data_ptr<int8_t>();
        int64_t state_root_base = (pass_index * rounds + r) * (transition_mix_rounds + 1);
        state_i8_pair_merkle_root_store_scratch(
            cur_a,
            cur_b,
            n64,
            block_size,
            state_roots,
            state_root_base,
            stream,
            state_merkle_flat,
            state_merkle_block_hashes);
        bool cur_is_scratch = false;
        for (int m = 0; m < transition_mix_rounds; ++m) {
            int8_t* dst_a = cur_is_scratch ? a.data_ptr<int8_t>() : scratch_a;
            int8_t* dst_b = cur_is_scratch ? b.data_ptr<int8_t>() : scratch_b;
            mix_i8_pair_transition_kernel<<<blocks_fill, threads, 0, stream>>>(
                cur_a, cur_b, dst_a, dst_b, total, seed, pass_index, r, m, transition_fanout);
            CUDA_OK(cudaGetLastError());
            cur_a = dst_a;
            cur_b = dst_b;
            cur_is_scratch = !cur_is_scratch;
            state_i8_pair_merkle_root_store_scratch(
                cur_a,
                cur_b,
                n64,
                block_size,
                state_roots,
                state_root_base + m + 1,
                stream,
                state_merkle_flat,
                state_merkle_block_hashes);
        }
        if (cur_is_scratch) {
            copy_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
                cur_a, cur_b, a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total);
            CUDA_OK(cudaGetLastError());
            cur_a = a.data_ptr<int8_t>();
            cur_b = b.data_ptr<int8_t>();
        }

        CUBLAS_OK(cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)n64, (int)n64, (int)n64,
            &alpha,
            cur_b, CUDA_R_8I, (int)n64,
            cur_a, CUDA_R_8I, (int)n64,
            &beta,
            q.data_ptr<int32_t>(), CUDA_R_32I, (int)n64,
            CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
        q_merkle_root_i32_store_scratch(
            q, block_size, round_roots, pass_index * rounds + r, stream,
            q_merkle_flat, q_merkle_block_hashes);
    }

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

double run_workspace_round_transition_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    at::Tensor partial,
    at::Tensor root,
    at::Tensor state_roots,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int64_t block_size,
    int transition_mix_rounds,
    int transition_fanout,
    int device_index
) {
    TORCH_CHECK(transition_mix_rounds > 0, "transition_mix_rounds must be > 0");
    TORCH_CHECK(transition_fanout > 0, "transition_fanout must be > 0");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && q.is_cuda() && partial.is_cuda() && root.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "a/b must be int8");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(partial.dtype() == torch::kInt64, "partial must be int64");
    TORCH_CHECK(root.dtype() == torch::kInt64, "root must be int64");
    TORCH_CHECK(state_roots.is_cuda() && state_roots.dtype() == torch::kUInt8,
                "state_roots must be CUDA uint8");
    TORCH_CHECK(round_roots.is_cuda() && round_roots.dtype() == torch::kUInt8,
                "round_roots must be CUDA uint8");
    TORCH_CHECK(state_roots.dim() == 2 && state_roots.size(1) >= 32,
                "state_roots must be [num_roots, >=32]");
    TORCH_CHECK(round_roots.dim() == 2 && round_roots.size(1) >= 32,
                "round_roots must be [num_roots, >=32]");
    TORCH_CHECK(state_roots.size(0) >= transition_mix_rounds + 1,
                "state_roots tensor too small for requested round");
    TORCH_CHECK(round_roots.size(0) >= 1,
                "round_roots tensor too small for requested round");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));
    upload_merkle_tags(stream);

    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "a/b must be NxN");
    TORCH_CHECK(q.size(0) == n64 && q.size(1) == n64, "q must be NxN");
    int64_t total = n64 * n64;
    TORCH_CHECK(q.numel() * (int64_t)sizeof(int32_t) >= total * 2,
                "q scratch is too small for transition ping-pong buffers");
    TORCH_CHECK(n64 % block_size == 0, "matrix dimension must be divisible by block_size");

    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    int8_t* scratch = reinterpret_cast<int8_t*>(q.data_ptr<int32_t>());
    int8_t* scratch_a = scratch;
    int8_t* scratch_b = scratch + total;
    const int32_t alpha = 1;
    const int32_t beta = 0;

    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    at::Tensor state_merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor state_merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor q_merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor q_merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    fill_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total, seed, pass_index, round_index);
    CUDA_OK(cudaGetLastError());

    const int8_t* cur_a = a.data_ptr<int8_t>();
    const int8_t* cur_b = b.data_ptr<int8_t>();
    state_i8_pair_merkle_root_store_scratch(
        cur_a,
        cur_b,
        n64,
        block_size,
        state_roots,
        0,
        stream,
        state_merkle_flat,
        state_merkle_block_hashes);
    bool cur_is_scratch = false;
    for (int m = 0; m < transition_mix_rounds; ++m) {
        int8_t* dst_a = cur_is_scratch ? a.data_ptr<int8_t>() : scratch_a;
        int8_t* dst_b = cur_is_scratch ? b.data_ptr<int8_t>() : scratch_b;
        mix_i8_pair_transition_kernel<<<blocks_fill, threads, 0, stream>>>(
            cur_a, cur_b, dst_a, dst_b, total, seed, pass_index, round_index, m, transition_fanout);
        CUDA_OK(cudaGetLastError());
        cur_a = dst_a;
        cur_b = dst_b;
        cur_is_scratch = !cur_is_scratch;
        state_i8_pair_merkle_root_store_scratch(
            cur_a,
            cur_b,
            n64,
            block_size,
            state_roots,
            m + 1,
            stream,
            state_merkle_flat,
            state_merkle_block_hashes);
    }
    if (cur_is_scratch) {
        copy_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
            cur_a, cur_b, a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total);
        CUDA_OK(cudaGetLastError());
        cur_a = a.data_ptr<int8_t>();
        cur_b = b.data_ptr<int8_t>();
    }

    CUBLAS_OK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)n64, (int)n64, (int)n64,
        &alpha,
        cur_b, CUDA_R_8I, (int)n64,
        cur_a, CUDA_R_8I, (int)n64,
        &beta,
        q.data_ptr<int32_t>(), CUDA_R_32I, (int)n64,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    q_merkle_root_i32_store_scratch(
        q, block_size, round_roots, 0, stream,
        q_merkle_flat, q_merkle_block_hashes);

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

double run_workspace_round_transition_state(
    at::Tensor a,
    at::Tensor b,
    at::Tensor q,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int transition_mix_rounds,
    int transition_fanout,
    int state_index,
    int device_index
) {
    TORCH_CHECK(transition_mix_rounds > 0, "transition_mix_rounds must be > 0");
    TORCH_CHECK(transition_fanout > 0, "transition_fanout must be > 0");
    TORCH_CHECK(state_index >= 0 && state_index <= transition_mix_rounds,
                "state_index must be in [0, transition_mix_rounds]");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && q.is_cuda(), "a/b/q must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "a/b must be int8");
    TORCH_CHECK(q.dtype() == torch::kInt32, "q must be int32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && q.dim() == 2, "a/b/q must be matrices");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && q.is_contiguous(), "a/b/q must be contiguous");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "a/b must be NxN");
    TORCH_CHECK(q.size(0) == n64 && q.size(1) == n64, "q must be NxN");
    int64_t total = n64 * n64;
    TORCH_CHECK(q.numel() * (int64_t)sizeof(int32_t) >= total * 2,
                "q scratch is too small for transition ping-pong buffers");

    int threads = 256;
    int blocks_fill = (int)((total + threads - 1) / threads);
    int8_t* scratch = reinterpret_cast<int8_t*>(q.data_ptr<int32_t>());
    int8_t* scratch_a = scratch;
    int8_t* scratch_b = scratch + total;

    fill_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total, seed, pass_index, round_index);
    CUDA_OK(cudaGetLastError());
    if (state_index == 0) {
        CUDA_OK(cudaStreamSynchronize(stream));
        return 0.0;
    }

    const int8_t* cur_a = a.data_ptr<int8_t>();
    const int8_t* cur_b = b.data_ptr<int8_t>();
    bool cur_is_scratch = false;
    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));
    for (int m = 0; m < state_index; ++m) {
        int8_t* dst_a = cur_is_scratch ? a.data_ptr<int8_t>() : scratch_a;
        int8_t* dst_b = cur_is_scratch ? b.data_ptr<int8_t>() : scratch_b;
        mix_i8_pair_transition_kernel<<<blocks_fill, threads, 0, stream>>>(
            cur_a, cur_b, dst_a, dst_b, total, seed, pass_index, round_index, m, transition_fanout);
        CUDA_OK(cudaGetLastError());
        cur_a = dst_a;
        cur_b = dst_b;
        cur_is_scratch = !cur_is_scratch;
    }
    if (cur_is_scratch) {
        copy_i8_pair_kernel<<<blocks_fill, threads, 0, stream>>>(
            cur_a, cur_b, a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), total);
        CUDA_OK(cudaGetLastError());
    }
    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

double run_fp64_identity_round_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t pass_index,
    int round_index,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda() && round_roots.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat64 && b.dtype() == torch::kFloat64 && c.dtype() == torch::kFloat64,
                "a/b/c must be float64");
    TORCH_CHECK(round_roots.dtype() == torch::kUInt8, "round_roots must be uint8");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "a/b/c must be matrices");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "a/b/c must be contiguous");
    TORCH_CHECK(round_roots.dim() == 2 && round_roots.size(1) >= 32,
                "round_roots must be [num_roots, >=32]");
    TORCH_CHECK(round_roots.size(0) >= 1, "round_roots tensor too small");
    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "a/b must be NxN");
    TORCH_CHECK(c.size(0) == n64 && c.size(1) == n64, "c must be NxN");
    TORCH_CHECK(n64 % block_size == 0, "matrix dimension must be divisible by block_size");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));
    upload_merkle_tags(stream);

    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    at::Tensor merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    run_fp64_identity_round_only(a, b, c, handle, stream, seed, pass_index, round_index);
    fp64_c_merkle_root_store_scratch(
        c, block_size, round_roots, 0, stream,
        merkle_flat, merkle_block_hashes);

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}

double run_fp64_identity_chain_committed(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor round_roots,
    uint64_t seed,
    int64_t start_pass,
    int64_t passes,
    int rounds,
    int64_t block_size,
    int device_index
) {
    TORCH_CHECK(passes >= 0, "passes must be >= 0");
    TORCH_CHECK(rounds > 0, "rounds must be > 0");
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda() && round_roots.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat64 && b.dtype() == torch::kFloat64 && c.dtype() == torch::kFloat64,
                "a/b/c must be float64");
    TORCH_CHECK(round_roots.dtype() == torch::kUInt8, "round_roots must be uint8");
    TORCH_CHECK(round_roots.dim() == 2 && round_roots.size(1) >= 32,
                "round_roots must be [num_roots, >=32]");
    TORCH_CHECK((start_pass + passes) * rounds <= round_roots.size(0),
                "round_roots tensor too small for requested chain");
    int64_t n64 = a.size(0);
    TORCH_CHECK(a.size(1) == n64 && b.size(0) == n64 && b.size(1) == n64,
                "a/b must be NxN");
    TORCH_CHECK(c.size(0) == n64 && c.size(1) == n64, "c must be NxN");
    TORCH_CHECK(n64 % block_size == 0, "matrix dimension must be divisible by block_size");

    c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device_index)};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
    cublasHandle_t handle = get_handle(device_index);
    CUBLAS_OK(cublasSetStream(handle, stream));
    upload_merkle_tags(stream);

    int64_t blocks_per_row = n64 / block_size;
    int64_t total_blocks = blocks_per_row * blocks_per_row;
    size_t total_nodes = merkle_total_nodes((size_t)total_blocks);
    auto dev_cuda = torch::Device(torch::kCUDA, device_index);
    at::Tensor merkle_flat = torch::empty(
        {(long)(total_nodes * 32)}, torch::dtype(torch::kUInt8).device(dev_cuda));
    at::Tensor merkle_block_hashes = torch::empty(
        {total_blocks, 32}, torch::dtype(torch::kUInt8).device(dev_cuda));

    cudaEvent_t ev_start, ev_stop;
    CUDA_OK(cudaEventCreate(&ev_start));
    CUDA_OK(cudaEventCreate(&ev_stop));
    CUDA_OK(cudaEventRecord(ev_start, stream));

    for (int64_t p = 0; p < passes; ++p) {
        int64_t pass_index = start_pass + p;
        for (int r = 0; r < rounds; ++r) {
            run_fp64_identity_round_only(a, b, c, handle, stream, seed, pass_index, r);
            fp64_c_merkle_root_store_scratch(
                c, block_size, round_roots, pass_index * rounds + r, stream,
                merkle_flat, merkle_block_hashes);
        }
    }

    CUDA_OK(cudaEventRecord(ev_stop, stream));
    CUDA_OK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    CUDA_OK(cudaEventDestroy(ev_start));
    CUDA_OK(cudaEventDestroy(ev_stop));
    return (double)ms;
}
