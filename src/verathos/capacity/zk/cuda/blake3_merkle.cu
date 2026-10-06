/**
 * blake3_merkle.cu — GPU-accelerated BLAKE3 Merkle tree leaf hashing.
 *
 * Computes hash_leaf(hash_flat_chunk(chunk_bytes)) for all chunks of an int8
 * weight tensor in parallel on GPU.  Each thread handles one chunk (128 bytes).
 *
 * The BLAKE3 implementation covers single-chunk messages only (≤1024 bytes),
 * which is sufficient for all VeriLLM Merkle tree operations:
 *   - chunk hash: 21 + 128 = 149 bytes
 *   - leaf  hash: 15 + 32  = 47  bytes
 *   - node  hash: 15 + 64  = 79  bytes  (not used here, tree built on CPU)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <tuple>
#include <vector>

// ======================================================================
//  BLAKE3 constants
// ======================================================================

static __constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// Pre-computed message word schedule (7 rounds).
// Round 0 = identity, subsequent rounds apply the BLAKE3 permutation.
static __constant__ int MSG_SCHEDULE[7][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    { 2,  6,  3, 10,  7,  0,  4, 13,  1, 11, 12,  5,  9, 14, 15,  8},
    { 3,  4, 10, 12, 13,  2,  7, 14,  6,  5,  9,  0, 11, 15,  8,  1},
    {10,  7, 12,  9, 14,  3, 13, 15,  4,  0, 11,  2,  5,  8,  1,  6},
    {12, 13,  9, 11, 15, 10, 14,  8,  7,  2,  5,  3,  0,  1,  6,  4},
    { 9, 14, 11,  5,  8, 12, 15,  1, 13,  3,  0, 10,  2,  6,  4,  7},
    {11, 15,  5,  0,  1,  9,  8,  6, 14, 10,  2, 12,  3,  4,  7, 13},
};

// BLAKE3 flags
#define BLAKE3_CHUNK_START  1u
#define BLAKE3_CHUNK_END    2u
#define BLAKE3_ROOT         8u

// Domain separation tags (matching verallm/crypto/merkle.py)
// DOMAIN_FLAT_CHUNK = b"VERILLM_FLAT_CHUNK_V1" (21 bytes)
static __constant__ uint8_t DOMAIN_FLAT_CHUNK[21] = {
    0x56, 0x45, 0x52, 0x49, 0x4c, 0x4c, 0x4d, 0x5f,  // VERALLM_
    0x46, 0x4c, 0x41, 0x54, 0x5f, 0x43, 0x48, 0x55,  // FLAT_CHU
    0x4e, 0x4b, 0x5f, 0x56, 0x31                       // NK_V1
};
#define DOMAIN_FLAT_CHUNK_LEN 21

// DOMAIN_LEAF = b"VERILLM_LEAF_V1" (15 bytes)
static __constant__ uint8_t DOMAIN_LEAF[15] = {
    0x56, 0x45, 0x52, 0x49, 0x4c, 0x4c, 0x4d, 0x5f,  // VERALLM_
    0x4c, 0x45, 0x41, 0x46, 0x5f, 0x56, 0x31           // LEAF_V1
};
#define DOMAIN_LEAF_LEN 15

// DOMAIN_NODE = b"VERILLM_NODE_V1" (15 bytes)
static __constant__ uint8_t DOMAIN_NODE[15] = {
    0x56, 0x45, 0x52, 0x49, 0x4c, 0x4c, 0x4d, 0x5f,  // VERALLM_
    0x4e, 0x4f, 0x44, 0x45, 0x5f, 0x56, 0x31           // NODE_V1
};
#define DOMAIN_NODE_LEN 15

// Thread block size
#define BLAKE3_THREADS 128
#define BLAKE3_RUNTIME_TENSOR_MAX_STAGES 128

struct RuntimeTensorSources {
    const uint8_t* first[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    const uint8_t* second[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    int32_t row_bytes[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    int32_t source_rows[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    int64_t source_row_stride_bytes[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    const int32_t* source_indices[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
    uint8_t combine_bfloat16[BLAKE3_RUNTIME_TENSOR_MAX_STAGES];
};

/** Copy selected live runtime rows into compact immutable staging.
 *
 * One block owns one row/stage pair and walks the complete row with coalesced
 * 16-byte transfers. The same source structure carries destination pointers
 * in ``second``; staged rows are tightly packed, so no second stride table is
 * required. Invalid scheduler slots are zeroed and marked -1 for the indexed
 * root reducer, preserving its fail-closed zero-root behavior.
 */
__global__ void copy_runtime_tensor_indexed_rows_kernel(
    RuntimeTensorSources tensors,
    const int32_t* __restrict__ source_indices,
    int32_t* __restrict__ staged_indices,
    int32_t stage_count,
    int32_t row_count)
{
    const int32_t row = (int32_t)blockIdx.x;
    const int32_t stage = (int32_t)blockIdx.y;
    if (stage >= stage_count || row >= row_count) return;
    const int32_t source_row = source_indices[
        (int64_t)stage * row_count + row
    ];
    const bool valid = (
        source_row >= 0 && source_row < tensors.source_rows[stage]
    );
    if (threadIdx.x == 0) {
        staged_indices[(int64_t)stage * row_count + row] = (
            valid ? row : -1
        );
    }
    const int32_t row_bytes = tensors.row_bytes[stage];
    const uint8_t* source = valid
        ? tensors.first[stage]
            + (int64_t)source_row
                * tensors.source_row_stride_bytes[stage]
        : nullptr;
    uint8_t* destination = const_cast<uint8_t*>(tensors.second[stage])
        + (int64_t)row * row_bytes;
    const bool vectorized = (
        valid
        && row_bytes % 16 == 0
        && tensors.source_row_stride_bytes[stage] % 16 == 0
        && ((uintptr_t)source % 16) == 0
        && ((uintptr_t)destination % 16) == 0
    );
    if (vectorized) {
        const int32_t vectors = row_bytes / 16;
        const uint4* source4 = reinterpret_cast<const uint4*>(source);
        uint4* destination4 = reinterpret_cast<uint4*>(destination);
        for (
            int32_t offset = (int32_t)threadIdx.x;
            offset < vectors;
            offset += (int32_t)blockDim.x
        ) {
            destination4[offset] = source4[offset];
        }
        return;
    }
    for (
        int32_t offset = (int32_t)threadIdx.x;
        offset < row_bytes;
        offset += (int32_t)blockDim.x
    ) {
        destination[offset] = valid ? source[offset] : 0;
    }
}

// ======================================================================
//  BLAKE3 device functions
// ======================================================================

/**
 * Rotate right for 32-bit unsigned integer.
 */
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/**
 * BLAKE3 quarter-round (G function).
 * Mixes four state words with two message words.
 */
__device__ __forceinline__ void blake3_G(
    uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d,
    uint32_t mx, uint32_t my)
{
    a = a + b + mx;   d = rotr32(d ^ a, 16);
    c = c + d;        b = rotr32(b ^ c, 12);
    a = a + b + my;   d = rotr32(d ^ a, 8);
    c = c + d;        b = rotr32(b ^ c, 7);
}

/**
 * BLAKE3 compression function.
 *
 * Processes one 64-byte block.  Updates cv[8] in-place with the output
 * chaining value (first 8 words of the finalized state).
 *
 * @param cv          Input/output chaining value (8 × uint32)
 * @param block       Message block as 16 little-endian uint32 words
 * @param counter_lo  Block counter low 32 bits (0 for single-chunk)
 * @param counter_hi  Block counter high 32 bits (0 for single-chunk)
 * @param block_len   Actual number of message bytes in this block (≤ 64)
 * @param flags       BLAKE3 flags (CHUNK_START, CHUNK_END, ROOT)
 */
__device__ void blake3_compress(
    uint32_t cv[8],
    const uint32_t block[16],
    uint32_t counter_lo,
    uint32_t counter_hi,
    uint32_t block_len,
    uint32_t flags)
{
    uint32_t s[16];

    // Initialize state
    s[0]  = cv[0];  s[1]  = cv[1];  s[2]  = cv[2];  s[3]  = cv[3];
    s[4]  = cv[4];  s[5]  = cv[5];  s[6]  = cv[6];  s[7]  = cv[7];
    s[8]  = BLAKE3_IV[0];  s[9]  = BLAKE3_IV[1];
    s[10] = BLAKE3_IV[2];  s[11] = BLAKE3_IV[3];
    s[12] = counter_lo;    s[13] = counter_hi;
    s[14] = block_len;     s[15] = flags;

    // 7 rounds with pre-computed message schedule
    #pragma unroll
    for (int r = 0; r < 7; r++) {
        // Column step
        blake3_G(s[0], s[4], s[8],  s[12], block[MSG_SCHEDULE[r][0]],  block[MSG_SCHEDULE[r][1]]);
        blake3_G(s[1], s[5], s[9],  s[13], block[MSG_SCHEDULE[r][2]],  block[MSG_SCHEDULE[r][3]]);
        blake3_G(s[2], s[6], s[10], s[14], block[MSG_SCHEDULE[r][4]],  block[MSG_SCHEDULE[r][5]]);
        blake3_G(s[3], s[7], s[11], s[15], block[MSG_SCHEDULE[r][6]],  block[MSG_SCHEDULE[r][7]]);
        // Diagonal step
        blake3_G(s[0], s[5], s[10], s[15], block[MSG_SCHEDULE[r][8]],  block[MSG_SCHEDULE[r][9]]);
        blake3_G(s[1], s[6], s[11], s[12], block[MSG_SCHEDULE[r][10]], block[MSG_SCHEDULE[r][11]]);
        blake3_G(s[2], s[7], s[8],  s[13], block[MSG_SCHEDULE[r][12]], block[MSG_SCHEDULE[r][13]]);
        blake3_G(s[3], s[4], s[9],  s[14], block[MSG_SCHEDULE[r][14]], block[MSG_SCHEDULE[r][15]]);
    }

    // Finalize: output = state[i] ^ state[i+8]
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        cv[i] = s[i] ^ s[i + 8];
    }
}

/**
 * Load 4 bytes from a buffer as a little-endian uint32.
 */
__device__ __forceinline__ uint32_t load_le32(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

/**
 * Store a uint32 as 4 little-endian bytes.
 */
__device__ __forceinline__ void store_le32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

/**
 * Hash a short message (≤ 1024 bytes) with BLAKE3.
 *
 * The message is treated as a single BLAKE3 chunk: split into 64-byte blocks,
 * compressed sequentially with CHUNK_START on first block, CHUNK_END|ROOT on
 * last block.  Output is 32 bytes written to `out`.
 *
 * @param msg      Message bytes (need not be aligned)
 * @param msg_len  Length in bytes (must be ≤ 1024)
 * @param out      Output buffer for 32-byte digest
 */
__device__ void blake3_hash_short(const uint8_t* msg, int msg_len, uint8_t out[32]) {
    int num_blocks = (msg_len + 63) / 64;
    if (num_blocks == 0) num_blocks = 1;  // empty message: one empty block

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

    for (int b = 0; b < num_blocks; b++) {
        // Load 64-byte block as 16 LE uint32 words (zero-pad if needed)
        uint32_t block_words[16];
        int block_start = b * 64;

        #pragma unroll
        for (int w = 0; w < 16; w++) {
            int byte_off = block_start + w * 4;
            uint8_t b0 = (byte_off     < msg_len) ? msg[byte_off]     : 0;
            uint8_t b1 = (byte_off + 1 < msg_len) ? msg[byte_off + 1] : 0;
            uint8_t b2 = (byte_off + 2 < msg_len) ? msg[byte_off + 2] : 0;
            uint8_t b3 = (byte_off + 3 < msg_len) ? msg[byte_off + 3] : 0;
            block_words[w] = (uint32_t)b0
                           | ((uint32_t)b1 << 8)
                           | ((uint32_t)b2 << 16)
                           | ((uint32_t)b3 << 24);
        }

        // Compute flags
        uint32_t flags = 0;
        if (b == 0)               flags |= BLAKE3_CHUNK_START;
        if (b == num_blocks - 1)  flags |= (BLAKE3_CHUNK_END | BLAKE3_ROOT);

        // Actual bytes in this block
        uint32_t block_len = (b == num_blocks - 1)
                           ? (uint32_t)(msg_len - b * 64)
                           : 64u;

        blake3_compress(cv, block_words, 0, 0, block_len, flags);
    }

    // Write 32-byte digest (8 LE uint32 words)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_le32(out + i * 4, cv[i]);
    }
}

// ======================================================================
//  Main kernel: compute Merkle leaf hashes for all chunks
// ======================================================================

/**
 * Each thread computes:
 *   chunk_hash = blake3(DOMAIN_FLAT_CHUNK + W_data[chunk_start:chunk_end])
 *   leaf_hash  = blake3(DOMAIN_LEAF + chunk_hash)
 *
 * This matches the CPU path:
 *   hash_leaf(hash_flat_chunk(chunk_bytes))
 * which is what MerkleTree._build_tree() stores at level 0.
 */
__global__ void blake3_merkle_leaf_hashes_kernel(
    uint8_t* __restrict__       leaf_hashes,   // [num_chunks × 32] output
    const int8_t* __restrict__  W_data,        // flattened int8 weight tensor
    int64_t                     total_elements,
    int32_t                     chunk_size,
    int32_t                     num_chunks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chunks) return;

    // ---- Step 1: Build message = DOMAIN_FLAT_CHUNK + chunk_bytes ----
    int64_t elem_start = (int64_t)tid * chunk_size;
    int64_t elem_end   = elem_start + chunk_size;
    if (elem_end > total_elements) elem_end = total_elements;
    int actual_bytes = (int)(elem_end - elem_start);
    int msg_len = DOMAIN_FLAT_CHUNK_LEN + actual_bytes;

    // Build message in local memory (max 21 + 128 = 149 bytes, fits in 3 blocks)
    // Using a fixed-size buffer avoids variable-length array issues.
    uint8_t msg[192];  // 3 × 64, enough for domain + chunk_size bytes

    // Zero-fill (covers padding for partial last block)
    #pragma unroll
    for (int i = 0; i < 192; i++) msg[i] = 0;

    // Copy domain tag from constant memory
    #pragma unroll
    for (int i = 0; i < DOMAIN_FLAT_CHUNK_LEN; i++) {
        msg[i] = DOMAIN_FLAT_CHUNK[i];
    }

    // Copy chunk data from global memory
    for (int i = 0; i < actual_bytes; i++) {
        msg[DOMAIN_FLAT_CHUNK_LEN + i] = (uint8_t)W_data[elem_start + i];
    }

    // ---- Step 2: chunk_hash = blake3(msg) ----
    uint8_t chunk_hash[32];
    blake3_hash_short(msg, msg_len, chunk_hash);

    // ---- Step 3: leaf_hash = blake3(DOMAIN_LEAF + chunk_hash) ----
    // Build 47-byte message in-place (reuse msg buffer)
    #pragma unroll
    for (int i = 0; i < 64; i++) msg[i] = 0;

    #pragma unroll
    for (int i = 0; i < DOMAIN_LEAF_LEN; i++) {
        msg[i] = DOMAIN_LEAF[i];
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        msg[DOMAIN_LEAF_LEN + i] = chunk_hash[i];
    }

    uint8_t leaf_hash[32];
    blake3_hash_short(msg, DOMAIN_LEAF_LEN + 32, leaf_hash);

    // ---- Step 4: Write 32-byte leaf hash to output ----
    int out_offset = tid * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        leaf_hashes[out_offset + i] = leaf_hash[i];
    }
}

// ======================================================================
//  Activation Merkle: hash_leaf(int64_block_bytes) via multi-chunk BLAKE3
// ======================================================================

// BLAKE3 PARENT flag
#define BLAKE3_PARENT  4u

/**
 * Hash a single BLAKE3 chunk (up to 1024 bytes) with a given counter.
 * Returns a 32-byte chaining value (not finalized as root).
 *
 * Unlike blake3_hash_short, this does NOT set ROOT flag — the caller
 * decides whether to finalize or chain into a parent hash.
 */
__device__ void blake3_hash_chunk(
    const uint8_t* data, int data_len, uint64_t counter,
    uint8_t cv_out[32])
{
    int num_blocks = (data_len + 63) / 64;
    if (num_blocks == 0) num_blocks = 1;

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

    for (int b = 0; b < num_blocks; b++) {
        uint32_t block_words[16];
        int block_start = b * 64;

        #pragma unroll
        for (int w = 0; w < 16; w++) {
            int byte_off = block_start + w * 4;
            uint8_t b0 = (byte_off     < data_len) ? data[byte_off]     : 0;
            uint8_t b1 = (byte_off + 1 < data_len) ? data[byte_off + 1] : 0;
            uint8_t b2 = (byte_off + 2 < data_len) ? data[byte_off + 2] : 0;
            uint8_t b3 = (byte_off + 3 < data_len) ? data[byte_off + 3] : 0;
            block_words[w] = (uint32_t)b0
                           | ((uint32_t)b1 << 8)
                           | ((uint32_t)b2 << 16)
                           | ((uint32_t)b3 << 24);
        }

        uint32_t flags = 0;
        if (b == 0)               flags |= BLAKE3_CHUNK_START;
        if (b == num_blocks - 1)  flags |= BLAKE3_CHUNK_END;

        uint32_t block_len = (b == num_blocks - 1)
                           ? (uint32_t)(data_len - b * 64)
                           : 64u;

        blake3_compress(cv, block_words,
                       (uint32_t)(counter & 0xFFFFFFFF),
                       (uint32_t)(counter >> 32),
                       block_len, flags);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_le32(cv_out + i * 4, cv[i]);
    }
}

/**
 * BLAKE3 parent node compression.
 *
 * Combines two 32-byte chaining values (left_cv ‖ right_cv) into one
 * 32-byte output using the PARENT flag.  The two CVs form a 64-byte
 * message block.
 *
 * @param left_cv   Left child chaining value (32 bytes)
 * @param right_cv  Right child chaining value (32 bytes)
 * @param is_root   If true, sets ROOT flag (final output)
 * @param out       Output 32-byte hash
 */
__device__ void blake3_parent_hash(
    const uint8_t left_cv[32], const uint8_t right_cv[32],
    bool is_root, uint8_t out[32])
{
    // Parent message = left_cv ‖ right_cv (64 bytes = 16 uint32 words)
    uint32_t block_words[16];
    for (int i = 0; i < 8; i++) {
        block_words[i]     = load_le32(left_cv + i * 4);
        block_words[i + 8] = load_le32(right_cv + i * 4);
    }

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

    uint32_t flags = BLAKE3_PARENT;
    if (is_root) flags |= BLAKE3_ROOT;

    blake3_compress(cv, block_words, 0, 0, 64, flags);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_le32(out + i * 4, cv[i]);
    }
}

/**
 * Load one 32-bit word from the virtual message
 * ``DOMAIN_LEAF || raw[0:data_bytes]`` without materializing that message.
 *
 * Activation rows are normally 2 KiB.  The old kernel copied each 1 KiB
 * BLAKE3 chunk byte-by-byte into thread-local memory and then read it back
 * byte-by-byte, causing large local-memory spills.  Only the first four words
 * and the final partial word straddle a message boundary; all interior words
 * can read the runtime bytes directly.
 */
__device__ __forceinline__ uint32_t load_leaf_message_word(
    const uint8_t* raw,
    int valid_data_bytes,
    int padded_data_bytes,
    int message_offset)
{
    const int data_offset = message_offset - DOMAIN_LEAF_LEN;
    if (
        data_offset >= 0
        && data_offset + 4 <= valid_data_bytes
    ) {
        return load_le32(raw + data_offset);
    }

    uint32_t word = 0;
    #pragma unroll
    for (int byte_index = 0; byte_index < 4; byte_index++) {
        const int offset = message_offset + byte_index;
        uint8_t value = 0;
        if (offset < DOMAIN_LEAF_LEN) {
            value = DOMAIN_LEAF[offset];
        } else {
            const int source = offset - DOMAIN_LEAF_LEN;
            if (
                source >= 0
                && source < valid_data_bytes
                && source < padded_data_bytes
            ) {
                value = raw[source];
            }
        }
        word |= (uint32_t)value << (8 * byte_index);
    }
    return word;
}

/**
 * Hash one BLAKE3 chunk of ``DOMAIN_LEAF || raw`` directly from global
 * memory.  Returns a chaining value (ROOT is applied only by the caller's
 * parent reduction).
 */
__device__ void blake3_hash_leaf_message_chunk(
    const uint8_t* raw,
    int valid_data_bytes,
    int padded_data_bytes,
    int message_len,
    int chunk_index,
    uint8_t cv_out[32])
{
    const int chunk_start = chunk_index * 1024;
    int chunk_len = message_len - chunk_start;
    if (chunk_len > 1024) chunk_len = 1024;
    int num_blocks = (chunk_len + 63) / 64;
    if (num_blocks == 0) num_blocks = 1;

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

    for (int block_index = 0; block_index < num_blocks; block_index++) {
        uint32_t block_words[16];
        const int block_offset = (
            chunk_start + block_index * 64
        );
        #pragma unroll
        for (int word_index = 0; word_index < 16; word_index++) {
            block_words[word_index] = load_leaf_message_word(
                raw,
                valid_data_bytes,
                padded_data_bytes,
                block_offset + word_index * 4
            );
        }

        uint32_t flags = 0;
        if (block_index == 0) flags |= BLAKE3_CHUNK_START;
        if (block_index == num_blocks - 1) flags |= BLAKE3_CHUNK_END;
        const uint32_t block_len = (
            block_index == num_blocks - 1
            ? (uint32_t)(chunk_len - block_index * 64)
            : 64u
        );
        blake3_compress(
            cv,
            block_words,
            (uint32_t)chunk_index,
            0,
            block_len,
            flags
        );
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_le32(cv_out + i * 4, cv[i]);
    }
}

__device__ __forceinline__ uint8_t load_runtime_tensor_byte(
    const uint8_t* first,
    const uint8_t* second,
    bool combine_bfloat16,
    int32_t byte_offset)
{
    if (!combine_bfloat16) return first[byte_offset];

    union BFloat16Bits {
        __nv_bfloat16 value;
        uint16_t bits;
    } left, right, output;
    const int32_t element_offset = byte_offset & ~1;
    left.value = *reinterpret_cast<const __nv_bfloat16*>(
        first + element_offset
    );
    right.value = *reinterpret_cast<const __nv_bfloat16*>(
        second + element_offset
    );
    output.value = __float2bfloat16_rn(
        __bfloat162float(left.value) + __bfloat162float(right.value)
    );
    return (uint8_t)(output.bits >> (8 * (byte_offset & 1)));
}

__device__ __forceinline__ uint32_t load_runtime_tensor_leaf_word(
    const uint8_t* first,
    const uint8_t* second,
    bool combine_bfloat16,
    int valid_data_bytes,
    int padded_data_bytes,
    int message_offset)
{
    const int data_offset = message_offset - DOMAIN_LEAF_LEN;
    if (
        !combine_bfloat16
        && data_offset >= 0
        && data_offset + 4 <= valid_data_bytes
    ) {
        return load_le32(first + data_offset);
    }

    uint32_t word = 0;
    #pragma unroll
    for (int byte_index = 0; byte_index < 4; byte_index++) {
        const int offset = message_offset + byte_index;
        uint8_t value = 0;
        if (offset < DOMAIN_LEAF_LEN) {
            value = DOMAIN_LEAF[offset];
        } else {
            const int source = offset - DOMAIN_LEAF_LEN;
            if (
                source >= 0
                && source < valid_data_bytes
                && source < padded_data_bytes
            ) {
                value = load_runtime_tensor_byte(
                    first,
                    second,
                    combine_bfloat16,
                    source
                );
            }
        }
        word |= (uint32_t)value << (8 * byte_index);
    }
    return word;
}

__device__ void blake3_hash_runtime_tensor_leaf_chunk(
    const uint8_t* first,
    const uint8_t* second,
    bool combine_bfloat16,
    int valid_data_bytes,
    int padded_data_bytes,
    int message_len,
    int chunk_index,
    uint8_t cv_out[32])
{
    const int chunk_start = chunk_index * 1024;
    int chunk_len = message_len - chunk_start;
    if (chunk_len > 1024) chunk_len = 1024;
    int num_blocks = (chunk_len + 63) / 64;
    if (num_blocks == 0) num_blocks = 1;

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) cv[i] = BLAKE3_IV[i];

    for (int block_index = 0; block_index < num_blocks; block_index++) {
        uint32_t block_words[16];
        const int block_offset = chunk_start + block_index * 64;
        #pragma unroll
        for (int word_index = 0; word_index < 16; word_index++) {
            block_words[word_index] = load_runtime_tensor_leaf_word(
                first,
                second,
                combine_bfloat16,
                valid_data_bytes,
                padded_data_bytes,
                block_offset + word_index * 4
            );
        }

        uint32_t flags = 0;
        if (block_index == 0) flags |= BLAKE3_CHUNK_START;
        if (block_index == num_blocks - 1) flags |= BLAKE3_CHUNK_END;
        const uint32_t block_len = (
            block_index == num_blocks - 1
            ? (uint32_t)(chunk_len - block_index * 64)
            : 64u
        );
        blake3_compress(
            cv,
            block_words,
            (uint32_t)chunk_index,
            0,
            block_len,
            flags
        );
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        store_le32(cv_out + i * 4, cv[i]);
    }
}

/**
 * Compute blake3(DOMAIN_LEAF + int64_block_bytes) for activation
 * Merkle leaves.
 *
 * Message = DOMAIN_LEAF (15 bytes) + block_data (chunk_size * 8 bytes).
 * For chunk_size=256: msg_len = 15 + 2048 = 2063 bytes → 3 BLAKE3 chunks.
 *
 * Multi-chunk BLAKE3:
 *   cv0 = hash_chunk(msg[0:1024], counter=0)
 *   cv1 = hash_chunk(msg[1024:2048], counter=1)
 *   cv2 = hash_chunk(msg[2048:2063], counter=2)
 *   p01 = parent_hash(cv0, cv1, is_root=false)
 *   result = parent_hash(p01, cv2, is_root=true)
 */
__global__ void blake3_activation_leaf_kernel(
    uint8_t* __restrict__        leaf_hashes,    // [num_chunks × 32] output
    const int64_t* __restrict__  data,           // flattened int64 activation
    int64_t                      total_elements,
    int32_t                      chunk_size,     // in int64 elements (e.g. 256)
    int32_t                      num_chunks)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_chunks) return;

    int64_t elem_start = (int64_t)tid * chunk_size;
    int64_t elem_end   = elem_start + chunk_size;
    if (elem_end > total_elements) elem_end = total_elements;
    int actual_elems = (int)(elem_end - elem_start);
    int data_bytes = actual_elems * 8;  // int64 = 8 bytes each
    int msg_len = DOMAIN_LEAF_LEN + data_bytes;

    // Build the full message in thread-local memory.
    // Max: 15 + 256*8 = 2063 bytes.  Round up to 3 BLAKE3 chunks (3072).
    // We process chunk-by-chunk to avoid a 3KB local buffer:
    // Instead, assemble each 1024-byte BLAKE3 chunk on the fly.

    // The message layout:
    //   [0..14]       = DOMAIN_LEAF (15 bytes)
    //   [15..2062]    = int64 data as little-endian bytes
    //
    // BLAKE3 chunk boundaries: [0..1023], [1024..2047], [2048..2062]

    int n_blake_chunks = (msg_len + 1023) / 1024;

    if (n_blake_chunks <= 1) {
        // Short message: single-chunk BLAKE3 (use blake3_hash_short path)
        uint8_t buf[1024];
        for (int i = 0; i < 1024; i++) buf[i] = 0;
        for (int i = 0; i < DOMAIN_LEAF_LEN; i++) buf[i] = DOMAIN_LEAF[i];
        // Copy int64 data as little-endian bytes
        const uint8_t* raw = (const uint8_t*)(data + elem_start);
        for (int i = 0; i < data_bytes; i++) buf[DOMAIN_LEAF_LEN + i] = raw[i];
        blake3_hash_short(buf, msg_len, leaf_hashes + tid * 32);
        return;
    }

    // Multi-chunk: hash the virtual prefixed message directly.  This is
    // byte-identical to the reference path but avoids a 1 KiB thread-local
    // buffer and two serial byte-copy passes per chunk.
    uint8_t chunk_cvs[4][32];  // max 4 chunk CVs (supports up to 4096-byte messages)
    const uint8_t* raw = (const uint8_t*)(data + elem_start);

    for (int c = 0; c < n_blake_chunks; c++) {
        blake3_hash_leaf_message_chunk(
            raw, data_bytes, data_bytes, msg_len, c, chunk_cvs[c]);
    }

    // Merge chunk CVs into root hash via binary parent hashing.
    // For 3 chunks: parent(cv0, cv1) → p01, then parent(p01, cv2, ROOT) → final
    // For 2 chunks: parent(cv0, cv1, ROOT) → final
    if (n_blake_chunks == 2) {
        blake3_parent_hash(chunk_cvs[0], chunk_cvs[1], true,
                          leaf_hashes + tid * 32);
    } else if (n_blake_chunks == 3) {
        uint8_t p01[32];
        blake3_parent_hash(chunk_cvs[0], chunk_cvs[1], false, p01);
        blake3_parent_hash(p01, chunk_cvs[2], true,
                          leaf_hashes + tid * 32);
    } else {
        // Fallback for 4 chunks (msg up to 4096 bytes)
        uint8_t p01[32], p23[32];
        blake3_parent_hash(chunk_cvs[0], chunk_cvs[1], false, p01);
        blake3_parent_hash(chunk_cvs[2], chunk_cvs[3], false, p23);
        blake3_parent_hash(p01, p23, true,
                          leaf_hashes + tid * 32);
    }
}

/**
 * Hash canonical zero-padded lanes directly from native runtime rows.
 *
 * Unlike ``blake3_activation_leaf_kernel``, row boundaries are explicit, so
 * the final partial lane of one row cannot read bytes from the next row.
 * This is the graph-capture path: it avoids materializing a full padded copy
 * of every activation while preserving the exact execution-anchor ABI.
 */
__global__ void blake3_activation_padded_row_leaf_kernel(
    uint8_t* __restrict__        leaf_hashes,
    const uint8_t* __restrict__  rows,
    int32_t                      row_count,
    int32_t                      row_bytes,
    int64_t                      row_stride_bytes,
    int32_t                      lane_bytes,
    int32_t                      lanes_per_row)
{
    const int64_t tid = (
        (int64_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    const int64_t total = (int64_t)row_count * lanes_per_row;
    if (tid >= total) return;

    const int32_t row = (int32_t)(tid / lanes_per_row);
    const int32_t lane = (int32_t)(tid % lanes_per_row);
    const int32_t lane_start = lane * lane_bytes;
    int32_t valid_bytes = row_bytes - lane_start;
    if (valid_bytes > lane_bytes) valid_bytes = lane_bytes;
    if (valid_bytes < 0) valid_bytes = 0;
    const uint8_t* raw = (
        rows + (int64_t)row * row_stride_bytes + lane_start
    );
    const int32_t message_len = DOMAIN_LEAF_LEN + lane_bytes;
    const int32_t chunk_count = (message_len + 1023) / 1024;
    uint8_t* output = leaf_hashes + tid * 32;

    if (chunk_count == 1) {
        uint8_t message[320];
        #pragma unroll
        for (int i = 0; i < 320; i++) message[i] = 0;
        #pragma unroll
        for (int i = 0; i < DOMAIN_LEAF_LEN; i++) {
            message[i] = DOMAIN_LEAF[i];
        }
        for (int i = 0; i < valid_bytes; i++) {
            message[DOMAIN_LEAF_LEN + i] = raw[i];
        }
        blake3_hash_short(message, message_len, output);
        return;
    }

    uint8_t chunk_cvs[4][32];
    for (int chunk = 0; chunk < chunk_count; chunk++) {
        blake3_hash_leaf_message_chunk(
            raw,
            valid_bytes,
            lane_bytes,
            message_len,
            chunk,
            chunk_cvs[chunk]
        );
    }
    if (chunk_count == 2) {
        blake3_parent_hash(
            chunk_cvs[0], chunk_cvs[1], true, output
        );
    } else if (chunk_count == 3) {
        uint8_t p01[32];
        blake3_parent_hash(
            chunk_cvs[0], chunk_cvs[1], false, p01
        );
        blake3_parent_hash(p01, chunk_cvs[2], true, output);
    } else {
        uint8_t p01[32], p23[32];
        blake3_parent_hash(
            chunk_cvs[0], chunk_cvs[1], false, p01
        );
        blake3_parent_hash(
            chunk_cvs[2], chunk_cvs[3], false, p23
        );
        blake3_parent_hash(p01, p23, true, output);
    }
}

/**
 * Hash 2-KiB canonical lanes with one four-thread subgroup per leaf.
 *
 * ``DOMAIN_LEAF || lane`` spans exactly three BLAKE3 chunks.  Those chunk
 * chaining values are independent, so subgroup lanes 0..2 compute them in
 * parallel and lane zero performs the two parent compressions.  The fourth
 * lane keeps subgroup geometry warp-aligned.  This changes only scheduling;
 * the resulting bytes are identical to
 * ``blake3_activation_padded_row_leaf_kernel``.
 */
__global__ void blake3_activation_padded_row_leaf_parallel_kernel(
    uint8_t* __restrict__        leaf_hashes,
    const uint8_t* __restrict__  rows,
    int32_t                      row_count,
    int32_t                      row_bytes,
    int64_t                      row_stride_bytes,
    int32_t                      lanes_per_row,
    const int32_t* __restrict__  retention_slots,
    int32_t                      retention_stage,
    uint8_t* __restrict__        retained_values,
    uint8_t* __restrict__        retained_hashes,
    int32_t                      retention_slot_count,
    int32_t                      retention_capacity,
    int32_t                      retention_max_lanes)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (
        (int32_t)threadIdx.x % subgroup_width
    );
    const int64_t leaf = (
        (int64_t)blockIdx.x * groups_per_block + group
    );
    const int64_t total = (int64_t)row_count * lanes_per_row;
    const bool active = leaf < total;
    const int32_t retention_slot = (
        active
        && retention_slots != nullptr
        && retention_stage >= 0
    ) ? retention_slots[retention_stage] : -1;
    const bool retain = (
        retention_slot >= 0
        && retention_slot < retention_slot_count
        && row_count <= retention_capacity
        && lanes_per_row <= retention_max_lanes
    );

    if (active && subgroup_lane < 3) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* raw = (
            rows + (int64_t)row * row_stride_bytes + lane_start
        );
        blake3_hash_leaf_message_chunk(
            raw,
            valid_bytes,
            2048,
            DOMAIN_LEAF_LEN + 2048,
            subgroup_lane,
            chunk_cvs[group][subgroup_lane]
        );
    }
    if (retain) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* raw = (
            rows + (int64_t)row * row_stride_bytes + lane_start
        );
        uint8_t* retained = (
            retained_values
            + (
                (
                    (int64_t)retention_slot * retention_capacity + row
                ) * retention_max_lanes + lane
            ) * 2048
        );
        for (int32_t offset = subgroup_lane; offset < 2048; offset += 4) {
            retained[offset] = offset < valid_bytes ? raw[offset] : 0;
        }
    }
    __syncthreads();

    if (active && subgroup_lane == 0) {
        uint8_t left[32];
        uint8_t output[32];
        blake3_parent_hash(
            chunk_cvs[group][0],
            chunk_cvs[group][1],
            false,
            left
        );
        blake3_parent_hash(
            left,
            chunk_cvs[group][2],
            true,
            output
        );
        #pragma unroll
        for (int32_t offset = 0; offset < 32; offset++) {
            leaf_hashes[leaf * 32 + offset] = output[offset];
        }
        if (retain) {
            const int32_t row = (int32_t)(leaf / lanes_per_row);
            const int32_t lane = (int32_t)(leaf % lanes_per_row);
            uint8_t* retained = (
                retained_hashes
                + (
                    (
                        (int64_t)retention_slot * retention_capacity + row
                    ) * retention_max_lanes + lane
                ) * 32
            );
            #pragma unroll
            for (int32_t offset = 0; offset < 32; offset++) {
                retained[offset] = output[offset];
            }
        }
    }
}

/** Hash logical rows directly from a paged K/V cache without gathering. */
__global__ void blake3_paged_cache_leaf_parallel_kernel(
    uint8_t* __restrict__ leaf_hashes,
    const uint8_t* __restrict__ cache,
    const int64_t* __restrict__ pages,
    int32_t logical_block_count,
    int32_t subblocks_per_page,
    int32_t physical_block_count,
    int32_t physical_rows,
    int32_t row_bytes,
    int64_t pair_stride_bytes,
    int64_t block_stride_bytes,
    int64_t row_stride_bytes,
    int32_t cache_index,
    int32_t component_count,
    int32_t pair_first,
    int32_t lanes_per_row)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (int32_t)threadIdx.x % subgroup_width;
    const int32_t logical_rows = subblocks_per_page * physical_rows;
    const int64_t rows_per_component = (
        (int64_t)logical_block_count * logical_rows
    );
    const int64_t total_rows = rows_per_component * component_count;
    const int64_t leaf = (int64_t)blockIdx.x * groups_per_block + group;
    const int64_t total_leaves = total_rows * lanes_per_row;
    const bool active = leaf < total_leaves;

    if (active && subgroup_lane < 3) {
        const int64_t row = leaf / lanes_per_row;
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t component = (int32_t)(row / rows_per_component);
        const int64_t component_row = row % rows_per_component;
        const int32_t logical_block = (int32_t)(
            component_row / logical_rows
        );
        const int32_t row_in_page = (int32_t)(
            component_row % logical_rows
        );
        const int32_t subblock = row_in_page / physical_rows;
        const int32_t row_in_block = row_in_page % physical_rows;
        const int64_t physical = pages[
            (int64_t)logical_block * subblocks_per_page + subblock
        ];
        const bool valid_physical = (
            physical >= 0 && physical < physical_block_count
        );
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = valid_physical ? row_bytes - lane_start : 0;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const int32_t selected_cache = cache_index < 0
            ? component : cache_index;
        const int64_t base_offset = pair_first
            ? (int64_t)selected_cache * pair_stride_bytes
                + physical * block_stride_bytes
                + (int64_t)row_in_block * row_stride_bytes
            : physical * block_stride_bytes
                + (int64_t)selected_cache * pair_stride_bytes
                + (int64_t)row_in_block * row_stride_bytes;
        const uint8_t* raw = cache + (valid_physical ? base_offset : 0)
            + lane_start;
        blake3_hash_leaf_message_chunk(
            raw,
            valid_bytes,
            2048,
            DOMAIN_LEAF_LEN + 2048,
            subgroup_lane,
            chunk_cvs[group][subgroup_lane]
        );
    }
    __syncthreads();

    if (active && subgroup_lane == 0) {
        uint8_t left[32];
        uint8_t output[32];
        blake3_parent_hash(
            chunk_cvs[group][0], chunk_cvs[group][1], false, left
        );
        blake3_parent_hash(left, chunk_cvs[group][2], true, output);
        #pragma unroll
        for (int32_t offset = 0; offset < 32; offset++) {
            leaf_hashes[leaf * 32 + offset] = output[offset];
        }
    }
}

/**
 * Hash a bounded stage-major runtime batch with one launch.
 *
 * ``rows`` is [stage, capacity, max_row_bytes].  Exact per-stage widths keep
 * canonical row padding out of the statement while allowing independent
 * stages to occupy the GPU concurrently during decode.
 */
__global__ void blake3_runtime_staged_leaf_parallel_kernel(
    uint8_t* __restrict__ hashes,
    const uint8_t* __restrict__ rows,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t max_row_bytes,
    int32_t max_lanes)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t stage = (int32_t)blockIdx.y;
    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (
        (int32_t)threadIdx.x % subgroup_width
    );
    const int64_t leaf = (
        (int64_t)blockIdx.x * groups_per_block + group
    );
    const int32_t row_bytes = row_widths[stage];
    const int32_t lane_bytes = lane_widths == nullptr
        ? 2048
        : lane_widths[stage];
    const int32_t lanes_per_row = (
        row_bytes + lane_bytes - 1
    ) / lane_bytes;
    const int64_t total = (int64_t)row_count * lanes_per_row;
    const bool geometry_valid = (
        row_bytes > 0
        && row_bytes <= max_row_bytes
        && (lane_bytes == 256 || lane_bytes == 2048)
        && lanes_per_row > 0
        && lanes_per_row <= max_lanes
    );
    const bool active = geometry_valid && leaf < total;

    if (active && lane_bytes == 256 && subgroup_lane == 0) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 256;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 256) valid_bytes = 256;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* raw = (
            rows
            + (
                (int64_t)stage * capacity + row
            ) * max_row_bytes
            + lane_start
        );
        uint8_t message[320];
        #pragma unroll
        for (int32_t offset = 0; offset < 320; offset++) {
            message[offset] = 0;
        }
        #pragma unroll
        for (int32_t offset = 0; offset < DOMAIN_LEAF_LEN; offset++) {
            message[offset] = DOMAIN_LEAF[offset];
        }
        for (int32_t offset = 0; offset < valid_bytes; offset++) {
            message[DOMAIN_LEAF_LEN + offset] = raw[offset];
        }
        blake3_hash_short(
            message,
            DOMAIN_LEAF_LEN + 256,
            hashes
                + (
                    (
                        (int64_t)stage * capacity + row
                    ) * max_lanes + lane
                ) * 32
        );
    }

    if (active && lane_bytes == 2048 && subgroup_lane < 3) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* raw = (
            rows
            + (
                (int64_t)stage * capacity + row
            ) * max_row_bytes
            + lane_start
        );
        blake3_hash_leaf_message_chunk(
            raw,
            valid_bytes,
            2048,
            DOMAIN_LEAF_LEN + 2048,
            subgroup_lane,
            chunk_cvs[group][subgroup_lane]
        );
    }
    __syncthreads();

    if (active && lane_bytes == 2048 && subgroup_lane == 0) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        uint8_t left[32];
        blake3_parent_hash(
            chunk_cvs[group][0],
            chunk_cvs[group][1],
            false,
            left
        );
        uint8_t* output = (
            hashes
            + (
                (
                    (int64_t)stage * capacity + row
                ) * max_lanes + lane
            ) * 32
        );
        blake3_parent_hash(
            left,
            chunk_cvs[group][2],
            true,
            output
        );
    }
}

/**
 * Hash stage-major runtime tensors without per-layer staging launches.
 *
 * Source pointers are kernel parameters captured by CUDA graphs. Aggregate
 * stages read two BF16 tensors and reproduce the runtime BF16 addition while
 * hashing, so the decoder does not materialize a second aggregate tensor.
 */
__global__ void blake3_runtime_tensor_leaf_parallel_kernel(
    uint8_t* __restrict__ hashes,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    RuntimeTensorSources sources,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t max_lanes)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t stage = (int32_t)blockIdx.y;
    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (
        (int32_t)threadIdx.x % subgroup_width
    );
    const int64_t leaf = (
        (int64_t)blockIdx.x * groups_per_block + group
    );
    const int32_t row_bytes = row_widths[stage];
    const int32_t source_row_bytes = sources.row_bytes[stage];
    const int32_t lane_bytes = lane_widths[stage];
    const int32_t lanes_per_row = (
        row_bytes + lane_bytes - 1
    ) / lane_bytes;
    const int64_t total = (int64_t)row_count * lanes_per_row;
    const bool geometry_valid = (
        row_bytes > 0
        && row_bytes == source_row_bytes
        && (lane_bytes == 256 || lane_bytes == 2048)
        && lanes_per_row > 0
        && lanes_per_row <= max_lanes
    );
    const bool active = geometry_valid && leaf < total;

    if (active && lane_bytes == 256 && subgroup_lane == 0) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 256;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 256) valid_bytes = 256;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* first = (
            sources.first[stage]
            + (int64_t)row * source_row_bytes
            + lane_start
        );
        const uint8_t* second = (
            sources.second[stage]
            + (int64_t)row * source_row_bytes
            + lane_start
        );
        const bool combine_bfloat16 = sources.combine_bfloat16[stage] != 0;
        uint8_t message[320];
        #pragma unroll
        for (int32_t offset = 0; offset < 320; offset++) {
            message[offset] = 0;
        }
        #pragma unroll
        for (int32_t offset = 0; offset < DOMAIN_LEAF_LEN; offset++) {
            message[offset] = DOMAIN_LEAF[offset];
        }
        for (int32_t offset = 0; offset < valid_bytes; offset++) {
            message[DOMAIN_LEAF_LEN + offset] = load_runtime_tensor_byte(
                first,
                second,
                combine_bfloat16,
                offset
            );
        }
        blake3_hash_short(
            message,
            DOMAIN_LEAF_LEN + 256,
            hashes
                + (
                    (
                        (int64_t)stage * capacity + row
                    ) * max_lanes + lane
                ) * 32
        );
    }

    if (active && lane_bytes == 2048 && subgroup_lane < 3) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* first = (
            sources.first[stage]
            + (int64_t)row * source_row_bytes
            + lane_start
        );
        const uint8_t* second = (
            sources.second[stage]
            + (int64_t)row * source_row_bytes
            + lane_start
        );
        blake3_hash_runtime_tensor_leaf_chunk(
            first,
            second,
            sources.combine_bfloat16[stage] != 0,
            valid_bytes,
            2048,
            DOMAIN_LEAF_LEN + 2048,
            subgroup_lane,
            chunk_cvs[group][subgroup_lane]
        );
    }
    __syncthreads();

    if (active && lane_bytes == 2048 && subgroup_lane == 0) {
        const int32_t row = (int32_t)(leaf / lanes_per_row);
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        uint8_t left[32];
        blake3_parent_hash(
            chunk_cvs[group][0],
            chunk_cvs[group][1],
            false,
            left
        );
        uint8_t* output = (
            hashes
            + (
                (
                    (int64_t)stage * capacity + row
                ) * max_lanes + lane
            ) * 32
        );
        blake3_parent_hash(
            left,
            chunk_cvs[group][2],
            true,
            output
        );
    }
}

/**
 * Hash selected opaque-cache rows only when an eager checkpoint is due.
 *
 * Unlike the graph-history kernel below, work is distributed by leaf across
 * the complete active request batch. A 2-D source map permits cache groups
 * to use different scheduler slots for different GDN stages without first
 * gathering the multi-megabyte recurrent rows.
 */
__global__ void blake3_runtime_tensor_indexed_leaf_parallel_kernel(
    uint8_t* __restrict__ hashes,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    const int32_t* __restrict__ source_indices,
    RuntimeTensorSources sources,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t max_lanes)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t stage = (int32_t)blockIdx.y;
    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (
        (int32_t)threadIdx.x % subgroup_width
    );
    const int64_t leaf = (
        (int64_t)blockIdx.x * groups_per_block + group
    );
    const int32_t row_bytes = row_widths[stage];
    const int32_t source_row_bytes = sources.row_bytes[stage];
    const int32_t lane_bytes = lane_widths[stage];
    const int32_t lanes_per_row = (
        row_bytes + lane_bytes - 1
    ) / lane_bytes;
    const int64_t total = (int64_t)row_count * lanes_per_row;
    const int32_t row = leaf < total
        ? (int32_t)(leaf / lanes_per_row)
        : 0;
    const int32_t source_row = leaf < total
        ? source_indices[(int64_t)stage * row_count + row]
        : -1;
    const bool geometry_valid = (
        row_bytes > 0
        && row_bytes == source_row_bytes
        && (lane_bytes == 256 || lane_bytes == 2048)
        && lanes_per_row > 0
        && lanes_per_row <= max_lanes
    );
    const bool active = (
        geometry_valid
        && leaf < total
        && source_row >= 0
        && source_row < sources.source_rows[stage]
    );

    if (active && lane_bytes == 256 && subgroup_lane == 0) {
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 256;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 256) valid_bytes = 256;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* first = (
            sources.first[stage]
            + (int64_t)source_row
                * sources.source_row_stride_bytes[stage]
            + lane_start
        );
        uint8_t message[320];
        #pragma unroll
        for (int32_t offset = 0; offset < 320; offset++) {
            message[offset] = 0;
        }
        #pragma unroll
        for (int32_t offset = 0; offset < DOMAIN_LEAF_LEN; offset++) {
            message[offset] = DOMAIN_LEAF[offset];
        }
        for (int32_t offset = 0; offset < valid_bytes; offset++) {
            message[DOMAIN_LEAF_LEN + offset] = first[offset];
        }
        blake3_hash_short(
            message,
            DOMAIN_LEAF_LEN + 256,
            hashes
                + (
                    (
                        (int64_t)stage * capacity + row
                    ) * max_lanes + lane
                ) * 32
        );
    }

    if (active && lane_bytes == 2048 && subgroup_lane < 3) {
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        const int32_t lane_start = lane * 2048;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > 2048) valid_bytes = 2048;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* first = (
            sources.first[stage]
            + (int64_t)source_row
                * sources.source_row_stride_bytes[stage]
            + lane_start
        );
        blake3_hash_runtime_tensor_leaf_chunk(
            first,
            first,
            false,
            valid_bytes,
            2048,
            DOMAIN_LEAF_LEN + 2048,
            subgroup_lane,
            chunk_cvs[group][subgroup_lane]
        );
    }
    __syncthreads();

    if (active && lane_bytes == 2048 && subgroup_lane == 0) {
        const int32_t lane = (int32_t)(leaf % lanes_per_row);
        uint8_t left[32];
        blake3_parent_hash(
            chunk_cvs[group][0],
            chunk_cvs[group][1],
            false,
            left
        );
        uint8_t* output = (
            hashes
            + (
                (
                    (int64_t)stage * capacity + row
                ) * max_lanes + lane
            ) * 32
        );
        blake3_parent_hash(
            left,
            chunk_cvs[group][2],
            true,
            output
        );
    }
}

__global__ void blake3_runtime_tensor_zero_invalid_indexed_roots_kernel(
    uint8_t* __restrict__ destination,
    const int32_t* __restrict__ source_indices,
    RuntimeTensorSources sources,
    int32_t stage_count,
    int32_t row_count,
    int64_t stage_stride,
    int64_t row_stride)
{
    const int64_t item = (
        (int64_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    const int64_t total = (int64_t)stage_count * row_count;
    if (item >= total) return;
    const int32_t stage = (int32_t)(item / row_count);
    const int32_t row = (int32_t)(item % row_count);
    const int32_t source_row = source_indices[item];
    if (source_row >= 0 && source_row < sources.source_rows[stage]) return;
    uint8_t* output = (
        destination + (int64_t)stage * stage_stride + row * row_stride
    );
    #pragma unroll
    for (int32_t byte = 0; byte < 32; byte++) output[byte] = 0;
}

/**
 * Hash indexed opaque cache rows into graph-resident history scratch.
 *
 * One block owns one logical scheduler row for one stage. Destination rows
 * with the -1 sentinel return before reading the cache, so ordinary decode
 * steps pay only the fixed launch/branch cost. Active checkpoints read the
 * scheduler-selected cache slot directly and never materialize a gathered
 * recurrent-state tensor.
 */
__global__ void blake3_runtime_tensor_indexed_leaf_kernel(
    uint8_t* __restrict__ hashes,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    const int64_t* __restrict__ destination_indices,
    RuntimeTensorSources sources,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t max_lanes)
{
    constexpr int32_t subgroup_width = 4;
    constexpr int32_t groups_per_block = BLAKE3_THREADS / subgroup_width;
    __shared__ uint8_t chunk_cvs[groups_per_block][3][32];

    const int32_t row = (int32_t)blockIdx.x;
    const int32_t stage = (int32_t)blockIdx.y;
    if (row >= row_count || stage >= stage_count) return;
    const int64_t destination_row = destination_indices[row];
    if (destination_row < 0) return;
    const int32_t source_row = sources.source_indices[stage][row];
    if (source_row < 0 || source_row >= sources.source_rows[stage]) return;

    const int32_t row_bytes = row_widths[stage];
    const int32_t source_row_bytes = sources.row_bytes[stage];
    const int32_t lane_bytes = lane_widths[stage];
    const int32_t lanes_per_row = (
        row_bytes + lane_bytes - 1
    ) / lane_bytes;
    if (
        row_bytes <= 0
        || row_bytes != source_row_bytes
        || (lane_bytes != 256 && lane_bytes != 2048)
        || lanes_per_row <= 0
        || lanes_per_row > max_lanes
    ) return;

    const int32_t group = (int32_t)threadIdx.x / subgroup_width;
    const int32_t subgroup_lane = (
        (int32_t)threadIdx.x % subgroup_width
    );
    for (
        int32_t lane_base = 0;
        lane_base < lanes_per_row;
        lane_base += groups_per_block
    ) {
        const int32_t lane = lane_base + group;
        const bool active = lane < lanes_per_row;
        const int32_t lane_start = active ? lane * lane_bytes : 0;
        int32_t valid_bytes = row_bytes - lane_start;
        if (valid_bytes > lane_bytes) valid_bytes = lane_bytes;
        if (valid_bytes < 0) valid_bytes = 0;
        const uint8_t* first = (
            sources.first[stage]
            + (int64_t)source_row
                * sources.source_row_stride_bytes[stage]
            + lane_start
        );
        const uint8_t* second = (
            sources.second[stage]
            + (int64_t)source_row
                * sources.source_row_stride_bytes[stage]
            + lane_start
        );

        if (active && lane_bytes == 256 && subgroup_lane == 0) {
            const bool combine_bfloat16 = (
                sources.combine_bfloat16[stage] != 0
            );
            uint8_t message[320];
            #pragma unroll
            for (int32_t offset = 0; offset < 320; offset++) {
                message[offset] = 0;
            }
            #pragma unroll
            for (
                int32_t offset = 0;
                offset < DOMAIN_LEAF_LEN;
                offset++
            ) {
                message[offset] = DOMAIN_LEAF[offset];
            }
            for (int32_t offset = 0; offset < valid_bytes; offset++) {
                message[DOMAIN_LEAF_LEN + offset] = load_runtime_tensor_byte(
                    first,
                    second,
                    combine_bfloat16,
                    offset
                );
            }
            blake3_hash_short(
                message,
                DOMAIN_LEAF_LEN + 256,
                hashes
                    + (
                        (
                            (int64_t)stage * capacity + row
                        ) * max_lanes + lane
                    ) * 32
            );
        }

        if (active && lane_bytes == 2048 && subgroup_lane < 3) {
            blake3_hash_runtime_tensor_leaf_chunk(
                first,
                second,
                sources.combine_bfloat16[stage] != 0,
                valid_bytes,
                2048,
                DOMAIN_LEAF_LEN + 2048,
                subgroup_lane,
                chunk_cvs[group][subgroup_lane]
            );
        }
        __syncthreads();

        if (active && lane_bytes == 2048 && subgroup_lane == 0) {
            uint8_t left[32];
            blake3_parent_hash(
                chunk_cvs[group][0],
                chunk_cvs[group][1],
                false,
                left
            );
            uint8_t* output = (
                hashes
                + (
                    (
                        (int64_t)stage * capacity + row
                    ) * max_lanes + lane
                ) * 32
            );
            blake3_parent_hash(
                left,
                chunk_cvs[group][2],
                true,
                output
            );
        }
        __syncthreads();
    }
}

/**
 * Reduce one level of independent row-local Merkle trees.
 *
 * ``input`` is laid out [row_count, input_width, 32].  Odd final nodes are
 * duplicated, matching ``MerkleTree`` and the Python execution-anchor
 * reference exactly.
 */
__device__ __forceinline__ void blake3_merkle_node_hash(
    const uint8_t* left,
    const uint8_t* right,
    uint8_t* output)
{
    uint8_t message[DOMAIN_NODE_LEN + 64];
    #pragma unroll
    for (int i = 0; i < DOMAIN_NODE_LEN; i++) {
        message[i] = DOMAIN_NODE[i];
    }
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        message[DOMAIN_NODE_LEN + i] = left[i];
        message[DOMAIN_NODE_LEN + 32 + i] = right[i];
    }
    blake3_hash_short(message, DOMAIN_NODE_LEN + 64, output);
}

__global__ void blake3_row_merkle_level_kernel(
    uint8_t* __restrict__ output,
    const uint8_t* __restrict__ input,
    int32_t row_count,
    int32_t input_width,
    int32_t output_width)
{
    const int64_t tid = (
        (int64_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    const int64_t total = (int64_t)row_count * output_width;
    if (tid >= total) return;

    const int32_t row = (int32_t)(tid / output_width);
    const int32_t column = (int32_t)(tid % output_width);
    const int32_t left_column = column * 2;
    const int32_t right_column = (
        left_column + 1 < input_width
        ? left_column + 1
        : left_column
    );
    const uint8_t* left = (
        input + ((int64_t)row * input_width + left_column) * 32
    );
    const uint8_t* right = (
        input + ((int64_t)row * input_width + right_column) * 32
    );

    blake3_merkle_node_hash(left, right, output + tid * 32);
}

__global__ void blake3_runtime_staged_row_reduce_kernel(
    uint8_t* __restrict__ destination,
    uint8_t* __restrict__ hashes,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    const int64_t* __restrict__ destination_indices,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t destination_rows,
    int64_t destination_stage_stride,
    int64_t destination_row_stride,
    int32_t max_lanes)
{
    const int64_t item = (
        (int64_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    const int64_t total = (int64_t)stage_count * row_count;
    if (item >= total) return;
    const int32_t stage = (int32_t)(item / row_count);
    const int32_t row = (int32_t)(item % row_count);
    const int64_t destination_row = destination_indices == nullptr
        ? (int64_t)row
        : destination_indices[row];
    if (destination_row < 0 || destination_row >= destination_rows) return;
    uint8_t* row_hashes = (
        hashes
        + (
            (
                (int64_t)stage * capacity + row
            ) * max_lanes
        ) * 32
    );
    const int32_t row_bytes = row_widths[stage];
    const int32_t lane_bytes = lane_widths == nullptr
        ? 2048
        : lane_widths[stage];
    int32_t width = (row_bytes + lane_bytes - 1) / lane_bytes;
    if (
        row_bytes <= 0
        || (lane_bytes != 256 && lane_bytes != 2048)
        || width <= 0
        || width > max_lanes
    ) {
        uint8_t* invalid_output = (
            destination
            + (int64_t)stage * destination_stage_stride
            + destination_row * destination_row_stride
        );
        #pragma unroll
        for (int32_t byte = 0; byte < 32; byte++) {
            invalid_output[byte] = 0;
        }
        return;
    }
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        for (int32_t column = 0; column < next_width; column++) {
            const int32_t left_column = column * 2;
            const int32_t right_column = (
                left_column + 1 < width
                ? left_column + 1
                : left_column
            );
            blake3_merkle_node_hash(
                row_hashes + (int64_t)left_column * 32,
                row_hashes + (int64_t)right_column * 32,
                row_hashes + (int64_t)column * 32
            );
        }
        width = next_width;
    }
    uint8_t* output = (
        destination
        + (int64_t)stage * destination_stage_stride
        + destination_row * destination_row_stride
    );
    #pragma unroll
    for (int32_t byte = 0; byte < 32; byte++) {
        output[byte] = row_hashes[byte];
    }
}

/**
 * Reduce indexed row trees in parallel without expanding global scratch.
 *
 * The indexed GDN path has up to 1,024 canonical 2-KiB leaves per recurrent
 * row.  The generic reducer above deliberately assigns one thread per row,
 * which is appropriate for narrow activation rows but serializes all 1,023
 * parent hashes for this geometry.  One block here owns a stage/row pair and
 * ping-pongs levels between two shared-memory spans.  Invalid scheduler slots
 * preserve the indexed ABI's all-zero sentinel.
 */
__global__ void blake3_runtime_indexed_row_reduce_parallel_kernel(
    uint8_t* __restrict__ destination,
    const uint8_t* __restrict__ hashes,
    const int32_t* __restrict__ row_widths,
    const int32_t* __restrict__ lane_widths,
    const int32_t* __restrict__ source_indices,
    const int64_t* __restrict__ destination_indices,
    RuntimeTensorSources sources,
    int32_t stage_count,
    int32_t row_count,
    int32_t capacity,
    int32_t destination_rows,
    int64_t destination_stage_stride,
    int64_t destination_row_stride,
    int32_t source_index_stage_stride,
    int32_t max_lanes)
{
    const int32_t row = (int32_t)blockIdx.x;
    const int32_t stage = (int32_t)blockIdx.y;
    if (row >= row_count || stage >= stage_count) return;

    const int64_t destination_row = destination_indices == nullptr
        ? (int64_t)row
        : destination_indices[row];
    if (
        destination_row < 0
        || destination_row >= destination_rows
    ) return;
    uint8_t* output = (
        destination
        + (int64_t)stage * destination_stage_stride
        + destination_row * destination_row_stride
    );
    const int32_t row_bytes = row_widths[stage];
    const int32_t lane_bytes = lane_widths[stage];
    const int32_t width = (
        lane_bytes > 0 ? (row_bytes + lane_bytes - 1) / lane_bytes : 0
    );
    const int32_t source_row = source_indices == nullptr
        ? sources.source_indices[stage][row]
        : source_indices[
            (int64_t)stage * source_index_stage_stride + row
        ];
    const bool valid = (
        row_bytes > 0
        && (lane_bytes == 256 || lane_bytes == 2048)
        && width > 0
        && width <= max_lanes
        && source_row >= 0
        && source_row < sources.source_rows[stage]
    );
    if (!valid) {
        for (
            int32_t byte = (int32_t)threadIdx.x;
            byte < 32;
            byte += (int32_t)blockDim.x
        ) {
            output[byte] = 0;
        }
        return;
    }

    extern __shared__ uint8_t shared_hashes[];
    uint8_t* first = shared_hashes;
    uint8_t* second = shared_hashes + (int64_t)max_lanes * 32;
    const uint8_t* global_row = (
        hashes
        + (
            ((int64_t)stage * capacity + row) * max_lanes
        ) * 32
    );
    for (
        int64_t byte = (int64_t)threadIdx.x;
        byte < (int64_t)width * 32;
        byte += (int32_t)blockDim.x
    ) {
        first[byte] = global_row[byte];
    }
    __syncthreads();

    int32_t current_width = width;
    bool current_is_first = true;
    while (current_width > 1) {
        const int32_t next_width = (current_width + 1) / 2;
        const uint8_t* level_input = current_is_first ? first : second;
        uint8_t* level_output = current_is_first ? second : first;
        for (
            int32_t column = (int32_t)threadIdx.x;
            column < next_width;
            column += (int32_t)blockDim.x
        ) {
            const int32_t left_column = column * 2;
            const int32_t right_column = (
                left_column + 1 < current_width
                ? left_column + 1
                : left_column
            );
            blake3_merkle_node_hash(
                level_input + (int64_t)left_column * 32,
                level_input + (int64_t)right_column * 32,
                level_output + (int64_t)column * 32
            );
        }
        __syncthreads();
        current_width = next_width;
        current_is_first = !current_is_first;
    }
    const uint8_t* root = current_is_first ? first : second;
    for (
        int32_t byte = (int32_t)threadIdx.x;
        byte < 32;
        byte += (int32_t)blockDim.x
    ) {
        output[byte] = root[byte];
    }
}

/**
 * Reduce variable-width activation rows in place.
 *
 * Every thread owns one row.  The 2-KiB leaf hashes for all stages were
 * produced by one large launch, while ``stage_chunk_offsets`` and
 * ``chunks_per_row`` preserve each stage's exact row-tree geometry.  Writing
 * a level into the beginning of its own input interval is safe because a
 * thread consumes columns 2*i and 2*i+1 before writing column i.
 */
__global__ void blake3_heterogeneous_row_reduce_kernel(
    uint8_t* __restrict__ hashes,
    const int64_t* __restrict__ stage_chunk_offsets,
    const int32_t* __restrict__ chunks_per_row,
    int32_t stage_count,
    int32_t rows_per_stage)
{
    const int64_t row_index = (
        (int64_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    const int64_t total_rows = (int64_t)stage_count * rows_per_stage;
    if (row_index >= total_rows) return;

    const int32_t stage = (int32_t)(row_index / rows_per_stage);
    const int32_t row = (int32_t)(row_index % rows_per_stage);
    const int32_t row_width = chunks_per_row[stage];
    const int64_t row_start = (
        stage_chunk_offsets[stage] + (int64_t)row * row_width
    );

    int32_t width = row_width;
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        for (int32_t column = 0; column < next_width; column++) {
            const int32_t left_column = column * 2;
            const int32_t right_column = (
                left_column + 1 < width
                ? left_column + 1
                : left_column
            );
            const uint8_t* left = (
                hashes + (row_start + left_column) * 32
            );
            const uint8_t* right = (
                hashes + (row_start + right_column) * 32
            );
            blake3_merkle_node_hash(
                left,
                right,
                hashes + (row_start + column) * 32
            );
        }
        width = next_width;
    }
}

/**
 * Append variable-width row roots to one frontier per runtime stage.
 */
__global__ void blake3_heterogeneous_streaming_peaks_kernel(
    uint8_t* __restrict__ peaks,
    uint8_t* __restrict__ deferred,
    const uint8_t* __restrict__ prior_deferred,
    const uint8_t* __restrict__ hashes,
    const int64_t* __restrict__ stage_chunk_offsets,
    const int32_t* __restrict__ chunks_per_row,
    int32_t stage_count,
    int32_t rows_per_stage,
    uint32_t base_count)
{
    const int32_t stage = (
        (int32_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    if (stage >= stage_count) return;

    const int32_t row_width = chunks_per_row[stage];
    const int64_t stage_start = stage_chunk_offsets[stage];
    uint32_t count = base_count;
    if (base_count > 0) {
        count -= 1;
        uint8_t carry[32];
        const uint8_t* source = prior_deferred + (int64_t)stage * 32;
        #pragma unroll
        for (int i = 0; i < 32; i++) carry[i] = source[i];

        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                carry[i] = merged[i];
                existing[i] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) destination[i] = carry[i];
        count += 1;
    }
    for (int32_t row = 0; row < rows_per_stage - 1; row++) {
        uint8_t carry[32];
        const uint8_t* source = (
            hashes + (stage_start + (int64_t)row * row_width) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) carry[i] = source[i];

        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                carry[i] = merged[i];
                existing[i] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) destination[i] = carry[i];
        count += 1;
    }

    const uint8_t* final_root = (
        hashes
        + (
            stage_start
            + (int64_t)(rows_per_stage - 1) * row_width
        ) * 32
    );
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        deferred[(int64_t)stage * 32 + i] = final_root[i];
    }
}

/**
 * Append row roots to one Merkle frontier per runtime stage.
 *
 * A thread owns one stage, so arbitrary scheduler chunk boundaries compose
 * exactly with the pre-existing frontier.  The expensive row hashing and
 * inner reductions remain massively parallel; this bounded serial tail
 * replaces hundreds of thousands of Python calls.
 */
__global__ void blake3_streaming_peaks_kernel(
    uint8_t* __restrict__ peaks,
    const uint8_t* __restrict__ row_roots,
    int32_t stage_count,
    int32_t row_stride,
    int32_t rows_to_append,
    uint32_t base_count)
{
    const int32_t stage = (
        (int32_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    if (stage >= stage_count) return;

    uint32_t count = base_count;
    for (int32_t row = 0; row < rows_to_append; row++) {
        uint8_t carry[32];
        const uint8_t* source = (
            row_roots
            + ((int64_t)stage * row_stride + row) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) carry[i] = source[i];

        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                carry[i] = merged[i];
                existing[i] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) destination[i] = carry[i];
        count += 1;
    }
}

/**
 * Append already-authenticated row roots to one frontier per stage.
 *
 * Runtime graph wrappers produce these roots directly from activation bytes.
 * This kernel only performs the outer context-tree accumulation, preserving
 * the same deferred-last-row convention as the raw-row reducers.
 */
__global__ void blake3_prehashed_streaming_peaks_kernel(
    uint8_t* __restrict__ peaks,
    uint8_t* __restrict__ deferred,
    const uint8_t* __restrict__ prior_deferred,
    const uint8_t* __restrict__ row_roots,
    int32_t stage_count,
    int32_t stage_stride,
    int32_t row_stride,
    int32_t rows_per_stage,
    uint32_t base_count)
{
    const int32_t stage = (
        (int32_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    if (stage >= stage_count) return;

    uint32_t count = base_count;
    if (base_count > 0) {
        count -= 1;
        uint8_t carry[32];
        const uint8_t* source = (
            prior_deferred + (int64_t)stage * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) carry[i] = source[i];
        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                carry[i] = merged[i];
                existing[i] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) destination[i] = carry[i];
        count += 1;
    }

    const int64_t stage_start = (
        (int64_t)stage * stage_stride * 32
    );
    for (int32_t row = 0; row < rows_per_stage - 1; row++) {
        uint8_t carry[32];
        const uint8_t* source = (
            row_roots
            + stage_start
            + (int64_t)row * row_stride * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) carry[i] = source[i];
        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                carry[i] = merged[i];
                existing[i] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int i = 0; i < 32; i++) destination[i] = carry[i];
        count += 1;
    }
    const uint8_t* final_source = (
        row_roots
        + stage_start
        + (int64_t)(rows_per_stage - 1) * row_stride * 32
    );
    uint8_t* final_destination = (
        deferred + (int64_t)stage * 32
    );
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        final_destination[i] = final_source[i];
    }
}

/** Update the normal deferred prefix and one sparse interval in one launch. */
__global__ void blake3_prehashed_streaming_peaks_ranges_kernel(
    uint8_t* __restrict__ peaks,
    uint8_t* __restrict__ deferred,
    const uint8_t* __restrict__ prior_deferred,
    uint8_t* __restrict__ range_hashes,
    int32_t* __restrict__ range_levels,
    int64_t* __restrict__ range_indices,
    int32_t* __restrict__ range_lengths,
    const uint8_t* __restrict__ row_roots,
    int32_t stage_count,
    int32_t rows_per_stage,
    uint32_t base_count,
    int32_t range_row_start,
    uint32_t range_base_count,
    int32_t capacity)
{
    const int32_t stage = (
        (int32_t)blockIdx.x * blockDim.x + threadIdx.x
    );
    if (stage >= stage_count) return;

    uint32_t count = base_count;
    if (base_count > 0) {
        count -= 1;
        uint8_t carry[32];
        const uint8_t* source = prior_deferred + (int64_t)stage * 32;
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) carry[byte] = source[byte];
        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int byte = 0; byte < 32; byte++) {
                carry[byte] = merged[byte];
                existing[byte] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) destination[byte] = carry[byte];
        count += 1;
    }

    const int64_t roots_base = (
        (int64_t)stage * rows_per_stage * 32
    );
    for (int32_t row = 0; row < rows_per_stage - 1; row++) {
        uint8_t carry[32];
        const uint8_t* source = row_roots + roots_base + (int64_t)row * 32;
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) carry[byte] = source[byte];
        int32_t level = 0;
        while ((count & (1u << level)) != 0u) {
            uint8_t merged[32];
            uint8_t* existing = (
                peaks + ((int64_t)stage * 32 + level) * 32
            );
            blake3_merkle_node_hash(existing, carry, merged);
            #pragma unroll
            for (int byte = 0; byte < 32; byte++) {
                carry[byte] = merged[byte];
                existing[byte] = 0;
            }
            level += 1;
        }
        uint8_t* destination = (
            peaks + ((int64_t)stage * 32 + level) * 32
        );
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) destination[byte] = carry[byte];
        count += 1;
    }
    const uint8_t* final_root = (
        row_roots + roots_base + (int64_t)(rows_per_stage - 1) * 32
    );
    #pragma unroll
    for (int byte = 0; byte < 32; byte++) {
        deferred[(int64_t)stage * 32 + byte] = final_root[byte];
    }

    int32_t length = range_lengths[stage];
    const int64_t metadata_base = (int64_t)stage * capacity;
    const int32_t range_rows = rows_per_stage - range_row_start;
    for (int32_t local_row = 0; local_row < range_rows; local_row++) {
        const int32_t row = range_row_start + local_row;
        uint8_t carry[32];
        const uint8_t* source = row_roots + roots_base + (int64_t)row * 32;
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) carry[byte] = source[byte];
        int32_t level = 0;
        int64_t index = (int64_t)range_base_count + local_row;
        while (length > 0) {
            const int64_t top = metadata_base + length - 1;
            const int32_t left_level = range_levels[top];
            const int64_t left_index = range_indices[top];
            if (
                left_level != level
                || (left_index & 1LL) != 0
                || index != left_index + 1
            ) {
                break;
            }
            uint8_t merged[32];
            blake3_merkle_node_hash(
                range_hashes + top * 32, carry, merged
            );
            #pragma unroll
            for (int byte = 0; byte < 32; byte++) carry[byte] = merged[byte];
            length -= 1;
            level += 1;
            index >>= 1;
        }
        if (length >= capacity) {
            range_lengths[stage] = capacity + 1;
            return;
        }
        const int64_t destination = metadata_base + length;
        range_levels[destination] = level;
        range_indices[destination] = index;
        #pragma unroll
        for (int byte = 0; byte < 32; byte++) {
            range_hashes[destination * 32 + byte] = carry[byte];
        }
        length += 1;
    }
    range_lengths[stage] = length;
}

// ======================================================================
//  Host wrapper
// ======================================================================

/**
 * GPU-accelerated BLAKE3 Merkle leaf hashing for int8 weight tensors.
 *
 * Computes hash_leaf(hash_flat_chunk(chunk_bytes)) for every chunk of the
 * flattened weight tensor.  Returns a CPU tensor of shape [num_chunks, 32]
 * (uint8) containing the leaf hashes, ready for MerkleTree construction.
 *
 * @param W            2D int8 tensor on CUDA [K, N] or 1D [K*N]
 * @param chunk_size   Elements per chunk (default 128)
 * @param device_index CUDA device index
 * @return             CPU uint8 tensor [num_chunks, 32]
 */
at::Tensor cuda_blake3_merkle_leaves(
    at::Tensor W,
    int32_t chunk_size,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);

    TORCH_CHECK(W.dtype() == at::kChar,
        "cuda_blake3_merkle_leaves: expected int8 tensor, got ", W.dtype());
    TORCH_CHECK(W.is_cuda(),
        "cuda_blake3_merkle_leaves: expected CUDA tensor");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    // Flatten to 1D contiguous
    auto W_flat = W.contiguous().view(-1);
    int64_t total_elements = W_flat.numel();
    int32_t num_chunks = (int32_t)((total_elements + chunk_size - 1) / chunk_size);

    // Allocate output on GPU
    auto options = at::TensorOptions().dtype(at::kByte).device(W.device());
    auto leaf_hashes_gpu = at::zeros({num_chunks, 32}, options);

    // Launch kernel
    int threads = BLAKE3_THREADS;
    int blocks  = (num_chunks + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    blake3_merkle_leaf_hashes_kernel<<<blocks, threads, 0, stream.stream()>>>(
        leaf_hashes_gpu.data_ptr<uint8_t>(),
        W_flat.data_ptr<int8_t>(),
        total_elements,
        chunk_size,
        num_chunks
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "blake3_merkle_leaf_hashes_kernel failed: ", cudaGetErrorString(err));

    // Transfer to CPU and return
    return leaf_hashes_gpu.cpu();
}


/**
 * GPU-accelerated BLAKE3 activation Merkle leaf hashing.
 *
 * Computes blake3(DOMAIN_LEAF + block_bytes) for every block of the
 * flattened int64 activation tensor.  Uses multi-chunk BLAKE3 for the
 * 2063-byte messages (15-byte domain + 2048-byte int64 block data).
 *
 * This matches the CPU activation commitment path:
 *   hash_leaf(block.tobytes())  where block = int64 tensor of block_size elements
 *
 * @param W            int64 CUDA tensor (flattened activations)
 * @param chunk_size   Elements per block (e.g. 256 → 2048 bytes)
 * @param device_index CUDA device index
 * @return             CPU uint8 tensor [num_blocks, 32]
 */
at::Tensor cuda_blake3_activation_merkle_leaves(
    at::Tensor W,
    int32_t chunk_size,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);

    TORCH_CHECK(W.dtype() == at::kLong,
        "cuda_blake3_activation_merkle_leaves: expected int64 tensor, got ", W.dtype());
    TORCH_CHECK(W.is_cuda(),
        "cuda_blake3_activation_merkle_leaves: expected CUDA tensor");
    TORCH_CHECK(chunk_size > 0 && chunk_size <= 256,
        "chunk_size must be in [1, 256] int64 elements");

    auto W_flat = W.contiguous().view(-1);
    int64_t total_elements = W_flat.numel();
    int32_t num_chunks = (int32_t)((total_elements + chunk_size - 1) / chunk_size);

    auto options = at::TensorOptions().dtype(at::kByte).device(W.device());
    auto leaf_hashes_gpu = at::zeros({num_chunks, 32}, options);

    int threads = BLAKE3_THREADS;
    int blocks  = (num_chunks + threads - 1) / threads;

    auto stream = at::cuda::getCurrentCUDAStream(device_index);

    blake3_activation_leaf_kernel<<<blocks, threads, 0, stream.stream()>>>(
        leaf_hashes_gpu.data_ptr<uint8_t>(),
        W_flat.data_ptr<int64_t>(),
        total_elements,
        chunk_size,
        num_chunks
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "blake3_activation_leaf_kernel failed: ", cudaGetErrorString(err));

    return leaf_hashes_gpu.cpu();
}


static at::Tensor activation_row_roots_device(
    at::Tensor packed,
    int32_t words_per_row,
    int32_t words_per_leaf,
    int device_index)
{
    const int32_t row_count = (int32_t)packed.size(0);
    int32_t width = words_per_row / words_per_leaf;
    const int64_t total_chunks = (int64_t)row_count * width;
    TORCH_CHECK(total_chunks <= 0x7fffffffLL,
        "activation row leaf count is outside the supported range");

    auto options = at::TensorOptions().dtype(at::kByte).device(packed.device());
    auto current = at::zeros({row_count, width, 32}, options);
    const int threads = BLAKE3_THREADS;
    int blocks = ((int32_t)total_chunks + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_activation_leaf_kernel<<<blocks, threads, 0, stream.stream()>>>(
        current.data_ptr<uint8_t>(),
        packed.data_ptr<int64_t>(),
        packed.numel(),
        words_per_leaf,
        (int32_t)total_chunks
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "blake3_activation_leaf_kernel failed: ", cudaGetErrorString(err));

    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        auto next = at::empty({row_count, next_width, 32}, options);
        const int64_t total_nodes = (int64_t)row_count * next_width;
        blocks = ((int32_t)total_nodes + threads - 1) / threads;
        blake3_row_merkle_level_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            next.data_ptr<uint8_t>(),
            current.data_ptr<uint8_t>(),
            row_count,
            width,
            next_width
        );
        err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess,
            "blake3_row_merkle_level_kernel failed: ",
            cudaGetErrorString(err));
        current = next;
        width = next_width;
    }
    return current.view({row_count, 32});
}


static at::Tensor validated_activation_rows(
    at::Tensor W,
    int32_t words_per_row,
    int32_t words_per_leaf,
    const char* operation)
{
    TORCH_CHECK(W.dtype() == at::kLong,
        operation, ": expected int64 tensor, got ", W.dtype());
    TORCH_CHECK(W.is_cuda(), operation, ": expected CUDA tensor");
    TORCH_CHECK(W.dim() == 2, operation, ": expected a 2D tensor");
    TORCH_CHECK(
        words_per_row > 0
        && (words_per_leaf == 32 || words_per_leaf == 256)
        && words_per_row % words_per_leaf == 0
        && W.size(1) == words_per_row,
        "words_per_row must match a supported words_per_leaf multiple"
    );
    TORCH_CHECK(W.size(0) > 0 && W.size(0) <= 0x7fffffffLL,
        "row count is outside the supported range");
    return W.contiguous();
}


/**
 * Hash fixed-width padded activation rows and reduce every row to one root.
 *
 * Each row is split into canonical 256-int64 / 2-KiB leaves.  The entire
 * inner row tree remains on device; only one 32-byte root per row crosses
 * to the CPU.
 */
at::Tensor cuda_blake3_activation_row_roots(
    at::Tensor W,
    int32_t words_per_row,
    int device_index,
    int32_t words_per_leaf)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    auto packed = validated_activation_rows(
        W,
        words_per_row,
        words_per_leaf,
        "cuda_blake3_activation_row_roots");
    return activation_row_roots_device(
        packed, words_per_row, words_per_leaf, device_index).cpu();
}

/**
 * Hash native FP16/BF16/FP32 runtime rows into a preallocated device buffer.
 *
 * The final partial lane is zero padded exactly like
 * ``execution_anchor_row_leaf_hash_v3``.  Nothing crosses to the CPU.
 */
void cuda_blake3_runtime_row_roots_into(
    at::Tensor destination,
    at::Tensor source,
    int32_t lane_bytes,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        source.is_cuda() && destination.is_cuda(),
        "runtime row roots require CUDA tensors"
    );
    TORCH_CHECK(
        source.device() == destination.device(),
        "runtime row roots require one CUDA device"
    );
    TORCH_CHECK(
        source.dim() == 2
        && source.stride(1) == 1
        && source.stride(0) >= source.size(1),
        "runtime row roots require dense, non-overlapping source rows"
    );
    TORCH_CHECK(
        source.scalar_type() == at::kHalf
        || source.scalar_type() == at::kBFloat16
        || source.scalar_type() == at::kFloat,
        "runtime row roots require FP16/BF16/FP32 source rows"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 2
        && destination.size(1) == 32
        && destination.is_contiguous(),
        "runtime row-root destination must be contiguous uint8 [N,32]"
    );
    TORCH_CHECK(
        lane_bytes == 256 || lane_bytes == 2048,
        "runtime row-root lane width is unsupported"
    );
    const int64_t row_count64 = source.size(0);
    const int64_t row_bytes64 = (
        source.size(1) * source.element_size()
    );
    const int64_t row_stride_bytes64 = (
        source.stride(0) * source.element_size()
    );
    TORCH_CHECK(
        row_count64 > 0
        && row_count64 <= 0x7fffffffLL
        && destination.size(0) >= row_count64,
        "runtime row-root destination is too small"
    );
    TORCH_CHECK(
        row_bytes64 > 0
        && row_bytes64 < (1LL << 24)
        && row_stride_bytes64 >= row_bytes64
        && row_stride_bytes64 < (1LL << 32),
        "runtime row width is outside the supported range"
    );
    const int32_t row_count = (int32_t)row_count64;
    const int32_t row_bytes = (int32_t)row_bytes64;
    const int32_t lanes_per_row = (
        row_bytes + lane_bytes - 1
    ) / lane_bytes;
    const int64_t total_leaves = (
        row_count64 * lanes_per_row
    );
    TORCH_CHECK(
        total_leaves <= 0x7fffffffLL,
        "runtime row leaf count is outside the supported range"
    );

    auto byte_options = at::TensorOptions()
        .dtype(at::kByte).device(source.device());
    auto current = at::empty(
        {row_count, lanes_per_row, 32},
        byte_options
    );
    const int threads = BLAKE3_THREADS;
    int blocks;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    if (lane_bytes == 2048) {
        const int32_t groups_per_block = threads / 4;
        blocks = (
            (int32_t)total_leaves + groups_per_block - 1
        ) / groups_per_block;
        blake3_activation_padded_row_leaf_parallel_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            current.data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(source.data_ptr()),
            row_count,
            row_bytes,
            row_stride_bytes64,
            lanes_per_row,
            nullptr,
            -1,
            nullptr,
            nullptr,
            0,
            0,
            0
        );
    } else {
        blocks = (
            (int32_t)total_leaves + threads - 1
        ) / threads;
        blake3_activation_padded_row_leaf_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            current.data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(source.data_ptr()),
            row_count,
            row_bytes,
            row_stride_bytes64,
            lane_bytes,
            lanes_per_row
        );
    }
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "runtime activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    int32_t width = lanes_per_row;
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        auto next = at::empty(
            {row_count, next_width, 32},
            byte_options
        );
        const int64_t total_nodes = (
            row_count64 * next_width
        );
        blocks = (
            (int32_t)total_nodes + threads - 1
        ) / threads;
        blake3_row_merkle_level_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            next.data_ptr<uint8_t>(),
            current.data_ptr<uint8_t>(),
            row_count,
            width,
            next_width
        );
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "runtime row reduction failed: ",
            cudaGetErrorString(err)
        );
        current = next;
        width = next_width;
    }
    destination.narrow(0, 0, row_count).copy_(
        current.view({row_count, 32})
    );
}

/** Hash and reduce complete logical pages directly from a paged K/V cache. */
void cuda_blake3_paged_cache_block_roots_into(
    at::Tensor destination,
    at::Tensor cache,
    at::Tensor pages,
    int32_t cache_index,
    bool pair_first,
    int32_t lane_bytes,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        destination.is_cuda() && cache.is_cuda() && pages.is_cuda()
        && destination.device() == cache.device()
        && pages.device() == cache.device(),
        "paged cache roots require one CUDA device"
    );
    TORCH_CHECK(
        cache.dim() == 5
        && (cache.scalar_type() == at::kHalf
            || cache.scalar_type() == at::kBFloat16
            || cache.scalar_type() == at::kFloat)
        && cache.stride(4) == 1
        && cache.stride(3) == cache.size(4),
        "paged cache roots require contiguous head rows"
    );
    TORCH_CHECK(
        pages.scalar_type() == at::kLong
        && pages.dim() == 2
        && pages.is_contiguous()
        && pages.size(0) > 0
        && pages.size(1) > 0,
        "paged cache root mapping is malformed"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 2
        && destination.size(1) == 32
        && destination.is_contiguous(),
        "paged cache root destination is malformed"
    );
    TORCH_CHECK(
        cache_index == -1 || cache_index == 0 || cache_index == 1,
        "paged cache component index is malformed"
    );
    TORCH_CHECK(
        lane_bytes == 2048,
        "paged cache roots require canonical 2-KiB lanes"
    );

    const int32_t logical_block_count = (int32_t)pages.size(0);
    const int32_t component_count = cache_index < 0 ? 2 : 1;
    const int32_t output_block_count = (
        logical_block_count * component_count
    );
    TORCH_CHECK(
        destination.size(0) == output_block_count,
        "paged cache root destination count is malformed"
    );
    const int32_t subblocks_per_page = (int32_t)pages.size(1);
    const int32_t physical_block_count = (int32_t)(
        pair_first ? cache.size(1) : cache.size(0)
    );
    const int32_t physical_rows = (int32_t)cache.size(2);
    const int64_t row_bytes64 = (
        cache.size(3) * cache.size(4) * cache.element_size()
    );
    const int64_t logical_rows64 = (
        (int64_t)subblocks_per_page * physical_rows
    );
    const int64_t total_rows64 = (
        (int64_t)output_block_count * logical_rows64
    );
    TORCH_CHECK(
        physical_block_count > 0
        && physical_rows > 0
        && row_bytes64 > 0
        && row_bytes64 < (1LL << 24)
        && logical_rows64 > 0
        && logical_rows64 <= 0x7fffffffLL
        && total_rows64 <= 0x7fffffffLL,
        "paged cache root geometry is unsupported"
    );
    const int32_t row_bytes = (int32_t)row_bytes64;
    const int32_t logical_rows = (int32_t)logical_rows64;
    const int32_t total_rows = (int32_t)total_rows64;
    const int32_t lanes_per_row = (row_bytes + lane_bytes - 1) / lane_bytes;
    const int64_t total_leaves = total_rows64 * lanes_per_row;
    TORCH_CHECK(
        total_leaves <= 0x7fffffffLL,
        "paged cache leaf count is unsupported"
    );

    auto byte_options = at::TensorOptions()
        .dtype(at::kByte).device(cache.device());
    auto current = at::empty(
        {total_rows, lanes_per_row, 32},
        byte_options
    );
    const int threads = BLAKE3_THREADS;
    const int32_t groups_per_block = threads / 4;
    int blocks = (
        (int32_t)total_leaves + groups_per_block - 1
    ) / groups_per_block;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    const int64_t element_bytes = cache.element_size();
    const int64_t pair_stride_bytes = cache.stride(pair_first ? 0 : 1)
        * element_bytes;
    const int64_t block_stride_bytes = cache.stride(pair_first ? 1 : 0)
        * element_bytes;
    const int64_t row_stride_bytes = cache.stride(2) * element_bytes;
    blake3_paged_cache_leaf_parallel_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        current.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(cache.data_ptr()),
        pages.data_ptr<int64_t>(),
        logical_block_count,
        subblocks_per_page,
        physical_block_count,
        physical_rows,
        row_bytes,
        pair_stride_bytes,
        block_stride_bytes,
        row_stride_bytes,
        cache_index,
        component_count,
        pair_first ? 1 : 0,
        lanes_per_row
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "paged cache leaf hashing failed: ",
        cudaGetErrorString(err)
    );

    int32_t width = lanes_per_row;
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        auto next = at::empty({total_rows, next_width, 32}, byte_options);
        const int64_t total_nodes = total_rows64 * next_width;
        blocks = ((int32_t)total_nodes + threads - 1) / threads;
        blake3_row_merkle_level_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            next.data_ptr<uint8_t>(),
            current.data_ptr<uint8_t>(),
            total_rows,
            width,
            next_width
        );
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "paged cache row reduction failed: ",
            cudaGetErrorString(err)
        );
        current = next;
        width = next_width;
    }

    current = current.view({output_block_count, logical_rows, 32});
    width = logical_rows;
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        auto next = at::empty(
            {output_block_count, next_width, 32},
            byte_options
        );
        const int64_t total_nodes = (
            (int64_t)output_block_count * next_width
        );
        blocks = ((int32_t)total_nodes + threads - 1) / threads;
        blake3_row_merkle_level_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            next.data_ptr<uint8_t>(),
            current.data_ptr<uint8_t>(),
            output_block_count,
            width,
            next_width
        );
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "paged cache block reduction failed: ",
            cudaGetErrorString(err)
        );
        current = next;
        width = next_width;
    }
    destination.copy_(current.view({output_block_count, 32}));
}

/**
 * Hash runtime rows and retain one post-nonce selected stage in a shared
 * bounded arena.  The selector remains device-resident so CUDA graphs keep
 * static addresses while ordinary serving performs root-only capture.
 */
void cuda_blake3_runtime_row_roots_retain_into(
    at::Tensor destination,
    at::Tensor source,
    at::Tensor retained_values,
    at::Tensor retained_hashes,
    at::Tensor retention_slots,
    int32_t retention_stage,
    int32_t lane_bytes,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        source.is_cuda()
        && destination.is_cuda()
        && retained_values.is_cuda()
        && retained_hashes.is_cuda()
        && retention_slots.is_cuda(),
        "runtime retained row roots require CUDA tensors"
    );
    TORCH_CHECK(
        source.device() == destination.device()
        && source.device() == retained_values.device()
        && source.device() == retained_hashes.device()
        && source.device() == retention_slots.device(),
        "runtime retained row roots require one CUDA device"
    );
    TORCH_CHECK(
        source.dim() == 2
        && source.stride(1) == 1
        && source.stride(0) >= source.size(1)
        && (
            source.scalar_type() == at::kHalf
            || source.scalar_type() == at::kBFloat16
            || source.scalar_type() == at::kFloat
        ),
        "runtime retained row roots require dense floating rows"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 2
        && destination.size(1) == 32
        && destination.is_contiguous(),
        "runtime retained root destination is malformed"
    );
    TORCH_CHECK(
        retained_values.scalar_type() == at::kByte
        && retained_values.dim() == 4
        && retained_values.size(3) == 2048
        && retained_values.is_contiguous()
        && retained_hashes.scalar_type() == at::kByte
        && retained_hashes.dim() == 4
        && retained_hashes.size(3) == 32
        && retained_hashes.is_contiguous()
        && retained_values.size(0) == retained_hashes.size(0)
        && retained_values.size(1) == retained_hashes.size(1)
        && retained_values.size(2) == retained_hashes.size(2),
        "runtime retained lane arena is malformed"
    );
    TORCH_CHECK(
        retention_slots.scalar_type() == at::kInt
        && retention_slots.dim() == 1
        && retention_slots.is_contiguous()
        && retention_stage >= 0
        && retention_stage < retention_slots.size(0),
        "runtime retained stage selector is malformed"
    );
    TORCH_CHECK(lane_bytes == 2048,
        "runtime retained lanes require the canonical 2-KiB lane");

    const int64_t row_count64 = source.size(0);
    const int64_t row_bytes64 = source.size(1) * source.element_size();
    const int64_t row_stride_bytes64 = (
        source.stride(0) * source.element_size()
    );
    const int64_t lanes_per_row64 = (row_bytes64 + 2047) / 2048;
    TORCH_CHECK(
        row_count64 > 0
        && row_count64 <= 0x7fffffffLL
        && row_count64 <= destination.size(0)
        && row_count64 <= retained_values.size(1)
        && row_bytes64 > 0
        && row_bytes64 < (1LL << 24)
        && row_stride_bytes64 >= row_bytes64
        && row_stride_bytes64 < (1LL << 32)
        && lanes_per_row64 > 0
        && lanes_per_row64 <= retained_values.size(2),
        "runtime retained row geometry is outside the bounded arena"
    );

    const int32_t row_count = (int32_t)row_count64;
    const int32_t row_bytes = (int32_t)row_bytes64;
    const int32_t lanes_per_row = (int32_t)lanes_per_row64;
    const int64_t total_leaves = row_count64 * lanes_per_row64;
    TORCH_CHECK(total_leaves <= 0x7fffffffLL,
        "runtime retained leaf count is outside the supported range");

    auto byte_options = at::TensorOptions()
        .dtype(at::kByte).device(source.device());
    auto current = at::empty(
        {row_count, lanes_per_row, 32},
        byte_options
    );
    const int threads = BLAKE3_THREADS;
    const int32_t groups_per_block = threads / 4;
    int blocks = (
        (int32_t)total_leaves + groups_per_block - 1
    ) / groups_per_block;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_activation_padded_row_leaf_parallel_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        current.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(source.data_ptr()),
        row_count,
        row_bytes,
        row_stride_bytes64,
        lanes_per_row,
        retention_slots.data_ptr<int32_t>(),
        retention_stage,
        retained_values.data_ptr<uint8_t>(),
        retained_hashes.data_ptr<uint8_t>(),
        (int32_t)retained_values.size(0),
        (int32_t)retained_values.size(1),
        (int32_t)retained_values.size(2)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "runtime retained activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    int32_t width = lanes_per_row;
    while (width > 1) {
        const int32_t next_width = (width + 1) / 2;
        auto next = at::empty(
            {row_count, next_width, 32},
            byte_options
        );
        const int64_t total_nodes = row_count64 * next_width;
        blocks = ((int32_t)total_nodes + threads - 1) / threads;
        blake3_row_merkle_level_kernel<<<
            blocks, threads, 0, stream.stream()
        >>>(
            next.data_ptr<uint8_t>(),
            current.data_ptr<uint8_t>(),
            row_count,
            width,
            next_width
        );
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "runtime retained row reduction failed: ",
            cudaGetErrorString(err)
        );
        current = next;
        width = next_width;
    }
    destination.narrow(0, 0, row_count).copy_(
        current.view({row_count, 32})
    );
}

static void cuda_blake3_runtime_staged_row_roots_into_impl(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    at::Tensor destination_indices,
    int32_t row_count,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        destination.is_cuda()
        && scratch.is_cuda()
        && staging.is_cuda()
        && row_widths.is_cuda()
        && destination.device() == staging.device()
        && scratch.device() == staging.device()
        && row_widths.device() == staging.device()
        && (
            !lane_widths.defined()
            || (
                lane_widths.is_cuda()
                && lane_widths.device() == staging.device()
            )
        )
        && (
            !destination_indices.defined()
            || (
                destination_indices.is_cuda()
                && destination_indices.device() == staging.device()
            )
        ),
        "staged runtime row roots require one CUDA device"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 3
        && destination.size(2) == 32
        && destination.stride(2) == 1
        && destination.stride(0) > 0
        && destination.stride(1) > 0,
        "staged runtime root destination is malformed"
    );
    TORCH_CHECK(
        scratch.scalar_type() == at::kByte
        && scratch.dim() == 4
        && scratch.size(3) == 32
        && scratch.is_contiguous(),
        "staged runtime root scratch is malformed"
    );
    TORCH_CHECK(
        (
            staging.scalar_type() == at::kHalf
            || staging.scalar_type() == at::kBFloat16
            || staging.scalar_type() == at::kFloat
        )
        && staging.dim() == 3
        && staging.is_contiguous(),
        "staged runtime rows require contiguous FP16/BF16/FP32 storage"
    );
    TORCH_CHECK(
        row_widths.scalar_type() == at::kInt
        && row_widths.dim() == 1
        && row_widths.is_contiguous(),
        "staged runtime row widths are malformed"
    );
    TORCH_CHECK(
        !destination_indices.defined()
        || (
            destination_indices.scalar_type() == at::kLong
            && destination_indices.dim() == 1
            && destination_indices.is_contiguous()
        ),
        "staged runtime destination indices are malformed"
    );
    TORCH_CHECK(
        !lane_widths.defined()
        || (
            lane_widths.scalar_type() == at::kInt
            && lane_widths.dim() == 1
            && lane_widths.is_contiguous()
        ),
        "staged runtime lane widths are malformed"
    );
    const int64_t stage_count64 = staging.size(0);
    const int64_t capacity64 = staging.size(1);
    const int64_t max_row_bytes64 = (
        staging.size(2) * staging.element_size()
    );
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= 0x7fffffffLL
        && capacity64 > 0
        && capacity64 <= 0x7fffffffLL
        && row_count > 0
        && row_count <= capacity64
        && max_row_bytes64 > 0
        && max_row_bytes64 < (1LL << 24)
        && row_widths.size(0) == stage_count64
        && (
            !lane_widths.defined()
            || lane_widths.size(0) == stage_count64
        )
        && destination.size(0) == stage_count64
        && destination.size(1) >= row_count
        && (
            !destination_indices.defined()
            || destination_indices.size(0) >= row_count
        )
        && scratch.size(0) == stage_count64
        && scratch.size(1) == capacity64,
        "staged runtime row geometry is inconsistent"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t capacity = (int32_t)capacity64;
    const int32_t max_row_bytes = (int32_t)max_row_bytes64;
    const int32_t max_lanes = (int32_t)scratch.size(2);
    TORCH_CHECK(
        max_lanes > 0
        && max_lanes <= 256
        && (int64_t)max_lanes * 2048 >= max_row_bytes64,
        "staged runtime row scratch width is inconsistent"
    );

    const int threads = BLAKE3_THREADS;
    const int32_t groups_per_block = threads / 4;
    const int32_t max_stage_leaves = row_count * max_lanes;
    const int blocks_x = (
        max_stage_leaves + groups_per_block - 1
    ) / groups_per_block;
    dim3 grid((unsigned int)blocks_x, (unsigned int)stage_count, 1);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_runtime_staged_leaf_parallel_kernel<<<
        grid, threads, 0, stream.stream()
    >>>(
        scratch.data_ptr<uint8_t>(),
        reinterpret_cast<const uint8_t*>(staging.data_ptr()),
        row_widths.data_ptr<int32_t>(),
        lane_widths.defined()
            ? lane_widths.data_ptr<int32_t>()
            : nullptr,
        stage_count,
        row_count,
        capacity,
        max_row_bytes,
        max_lanes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "staged runtime activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    const int64_t total_rows = (int64_t)stage_count * row_count;
    const int blocks = (
        (int32_t)total_rows + threads - 1
    ) / threads;
    blake3_runtime_staged_row_reduce_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        destination.data_ptr<uint8_t>(),
        scratch.data_ptr<uint8_t>(),
        row_widths.data_ptr<int32_t>(),
        lane_widths.defined()
            ? lane_widths.data_ptr<int32_t>()
            : nullptr,
        destination_indices.defined()
            ? destination_indices.data_ptr<int64_t>()
            : nullptr,
        stage_count,
        row_count,
        capacity,
        (int32_t)destination.size(1),
        destination.stride(0),
        destination.stride(1),
        max_lanes
    );
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "staged runtime row reduction failed: ",
        cudaGetErrorString(err)
    );
}

void cuda_blake3_runtime_staged_row_roots_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    int32_t row_count,
    int device_index)
{
    cuda_blake3_runtime_staged_row_roots_into_impl(
        destination,
        scratch,
        staging,
        row_widths,
        at::Tensor(),
        at::Tensor(),
        row_count,
        device_index
    );
}

void cuda_blake3_runtime_staged_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    int32_t row_count,
    int device_index)
{
    cuda_blake3_runtime_staged_row_roots_into_impl(
        destination,
        scratch,
        staging,
        row_widths,
        lane_widths,
        at::Tensor(),
        row_count,
        device_index
    );
}

void cuda_blake3_runtime_staged_row_roots_lanes_history_into(
    at::Tensor destination,
    at::Tensor scratch,
    at::Tensor staging,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    at::Tensor destination_indices,
    int32_t row_count,
    int device_index)
{
    cuda_blake3_runtime_staged_row_roots_into_impl(
        destination,
        scratch,
        staging,
        row_widths,
        lane_widths,
        destination_indices,
        row_count,
        device_index
    );
}

void cuda_blake3_runtime_tensor_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources_first,
    std::vector<at::Tensor> sources_second,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    int32_t row_count,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    const int64_t stage_count64 = (int64_t)sources_first.size();
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= BLAKE3_RUNTIME_TENSOR_MAX_STAGES
        && sources_second.size() == sources_first.size(),
        "runtime tensor root source inventory is malformed"
    );
    TORCH_CHECK(
        destination.is_cuda()
        && scratch.is_cuda()
        && row_widths.is_cuda()
        && lane_widths.is_cuda()
        && destination.device() == scratch.device()
        && destination.device() == row_widths.device()
        && destination.device() == lane_widths.device(),
        "runtime tensor roots require one CUDA device"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 3
        && destination.size(2) == 32
        && destination.is_contiguous()
        && destination.size(0) == stage_count64,
        "runtime tensor root destination is malformed"
    );
    TORCH_CHECK(
        scratch.scalar_type() == at::kByte
        && scratch.dim() == 4
        && scratch.size(0) == stage_count64
        && scratch.size(3) == 32
        && scratch.is_contiguous(),
        "runtime tensor root scratch is malformed"
    );
    TORCH_CHECK(
        row_widths.scalar_type() == at::kInt
        && row_widths.dim() == 1
        && row_widths.size(0) == stage_count64
        && row_widths.is_contiguous()
        && lane_widths.scalar_type() == at::kInt
        && lane_widths.dim() == 1
        && lane_widths.size(0) == stage_count64
        && lane_widths.is_contiguous(),
        "runtime tensor row geometry is malformed"
    );
    const int64_t capacity64 = scratch.size(1);
    const int32_t max_lanes = (int32_t)scratch.size(2);
    TORCH_CHECK(
        capacity64 > 0
        && capacity64 <= 0x7fffffffLL
        && row_count > 0
        && row_count <= capacity64
        && destination.size(1) >= row_count
        && max_lanes > 0
        && max_lanes <= 256,
        "runtime tensor root capacity is inconsistent"
    );

    RuntimeTensorSources sources = {};
    for (int64_t stage = 0; stage < stage_count64; stage++) {
        const at::Tensor& first = sources_first[(size_t)stage];
        const at::Tensor& second = sources_second[(size_t)stage];
        TORCH_CHECK(
            first.is_cuda()
            && second.is_cuda()
            && first.device() == destination.device()
            && second.device() == destination.device()
            && first.scalar_type() == at::kBFloat16
            && second.scalar_type() == at::kBFloat16
            && first.dim() == 2
            && second.dim() == 2
            && first.is_contiguous()
            && second.is_contiguous()
            && first.size(0) == row_count
            && second.sizes() == first.sizes()
            && first.size(1) > 0
            && first.size(1) * first.element_size() < (1LL << 24),
            "runtime tensor root source is malformed"
        );
        sources.first[stage] = reinterpret_cast<const uint8_t*>(
            first.data_ptr()
        );
        sources.second[stage] = reinterpret_cast<const uint8_t*>(
            second.data_ptr()
        );
        sources.row_bytes[stage] = (int32_t)(
            first.size(1) * first.element_size()
        );
        sources.source_rows[stage] = (int32_t)first.size(0);
        sources.combine_bfloat16[stage] = (
            first.data_ptr() == second.data_ptr() ? 0 : 1
        );
    }

    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t capacity = (int32_t)capacity64;
    const int threads = BLAKE3_THREADS;
    const int32_t groups_per_block = threads / 4;
    const int32_t max_stage_leaves = row_count * max_lanes;
    const int blocks_x = (
        max_stage_leaves + groups_per_block - 1
    ) / groups_per_block;
    dim3 grid((unsigned int)blocks_x, (unsigned int)stage_count, 1);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_runtime_tensor_leaf_parallel_kernel<<<
        grid, threads, 0, stream.stream()
    >>>(
        scratch.data_ptr<uint8_t>(),
        row_widths.data_ptr<int32_t>(),
        lane_widths.data_ptr<int32_t>(),
        sources,
        stage_count,
        row_count,
        capacity,
        max_lanes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "runtime tensor activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    const int64_t total_rows = stage_count64 * row_count;
    const int blocks = ((int32_t)total_rows + threads - 1) / threads;
    blake3_runtime_staged_row_reduce_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        destination.data_ptr<uint8_t>(),
        scratch.data_ptr<uint8_t>(),
        row_widths.data_ptr<int32_t>(),
        lane_widths.data_ptr<int32_t>(),
        nullptr,
        stage_count,
        row_count,
        capacity,
        (int32_t)destination.size(1),
        destination.stride(0),
        destination.stride(1),
        max_lanes
    );
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "runtime tensor row reduction failed: ",
        cudaGetErrorString(err)
    );
}

void cuda_copy_runtime_tensor_indexed_rows_into(
    std::vector<at::Tensor> destinations,
    std::vector<at::Tensor> sources,
    at::Tensor source_indices,
    at::Tensor staged_indices,
    int32_t row_count,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    const int64_t stage_count64 = (int64_t)sources.size();
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= BLAKE3_RUNTIME_TENSOR_MAX_STAGES
        && destinations.size() == sources.size(),
        "indexed runtime tensor snapshot inventory is malformed"
    );
    TORCH_CHECK(
        source_indices.is_cuda()
        && staged_indices.is_cuda()
        && source_indices.device() == staged_indices.device()
        && source_indices.scalar_type() == at::kInt
        && staged_indices.scalar_type() == at::kInt
        && source_indices.dim() == 2
        && staged_indices.dim() == 2
        && source_indices.size(0) == stage_count64
        && staged_indices.size(0) == stage_count64
        && source_indices.size(1) == row_count
        && staged_indices.size(1) == row_count
        && source_indices.is_contiguous()
        && staged_indices.is_contiguous()
        && row_count > 0,
        "indexed runtime tensor snapshot maps are malformed"
    );

    RuntimeTensorSources runtime_tensors = {};
    for (int64_t stage = 0; stage < stage_count64; stage++) {
        const at::Tensor& source = sources[(size_t)stage];
        const at::Tensor& destination = destinations[(size_t)stage];
        TORCH_CHECK(
            source.is_cuda()
            && destination.is_cuda()
            && source.device() == source_indices.device()
            && destination.device() == source_indices.device()
            && source.scalar_type() == destination.scalar_type()
            && (
                source.scalar_type() == at::kHalf
                || source.scalar_type() == at::kBFloat16
                || source.scalar_type() == at::kFloat
            )
            && source.dim() >= 2
            && source.dim() <= 4
            && destination.dim() == source.dim()
            && destination.size(0) >= row_count
            && source.size(0) > 0
            && source.size(0) <= 0x7fffffffLL
            && source.numel() % source.size(0) == 0
            && destination.is_contiguous(),
            "indexed runtime tensor snapshot geometry is malformed"
        );
        int64_t expected_stride = 1;
        for (int64_t dim = source.dim() - 1; dim >= 1; dim--) {
            TORCH_CHECK(
                source.size(dim) == destination.size(dim)
                && source.size(dim) > 0
                && source.stride(dim) == expected_stride,
                "indexed runtime tensor snapshot rows must be contiguous"
            );
            expected_stride *= source.size(dim);
        }
        const int64_t row_bytes = expected_stride * source.element_size();
        const int64_t source_row_stride_bytes = (
            source.stride(0) * source.element_size()
        );
        TORCH_CHECK(
            row_bytes > 0
            && row_bytes < (1LL << 24)
            && source_row_stride_bytes >= row_bytes
            && source_row_stride_bytes < (1LL << 32),
            "indexed runtime tensor snapshot row stride is malformed"
        );
        runtime_tensors.first[stage] = reinterpret_cast<const uint8_t*>(
            source.data_ptr()
        );
        runtime_tensors.second[stage] = reinterpret_cast<const uint8_t*>(
            destination.data_ptr()
        );
        runtime_tensors.row_bytes[stage] = (int32_t)row_bytes;
        runtime_tensors.source_rows[stage] = (int32_t)source.size(0);
        runtime_tensors.source_row_stride_bytes[stage] = (
            source_row_stride_bytes
        );
    }

    const int32_t stage_count = (int32_t)stage_count64;
    dim3 grid((unsigned int)row_count, (unsigned int)stage_count, 1);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    copy_runtime_tensor_indexed_rows_kernel<<<
        grid, 256, 0, stream.stream()
    >>>(
        runtime_tensors,
        source_indices.data_ptr<int32_t>(),
        staged_indices.data_ptr<int32_t>(),
        stage_count,
        row_count
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "indexed runtime tensor snapshot failed: ",
        cudaGetErrorString(err)
    );
}

void cuda_blake3_runtime_tensor_indexed_row_roots_lanes_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    at::Tensor source_indices,
    int32_t row_count,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    const int64_t stage_count64 = (int64_t)sources.size();
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= BLAKE3_RUNTIME_TENSOR_MAX_STAGES,
        "indexed runtime tensor root source inventory is malformed"
    );
    TORCH_CHECK(
        destination.is_cuda()
        && scratch.is_cuda()
        && row_widths.is_cuda()
        && lane_widths.is_cuda()
        && source_indices.is_cuda()
        && destination.device() == scratch.device()
        && destination.device() == row_widths.device()
        && destination.device() == lane_widths.device()
        && destination.device() == source_indices.device(),
        "indexed runtime tensor roots require one CUDA device"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 3
        && destination.size(0) == stage_count64
        && destination.size(2) == 32
        && destination.is_contiguous(),
        "indexed runtime tensor root destination is malformed"
    );
    TORCH_CHECK(
        scratch.scalar_type() == at::kByte
        && scratch.dim() == 4
        && scratch.size(0) == stage_count64
        && scratch.size(3) == 32
        && scratch.is_contiguous(),
        "indexed runtime tensor root scratch is malformed"
    );
    TORCH_CHECK(
        row_widths.scalar_type() == at::kInt
        && row_widths.dim() == 1
        && row_widths.size(0) == stage_count64
        && row_widths.is_contiguous()
        && lane_widths.scalar_type() == at::kInt
        && lane_widths.dim() == 1
        && lane_widths.size(0) == stage_count64
        && lane_widths.is_contiguous(),
        "indexed runtime tensor row geometry is malformed"
    );
    TORCH_CHECK(
        source_indices.scalar_type() == at::kInt
        && source_indices.dim() == 2
        && source_indices.size(0) == stage_count64
        && source_indices.size(1) >= row_count
        && source_indices.is_contiguous(),
        "indexed runtime tensor source map is malformed"
    );
    const int64_t capacity64 = scratch.size(1);
    const int32_t max_lanes = (int32_t)scratch.size(2);
    TORCH_CHECK(
        capacity64 > 0
        && capacity64 <= 0x7fffffffLL
        && row_count > 0
        && row_count <= capacity64
        && destination.size(1) >= row_count
        && max_lanes > 0
        && max_lanes <= 65536,
        "indexed runtime tensor root capacity is inconsistent"
    );

    RuntimeTensorSources runtime_sources = {};
    for (int64_t stage = 0; stage < stage_count64; stage++) {
        const at::Tensor& source = sources[(size_t)stage];
        TORCH_CHECK(
            source.is_cuda()
            && source.device() == destination.device()
            && (
                source.scalar_type() == at::kHalf
                || source.scalar_type() == at::kBFloat16
                || source.scalar_type() == at::kFloat
            )
            && source.dim() >= 2
            && source.dim() <= 4
            && source.size(0) > 0
            && source.size(0) <= 0x7fffffffLL
            && source.numel() % source.size(0) == 0
            && source.numel() / source.size(0) > 0
            && source.numel() / source.size(0) * source.element_size()
                < (1LL << 24),
            "indexed runtime tensor root source is malformed"
        );
        runtime_sources.first[stage] = reinterpret_cast<const uint8_t*>(
            source.data_ptr()
        );
        runtime_sources.second[stage] = runtime_sources.first[stage];
        runtime_sources.row_bytes[stage] = (int32_t)(
            source.numel() / source.size(0) * source.element_size()
        );
        runtime_sources.source_rows[stage] = (int32_t)source.size(0);
        runtime_sources.combine_bfloat16[stage] = 0;
        int64_t expected_stride = 1;
        for (int64_t dim = source.dim() - 1; dim >= 1; dim--) {
            TORCH_CHECK(
                source.size(dim) > 0
                && source.size(dim) <= 0x7fffffffLL
                && source.stride(dim) == expected_stride,
                "indexed runtime tensor rows must be contiguous"
            );
            expected_stride *= source.size(dim);
        }
        const int64_t source_row_stride_bytes = (
            source.stride(0) * source.element_size()
        );
        TORCH_CHECK(
            source_row_stride_bytes >= runtime_sources.row_bytes[stage]
            && source_row_stride_bytes < (1LL << 32),
            "indexed runtime tensor row stride is malformed"
        );
        runtime_sources.source_row_stride_bytes[stage] = (
            source_row_stride_bytes
        );
    }

    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t capacity = (int32_t)capacity64;
    const int32_t groups_per_block = BLAKE3_THREADS / 4;
    const int64_t max_stage_leaves = (int64_t)row_count * max_lanes;
    TORCH_CHECK(
        max_stage_leaves > 0 && max_stage_leaves <= 0x7fffffffLL,
        "indexed runtime tensor leaf count is outside the supported range"
    );
    const int blocks_x = (
        (int32_t)max_stage_leaves + groups_per_block - 1
    ) / groups_per_block;
    dim3 grid((unsigned int)blocks_x, (unsigned int)stage_count, 1);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_runtime_tensor_indexed_leaf_parallel_kernel<<<
        grid, BLAKE3_THREADS, 0, stream.stream()
    >>>(
        scratch.data_ptr<uint8_t>(),
        row_widths.data_ptr<int32_t>(),
        lane_widths.data_ptr<int32_t>(),
        source_indices.data_ptr<int32_t>(),
        runtime_sources,
        stage_count,
        row_count,
        capacity,
        max_lanes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "indexed runtime tensor activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    const int64_t total_rows = stage_count64 * row_count;
    const int blocks = (
        (int32_t)total_rows + BLAKE3_THREADS - 1
    ) / BLAKE3_THREADS;
    if (max_lanes <= 1024) {
        const size_t shared_bytes = (size_t)(
            max_lanes + (max_lanes + 1) / 2
        ) * 32;
        dim3 reduce_grid(
            (unsigned int)row_count,
            (unsigned int)stage_count,
            1
        );
        blake3_runtime_indexed_row_reduce_parallel_kernel<<<
            reduce_grid,
            BLAKE3_THREADS,
            shared_bytes,
            stream.stream()
        >>>(
            destination.data_ptr<uint8_t>(),
            scratch.data_ptr<uint8_t>(),
            row_widths.data_ptr<int32_t>(),
            lane_widths.data_ptr<int32_t>(),
            source_indices.data_ptr<int32_t>(),
            nullptr,
            runtime_sources,
            stage_count,
            row_count,
            capacity,
            (int32_t)destination.size(1),
            destination.stride(0),
            destination.stride(1),
            row_count,
            max_lanes
        );
    } else {
        blake3_runtime_staged_row_reduce_kernel<<<
            blocks, BLAKE3_THREADS, 0, stream.stream()
        >>>(
            destination.data_ptr<uint8_t>(),
            scratch.data_ptr<uint8_t>(),
            row_widths.data_ptr<int32_t>(),
            lane_widths.data_ptr<int32_t>(),
            nullptr,
            stage_count,
            row_count,
            capacity,
            (int32_t)destination.size(1),
            destination.stride(0),
            destination.stride(1),
            max_lanes
        );
    }
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "indexed runtime tensor row reduction failed: ",
        cudaGetErrorString(err)
    );
    if (max_lanes > 1024) {
        blake3_runtime_tensor_zero_invalid_indexed_roots_kernel<<<
            blocks, BLAKE3_THREADS, 0, stream.stream()
        >>>(
            destination.data_ptr<uint8_t>(),
            source_indices.data_ptr<int32_t>(),
            runtime_sources,
            stage_count,
            row_count,
            destination.stride(0),
            destination.stride(1)
        );
        err = cudaGetLastError();
        TORCH_CHECK(
            err == cudaSuccess,
            "indexed runtime tensor source validation failed: ",
            cudaGetErrorString(err)
        );
    }
}

void cuda_blake3_runtime_tensor_indexed_row_roots_lanes_history_into(
    at::Tensor destination,
    at::Tensor scratch,
    std::vector<at::Tensor> sources,
    at::Tensor row_widths,
    at::Tensor lane_widths,
    std::vector<at::Tensor> source_indices,
    at::Tensor destination_indices,
    int32_t row_count,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    const int64_t stage_count64 = (int64_t)sources.size();
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= BLAKE3_RUNTIME_TENSOR_MAX_STAGES,
        "indexed runtime tensor root source inventory is malformed"
    );
    TORCH_CHECK(
        destination.is_cuda()
        && scratch.is_cuda()
        && row_widths.is_cuda()
        && lane_widths.is_cuda()
        && destination_indices.is_cuda()
        && destination.device() == scratch.device()
        && destination.device() == row_widths.device()
        && destination.device() == lane_widths.device()
        && destination.device() == destination_indices.device(),
        "indexed runtime tensor roots require one CUDA device"
    );
    TORCH_CHECK(
        destination.scalar_type() == at::kByte
        && destination.dim() == 3
        && destination.size(2) == 32
        && destination.is_contiguous()
        && destination.size(0) == stage_count64,
        "indexed runtime tensor root destination is malformed"
    );
    TORCH_CHECK(
        scratch.scalar_type() == at::kByte
        && scratch.dim() == 4
        && scratch.size(0) == stage_count64
        && scratch.size(3) == 32
        && scratch.is_contiguous(),
        "indexed runtime tensor root scratch is malformed"
    );
    TORCH_CHECK(
        row_widths.scalar_type() == at::kInt
        && row_widths.dim() == 1
        && row_widths.size(0) == stage_count64
        && row_widths.is_contiguous()
        && lane_widths.scalar_type() == at::kInt
        && lane_widths.dim() == 1
        && lane_widths.size(0) == stage_count64
        && lane_widths.is_contiguous(),
        "indexed runtime tensor row geometry is malformed"
    );
    TORCH_CHECK(
        destination_indices.scalar_type() == at::kLong
        && destination_indices.dim() == 1
        && destination_indices.is_contiguous()
        && row_count > 0
        && destination_indices.size(0) >= row_count,
        "indexed runtime tensor row mappings are malformed"
    );
    const int64_t capacity64 = scratch.size(1);
    const int32_t max_lanes = (int32_t)scratch.size(2);
    TORCH_CHECK(
        capacity64 > 0
        && capacity64 <= 0x7fffffffLL
        && row_count <= capacity64
        && destination.size(1) > 0
        && destination.size(1) <= 0x7fffffffLL
        && max_lanes > 0
        && max_lanes <= 65536,
        "indexed runtime tensor root capacity is inconsistent"
    );

    RuntimeTensorSources runtime_sources = {};
    TORCH_CHECK(
        (int64_t)source_indices.size() == stage_count64,
        "indexed runtime tensor source-index inventory is malformed"
    );
    for (int64_t stage = 0; stage < stage_count64; stage++) {
        const at::Tensor& source = sources[(size_t)stage];
        const at::Tensor& indices = source_indices[(size_t)stage];
        TORCH_CHECK(
            source.is_cuda()
            && source.device() == destination.device()
            && (
                source.scalar_type() == at::kHalf
                || source.scalar_type() == at::kBFloat16
                || source.scalar_type() == at::kFloat
            )
            && source.dim() >= 2
            && source.dim() <= 4
            && source.size(0) > 0
            && source.size(0) <= 0x7fffffffLL
            && source.numel() % source.size(0) == 0
            && source.numel() / source.size(0) > 0
            && source.numel() / source.size(0) * source.element_size()
                < (1LL << 24),
            "indexed runtime tensor root source is malformed"
        );
        TORCH_CHECK(
            indices.is_cuda()
            && indices.device() == destination.device()
            && indices.scalar_type() == at::kInt
            && indices.dim() == 1
            && indices.is_contiguous()
            && indices.size(0) >= row_count,
            "indexed runtime tensor source indices are malformed"
        );
        runtime_sources.first[stage] = reinterpret_cast<const uint8_t*>(
            source.data_ptr()
        );
        runtime_sources.second[stage] = runtime_sources.first[stage];
        runtime_sources.row_bytes[stage] = (int32_t)(
            source.numel() / source.size(0) * source.element_size()
        );
        runtime_sources.source_rows[stage] = (int32_t)source.size(0);
        runtime_sources.source_indices[stage] = indices.data_ptr<int32_t>();
        runtime_sources.combine_bfloat16[stage] = 0;
        int64_t expected_stride = 1;
        for (int64_t dim = source.dim() - 1; dim >= 1; dim--) {
            TORCH_CHECK(
                source.size(dim) > 0
                && source.size(dim) <= 0x7fffffffLL
                && source.stride(dim) == expected_stride,
                "indexed runtime tensor history rows must be contiguous"
            );
            expected_stride *= source.size(dim);
        }
        const int64_t source_row_stride_bytes = (
            source.stride(0) * source.element_size()
        );
        TORCH_CHECK(
            source_row_stride_bytes >= runtime_sources.row_bytes[stage]
            && source_row_stride_bytes < (1LL << 32),
            "indexed runtime tensor history row stride is malformed"
        );
        runtime_sources.source_row_stride_bytes[stage] = (
            source_row_stride_bytes
        );
    }

    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t capacity = (int32_t)capacity64;
    dim3 grid((unsigned int)row_count, (unsigned int)stage_count, 1);
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_runtime_tensor_indexed_leaf_kernel<<<
        grid, BLAKE3_THREADS, 0, stream.stream()
    >>>(
        scratch.data_ptr<uint8_t>(),
        row_widths.data_ptr<int32_t>(),
        lane_widths.data_ptr<int32_t>(),
        destination_indices.data_ptr<int64_t>(),
        runtime_sources,
        stage_count,
        row_count,
        capacity,
        max_lanes
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "indexed runtime tensor activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    const int64_t total_rows = stage_count64 * row_count;
    const int blocks = (
        (int32_t)total_rows + BLAKE3_THREADS - 1
    ) / BLAKE3_THREADS;
    if (max_lanes <= 1024) {
        const size_t shared_bytes = (size_t)(
            max_lanes + (max_lanes + 1) / 2
        ) * 32;
        blake3_runtime_indexed_row_reduce_parallel_kernel<<<
            grid,
            BLAKE3_THREADS,
            shared_bytes,
            stream.stream()
        >>>(
            destination.data_ptr<uint8_t>(),
            scratch.data_ptr<uint8_t>(),
            row_widths.data_ptr<int32_t>(),
            lane_widths.data_ptr<int32_t>(),
            nullptr,
            destination_indices.data_ptr<int64_t>(),
            runtime_sources,
            stage_count,
            row_count,
            capacity,
            (int32_t)destination.size(1),
            destination.stride(0),
            destination.stride(1),
            0,
            max_lanes
        );
    } else {
        blake3_runtime_staged_row_reduce_kernel<<<
            blocks, BLAKE3_THREADS, 0, stream.stream()
        >>>(
            destination.data_ptr<uint8_t>(),
            scratch.data_ptr<uint8_t>(),
            row_widths.data_ptr<int32_t>(),
            lane_widths.data_ptr<int32_t>(),
            destination_indices.data_ptr<int64_t>(),
            stage_count,
            row_count,
            capacity,
            (int32_t)destination.size(1),
            destination.stride(0),
            destination.stride(1),
            max_lanes
        );
    }
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "indexed runtime tensor row reduction failed: ",
        cudaGetErrorString(err)
    );
}

std::tuple<at::Tensor, at::Tensor>
cuda_blake3_prehashed_update_peaks(
    at::Tensor row_roots,
    int64_t base_count,
    at::Tensor initial_peaks,
    at::Tensor prior_deferred,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        row_roots.dtype() == at::kByte
        && row_roots.is_cuda()
        && row_roots.dim() == 3
        && row_roots.size(2) == 32,
        "prehashed activation roots must be CUDA uint8 [stages,rows,32]"
    );
    auto roots = row_roots.contiguous();
    const int64_t stage_count64 = roots.size(0);
    const int64_t rows_per_stage64 = roots.size(1);
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= 0x7fffffffLL
        && rows_per_stage64 > 0
        && rows_per_stage64 <= 0x7fffffffLL
        && base_count >= 0
        && base_count + rows_per_stage64 < (1LL << 32),
        "prehashed activation chronology is outside the supported range"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t rows_per_stage = (int32_t)rows_per_stage64;
    TORCH_CHECK(
        initial_peaks.dtype() == at::kByte
        && initial_peaks.is_cuda()
        && initial_peaks.device() == roots.device()
        && initial_peaks.dim() == 3
        && initial_peaks.size(0) == stage_count
        && initial_peaks.size(1) == 32
        && initial_peaks.size(2) == 32,
        "initial prehashed execution-anchor peaks are malformed"
    );
    TORCH_CHECK(
        prior_deferred.dtype() == at::kByte
        && prior_deferred.is_cuda()
        && prior_deferred.device() == roots.device()
        && prior_deferred.dim() == 2
        && prior_deferred.size(0) == stage_count
        && prior_deferred.size(1) == 32,
        "prior prehashed execution-anchor rows are malformed"
    );

    auto peaks = initial_peaks.contiguous().clone();
    auto deferred = at::empty_like(prior_deferred);
    const int threads = BLAKE3_THREADS;
    const int blocks = (stage_count + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_prehashed_streaming_peaks_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        peaks.data_ptr<uint8_t>(),
        deferred.data_ptr<uint8_t>(),
        prior_deferred.data_ptr<uint8_t>(),
        roots.data_ptr<uint8_t>(),
        stage_count,
        rows_per_stage,
        1,
        rows_per_stage,
        (uint32_t)base_count
    );
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "prehashed execution-anchor frontier update failed: ",
        cudaGetErrorString(err)
    );
    return std::make_tuple(peaks, deferred);
}

void cuda_blake3_prehashed_update_peaks_inplace(
    at::Tensor row_roots,
    int64_t base_count,
    at::Tensor peaks,
    at::Tensor deferred,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        row_roots.dtype() == at::kByte
        && row_roots.is_cuda()
        && row_roots.dim() == 3
        && row_roots.size(2) == 32
        && row_roots.stride(2) == 1
        && row_roots.stride(0) % 32 == 0
        && row_roots.stride(1) % 32 == 0,
        "in-place prehashed roots must be CUDA uint8 [stages,rows,32]"
    );
    const int64_t stage_count64 = row_roots.size(0);
    const int64_t rows_per_stage64 = row_roots.size(1);
    const int64_t stage_stride64 = row_roots.stride(0) / 32;
    const int64_t row_stride64 = row_roots.stride(1) / 32;
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= 0x7fffffffLL
        && rows_per_stage64 > 0
        && rows_per_stage64 <= 0x7fffffffLL
        && (
            (row_stride64 == 1 && stage_stride64 >= rows_per_stage64)
            || (stage_stride64 == 1 && row_stride64 >= stage_count64)
        )
        && stage_stride64 <= 0x7fffffffLL
        && row_stride64 <= 0x7fffffffLL
        && base_count >= 0
        && base_count + rows_per_stage64 < (1LL << 32),
        "in-place prehashed chronology is outside the supported range"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    TORCH_CHECK(
        peaks.dtype() == at::kByte
        && peaks.is_cuda()
        && peaks.device() == row_roots.device()
        && peaks.is_contiguous()
        && peaks.sizes() == at::IntArrayRef({stage_count, 32, 32})
        && deferred.dtype() == at::kByte
        && deferred.is_cuda()
        && deferred.device() == row_roots.device()
        && deferred.is_contiguous()
        && deferred.sizes() == at::IntArrayRef({stage_count, 32}),
        "in-place prehashed frontier state is malformed"
    );

    const int threads = BLAKE3_THREADS;
    const int blocks = (stage_count + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_prehashed_streaming_peaks_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        peaks.data_ptr<uint8_t>(),
        deferred.data_ptr<uint8_t>(),
        deferred.data_ptr<uint8_t>(),
        row_roots.data_ptr<uint8_t>(),
        stage_count,
        (int32_t)stage_stride64,
        (int32_t)row_stride64,
        (int32_t)rows_per_stage64,
        (uint32_t)base_count
    );
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "in-place prehashed frontier update failed: ",
        cudaGetErrorString(err)
    );
}


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
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        row_roots.dtype() == at::kByte
        && row_roots.is_cuda()
        && row_roots.dim() == 3
        && row_roots.size(2) == 32,
        "combined prehashed roots must be CUDA uint8 [stages,rows,32]"
    );
    auto roots = row_roots.contiguous();
    const int64_t stage_count64 = roots.size(0);
    const int64_t rows_per_stage64 = roots.size(1);
    constexpr int32_t capacity = 64;
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= 0x7fffffffLL
        && rows_per_stage64 > 0
        && rows_per_stage64 <= 0x7fffffffLL
        && base_count >= 0
        && base_count + rows_per_stage64 < (1LL << 32)
        && range_start >= 0
        && range_start <= range_base_count
        && range_row_start >= 0
        && range_row_start < rows_per_stage64
        && range_base_count + rows_per_stage64 - range_row_start < (1LL << 32),
        "combined prehashed chronology is outside the supported range"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    const int32_t rows_per_stage = (int32_t)rows_per_stage64;
    TORCH_CHECK(
        initial_peaks.dtype() == at::kByte
        && initial_peaks.is_cuda()
        && initial_peaks.device() == roots.device()
        && initial_peaks.sizes() == at::IntArrayRef({stage_count, 32, 32})
        && prior_deferred.dtype() == at::kByte
        && prior_deferred.is_cuda()
        && prior_deferred.device() == roots.device()
        && prior_deferred.sizes() == at::IntArrayRef({stage_count, 32})
        && range_hashes.dtype() == at::kByte
        && range_hashes.is_cuda()
        && range_hashes.is_contiguous()
        && range_hashes.device() == roots.device()
        && range_hashes.sizes() == at::IntArrayRef({stage_count, capacity, 32})
        && range_levels.dtype() == at::kInt
        && range_levels.is_cuda()
        && range_levels.is_contiguous()
        && range_levels.device() == roots.device()
        && range_levels.sizes() == at::IntArrayRef({stage_count, capacity})
        && range_indices.dtype() == at::kLong
        && range_indices.is_cuda()
        && range_indices.is_contiguous()
        && range_indices.device() == roots.device()
        && range_indices.sizes() == at::IntArrayRef({stage_count, capacity})
        && range_lengths.dtype() == at::kInt
        && range_lengths.is_cuda()
        && range_lengths.is_contiguous()
        && range_lengths.device() == roots.device()
        && range_lengths.sizes() == at::IntArrayRef({stage_count}),
        "combined prehashed frontier state is malformed"
    );

    auto peaks = initial_peaks.contiguous().clone();
    auto deferred = at::empty_like(prior_deferred);
    // The range state is private prover-local scratch. Mutate it in place so
    // ordinary decode does not clone O(stage_count * 64) bytes every token.
    auto hashes = range_hashes;
    auto levels = range_levels;
    auto indices = range_indices;
    auto lengths = range_lengths;
    const int threads = BLAKE3_THREADS;
    const int blocks = (stage_count + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_prehashed_streaming_peaks_ranges_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        peaks.data_ptr<uint8_t>(),
        deferred.data_ptr<uint8_t>(),
        prior_deferred.data_ptr<uint8_t>(),
        hashes.data_ptr<uint8_t>(),
        levels.data_ptr<int32_t>(),
        indices.data_ptr<int64_t>(),
        lengths.data_ptr<int32_t>(),
        roots.data_ptr<uint8_t>(),
        stage_count,
        rows_per_stage,
        (uint32_t)base_count,
        (int32_t)range_row_start,
        (uint32_t)range_base_count,
        capacity
    );
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "combined execution-anchor frontier update failed: ",
        cudaGetErrorString(err)
    );
    return std::make_tuple(
        peaks, deferred, hashes, levels, indices, lengths
    );
}


/**
 * Hash a stage-major batch of runtime rows and append them to authenticated
 * O(log context) frontiers entirely on device.
 */
std::tuple<at::Tensor, at::Tensor> cuda_blake3_activation_update_peaks(
    at::Tensor W,
    int32_t words_per_row,
    int32_t rows_per_stage,
    int64_t base_count,
    at::Tensor initial_peaks,
    int device_index,
    int32_t words_per_leaf)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    auto packed = validated_activation_rows(
        W,
        words_per_row,
        words_per_leaf,
        "cuda_blake3_activation_update_peaks");
    TORCH_CHECK(
        rows_per_stage > 0
        && packed.size(0) % rows_per_stage == 0,
        "rows_per_stage does not divide the activation batch"
    );
    const int64_t stage_count64 = packed.size(0) / rows_per_stage;
    TORCH_CHECK(stage_count64 > 0 && stage_count64 <= 0x7fffffffLL,
        "stage count is outside the supported range");
    TORCH_CHECK(
        base_count >= 0
        && base_count + rows_per_stage < (1LL << 32),
        "updated execution-anchor row count is outside the supported range"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    TORCH_CHECK(
        initial_peaks.dtype() == at::kByte
        && initial_peaks.is_cuda()
        && initial_peaks.device() == packed.device()
        && initial_peaks.dim() == 3
        && initial_peaks.size(0) == stage_count
        && initial_peaks.size(1) == 32
        && initial_peaks.size(2) == 32,
        "initial execution-anchor peaks are malformed"
    );

    auto row_roots = activation_row_roots_device(
        packed, words_per_row, words_per_leaf, device_index);
    auto peaks = initial_peaks.contiguous().clone();
    const int threads = BLAKE3_THREADS;
    const int blocks = (stage_count + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    blake3_streaming_peaks_kernel<<<blocks, threads, 0, stream.stream()>>>(
        peaks.data_ptr<uint8_t>(),
        row_roots.data_ptr<uint8_t>(),
        stage_count,
        rows_per_stage,
        rows_per_stage - 1,
        (uint32_t)base_count
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "blake3_streaming_peaks_kernel failed: ", cudaGetErrorString(err));
    auto deferred = row_roots.view({
        stage_count, rows_per_stage, 32
    }).select(1, rows_per_stage - 1).contiguous();
    return std::make_tuple(peaks.cpu(), deferred.cpu());
}


/**
 * Hash and frontier-update a bounded batch of variable-width stages.
 *
 * ``W`` concatenates complete padded stage tensors in stage-major order.
 * The metadata retains exact row boundaries, so this is byte-identical to
 * calling ``cuda_blake3_activation_update_peaks`` once per distinct width
 * while avoiding fragmented launches and synchronizations.
 */
std::tuple<at::Tensor, at::Tensor>
cuda_blake3_activation_update_peaks_heterogeneous(
    at::Tensor W,
    at::Tensor stage_chunk_offsets,
    at::Tensor chunks_per_row,
    int32_t rows_per_stage,
    int64_t base_count,
    at::Tensor initial_peaks,
    at::Tensor prior_deferred,
    int device_index)
{
    c10::cuda::CUDAGuard device_guard(device_index);
    TORCH_CHECK(
        W.dtype() == at::kLong && W.is_cuda(),
        "heterogeneous activation rows require a CUDA int64 tensor"
    );
    auto packed = W.contiguous().view(-1);
    TORCH_CHECK(
        packed.numel() > 0 && packed.numel() % 256 == 0,
        "heterogeneous activation rows must contain complete 2-KiB leaves"
    );
    TORCH_CHECK(
        rows_per_stage > 0
        && base_count >= 0
        && base_count + rows_per_stage < (1LL << 32),
        "heterogeneous activation row chronology is outside the supported range"
    );
    TORCH_CHECK(
        stage_chunk_offsets.dtype() == at::kLong
        && chunks_per_row.dtype() == at::kInt
        && stage_chunk_offsets.dim() == 1
        && chunks_per_row.dim() == 1,
        "heterogeneous activation row metadata is malformed"
    );
    auto offsets_cpu = stage_chunk_offsets.cpu().contiguous();
    auto widths_cpu = chunks_per_row.cpu().contiguous();
    const int64_t stage_count64 = widths_cpu.numel();
    TORCH_CHECK(
        stage_count64 > 0
        && stage_count64 <= 0x7fffffffLL
        && offsets_cpu.numel() == stage_count64 + 1,
        "heterogeneous activation stage count is unsupported"
    );
    const int32_t stage_count = (int32_t)stage_count64;
    const int64_t total_chunks = packed.numel() / 256;
    TORCH_CHECK(
        total_chunks <= 0x7fffffffLL,
        "heterogeneous activation leaf count is unsupported"
    );
    const int64_t* offsets = offsets_cpu.data_ptr<int64_t>();
    const int32_t* widths = widths_cpu.data_ptr<int32_t>();
    TORCH_CHECK(offsets[0] == 0, "heterogeneous activation offsets must start at zero");
    for (int32_t stage = 0; stage < stage_count; stage++) {
        TORCH_CHECK(
            widths[stage] > 0
            && offsets[stage + 1] - offsets[stage]
                == (int64_t)widths[stage] * rows_per_stage,
            "heterogeneous activation stage geometry is inconsistent"
        );
    }
    TORCH_CHECK(
        offsets[stage_count] == total_chunks,
        "heterogeneous activation offsets do not cover the input"
    );
    TORCH_CHECK(
        initial_peaks.dtype() == at::kByte
        && initial_peaks.is_cuda()
        && initial_peaks.device() == packed.device()
        && initial_peaks.dim() == 3
        && initial_peaks.size(0) == stage_count
        && initial_peaks.size(1) == 32
        && initial_peaks.size(2) == 32,
        "initial heterogeneous execution-anchor peaks are malformed"
    );
    TORCH_CHECK(
        prior_deferred.dtype() == at::kByte
        && prior_deferred.is_cuda()
        && prior_deferred.device() == packed.device()
        && prior_deferred.dim() == 2
        && prior_deferred.size(0) == stage_count
        && prior_deferred.size(1) == 32,
        "prior heterogeneous execution-anchor rows are malformed"
    );

    auto byte_options = at::TensorOptions()
        .dtype(at::kByte).device(packed.device());
    auto hashes = at::empty({total_chunks, 32}, byte_options);
    auto peaks = initial_peaks.contiguous().clone();
    auto deferred = at::empty({stage_count, 32}, byte_options);
    auto offsets_gpu = offsets_cpu.to(packed.device());
    auto widths_gpu = widths_cpu.to(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream(device_index);
    const int threads = BLAKE3_THREADS;

    int blocks = ((int32_t)total_chunks + threads - 1) / threads;
    blake3_activation_leaf_kernel<<<blocks, threads, 0, stream.stream()>>>(
        hashes.data_ptr<uint8_t>(),
        packed.data_ptr<int64_t>(),
        packed.numel(),
        256,
        (int32_t)total_chunks
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "heterogeneous activation leaf kernel failed: ",
        cudaGetErrorString(err)
    );

    const int64_t total_rows = stage_count64 * rows_per_stage;
    TORCH_CHECK(
        total_rows <= 0x7fffffffLL,
        "heterogeneous activation row count is unsupported"
    );
    blocks = ((int32_t)total_rows + threads - 1) / threads;
    blake3_heterogeneous_row_reduce_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        hashes.data_ptr<uint8_t>(),
        offsets_gpu.data_ptr<int64_t>(),
        widths_gpu.data_ptr<int32_t>(),
        stage_count,
        rows_per_stage
    );
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "heterogeneous activation row reduction failed: ",
        cudaGetErrorString(err)
    );

    blocks = (stage_count + threads - 1) / threads;
    blake3_heterogeneous_streaming_peaks_kernel<<<
        blocks, threads, 0, stream.stream()
    >>>(
        peaks.data_ptr<uint8_t>(),
        deferred.data_ptr<uint8_t>(),
        prior_deferred.data_ptr<uint8_t>(),
        hashes.data_ptr<uint8_t>(),
        offsets_gpu.data_ptr<int64_t>(),
        widths_gpu.data_ptr<int32_t>(),
        stage_count,
        rows_per_stage,
        (uint32_t)base_count
    );
    err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "heterogeneous activation frontier update failed: ",
        cudaGetErrorString(err)
    );
    return std::make_tuple(peaks, deferred);
}
