// keccak256.h — C++ Fiat-Shamir transcript using OpenSSL SHA3-256
//
// Exactly reproduces Python's Transcript class from zkllm/crypto/transcript.py.
// Uses OpenSSL EVP API for SHA3-256 (FIPS 202 Keccak-256).
//
// Thread-safe: each NativeTranscript instance is independent.
// No CUDA dependency — pure host-side computation.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>
#include <openssl/evp.h>

namespace zkllm {

static constexpr uint64_t TRANSCRIPT_MERSENNE_P = (1ULL << 61) - 1;

// --------------------------------------------------------------------------
//  Low-level helpers
// --------------------------------------------------------------------------

// Pack uint64_t as 8 bytes little-endian (matches Python struct.pack("<Q", v))
inline void pack_le64(uint8_t out[8], uint64_t value) {
    for (int i = 0; i < 8; i++) {
        out[i] = static_cast<uint8_t>(value & 0xFF);
        value >>= 8;
    }
}

// Read little-endian uint64_t from 8 bytes (matches Python int.from_bytes(b, "little"))
inline uint64_t read_le64(const uint8_t* data) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; i--) {
        result = (result << 8) | data[i];
    }
    return result;
}

// SHA3-256 hash of concatenated byte buffers.
// Uses OpenSSL EVP API for streaming multi-part input.
inline std::array<uint8_t, 32> sha3_256(
    std::initializer_list<std::pair<const uint8_t*, size_t>> parts
) {
    std::array<uint8_t, 32> digest{};
    unsigned int digest_len = 32;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha3_256(), nullptr);
    for (const auto& part : parts) {
        if (part.second > 0) {
            EVP_DigestUpdate(ctx, part.first, part.second);
        }
    }
    EVP_DigestFinal_ex(ctx, digest.data(), &digest_len);
    EVP_MD_CTX_free(ctx);

    return digest;
}

// --------------------------------------------------------------------------
//  NativeTranscript — bit-exact replica of Python Transcript
// --------------------------------------------------------------------------

class NativeTranscript {
    std::array<uint8_t, 32> state_;

public:
    // Initialize from a 32-byte state (matching Python Transcript._state)
    explicit NativeTranscript(const std::array<uint8_t, 32>& initial_state)
        : state_(initial_state) {}

    // Initialize from raw byte pointer (must point to >= 32 bytes)
    explicit NativeTranscript(const uint8_t* state_bytes) {
        std::memcpy(state_.data(), state_bytes, 32);
    }

    // ----- absorb -----
    // Python: state = sha3_256(state || label || pack_le64(value % 2^64))
    void absorb(const uint8_t* label, size_t label_len, uint64_t value) {
        uint8_t value_bytes[8];
        pack_le64(value_bytes, value);

        state_ = sha3_256({
            {state_.data(), 32},
            {label, label_len},
            {value_bytes, 8},
        });
    }

    void absorb(const std::string& label, uint64_t value) {
        absorb(reinterpret_cast<const uint8_t*>(label.data()),
               label.size(), value);
    }

    // ----- absorb_bytes -----
    // Python: state = sha3_256(state || label || pack_le64(len(data)) || data)
    void absorb_bytes(const uint8_t* label, size_t label_len,
                      const uint8_t* data, size_t data_len) {
        uint8_t len_bytes[8];
        pack_le64(len_bytes, static_cast<uint64_t>(data_len));

        state_ = sha3_256({
            {state_.data(), 32},
            {label, label_len},
            {len_bytes, 8},
            {data, data_len},
        });
    }

    void absorb_bytes(const std::string& label,
                      const uint8_t* data, size_t data_len) {
        absorb_bytes(reinterpret_cast<const uint8_t*>(label.data()),
                     label.size(), data, data_len);
    }

    // ----- squeeze -----
    // Python:
    //   digest = sha3_256(state || "squeeze:" || label)
    //   state  = sha3_256(state || digest)
    //   return int.from_bytes(digest[:8], "little") % P
    uint64_t squeeze(const uint8_t* label, size_t label_len) {
        static const uint8_t squeeze_prefix[] = {
            's', 'q', 'u', 'e', 'e', 'z', 'e', ':'
        };

        // digest = sha3_256(state || "squeeze:" || label)
        auto digest = sha3_256({
            {state_.data(), 32},
            {squeeze_prefix, 8},
            {label, label_len},
        });

        // state = sha3_256(state || digest)
        state_ = sha3_256({
            {state_.data(), 32},
            {digest.data(), 32},
        });

        // Extract field element from first 8 bytes (little-endian), mod P
        uint64_t raw = read_le64(digest.data());
        return raw % TRANSCRIPT_MERSENNE_P;
    }

    uint64_t squeeze(const std::string& label) {
        return squeeze(reinterpret_cast<const uint8_t*>(label.data()),
                       label.size());
    }

    // ----- squeeze_n -----
    // Python: for i in range(n): sublabel = label + b":" + pack("<Q", i)
    std::vector<uint64_t> squeeze_n(
        const uint8_t* label, size_t label_len, int n
    ) {
        std::vector<uint64_t> result;
        result.reserve(n);

        for (int i = 0; i < n; i++) {
            // Build sublabel: label + ":" + pack_le64(i)
            std::vector<uint8_t> sublabel(label, label + label_len);
            sublabel.push_back(':');
            uint8_t idx_bytes[8];
            pack_le64(idx_bytes, static_cast<uint64_t>(i));
            sublabel.insert(sublabel.end(), idx_bytes, idx_bytes + 8);

            result.push_back(
                squeeze(sublabel.data(), sublabel.size()));
        }
        return result;
    }

    std::vector<uint64_t> squeeze_n(const std::string& label, int n) {
        return squeeze_n(
            reinterpret_cast<const uint8_t*>(label.data()),
            label.size(), n);
    }

    // ----- state access -----
    const std::array<uint8_t, 32>& get_state() const { return state_; }
};

}  // namespace zkllm
