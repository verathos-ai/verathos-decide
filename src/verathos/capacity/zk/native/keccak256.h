// Fiat-Shamir transcript over SHA3-256 (OpenSSL EVP), bit-identical to the
// Python Transcript in verathos.capacity.zk.transcript.

#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>
#include <openssl/evp.h>

namespace verathos {

static constexpr uint64_t TRANSCRIPT_MERSENNE_P = (1ULL << 61) - 1;

// --------------------------------------------------------------------------
//  Low-level helpers
// --------------------------------------------------------------------------

// 8-byte little-endian encoding.
inline void pack_le64(uint8_t out[8], uint64_t value) {
    for (int i = 0; i < 8; i++) {
        out[i] = static_cast<uint8_t>(value & 0xFF);
        value >>= 8;
    }
}

// Inverse of pack_le64.
inline uint64_t read_le64(const uint8_t* data) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; i--) {
        result = (result << 8) | data[i];
    }
    return result;
}

// SHA3-256 of the concatenated parts.
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
//  NativeTranscript
// --------------------------------------------------------------------------

class NativeTranscript {
    std::array<uint8_t, 32> state_;

public:
    explicit NativeTranscript(const std::array<uint8_t, 32>& initial_state)
        : state_(initial_state) {}

    // state = sha3_256(state || label || le64(value))
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

    // state = sha3_256(state || label || le64(len(data)) || data)
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

    // digest = sha3_256(state || "squeeze:" || label)
    // state  = sha3_256(state || digest)
    // return le64(digest[:8]) % p
    uint64_t squeeze(const uint8_t* label, size_t label_len) {
        static const uint8_t squeeze_prefix[] = {
            's', 'q', 'u', 'e', 'e', 'z', 'e', ':'
        };

        auto digest = sha3_256({
            {state_.data(), 32},
            {squeeze_prefix, 8},
            {label, label_len},
        });

        state_ = sha3_256({
            {state_.data(), 32},
            {digest.data(), 32},
        });

        uint64_t raw = read_le64(digest.data());
        return raw % TRANSCRIPT_MERSENNE_P;
    }

    uint64_t squeeze(const std::string& label) {
        return squeeze(reinterpret_cast<const uint8_t*>(label.data()),
                       label.size());
    }

    // squeeze(label || ":" || le64(i)) for i in 0..n-1
    std::vector<uint64_t> squeeze_n(
        const uint8_t* label, size_t label_len, int n
    ) {
        std::vector<uint64_t> result;
        result.reserve(n);

        for (int i = 0; i < n; i++) {
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

    const std::array<uint8_t, 32>& get_state() const { return state_; }
};

}  // namespace verathos
