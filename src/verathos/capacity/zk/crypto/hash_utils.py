"""
Centralized hash function for ZkLLM.

Uses BLAKE3 (5-10x faster than SHA256, SIMD + Rayon-backed).
blake3 is a core dependency — the SHA256 fallback is removed because
a silent hash mismatch between prover and verifier causes proof
verification failures that are extremely hard to diagnose.
"""

import blake3 as _blake3


def hash_digest(data) -> bytes:
    """Compute 32-byte hash digest of data (bytes or buffer)."""
    return _blake3.blake3(data).digest()


def hash_update_digest(*parts) -> bytes:
    """Compute hash digest of concatenated parts."""
    h = _blake3.blake3()
    for part in parts:
        h.update(part)
    return h.digest()


HASH_NAME = "blake3"
