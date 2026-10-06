"""
Fiat-Shamir transcript for non-interactive proofs.

Uses Keccak256 (SHA-3) for compatibility with EVM and
to match the original zkgemm protocol design.
"""

from typing import List
import hashlib
import struct

from verathos.capacity.zk.crypto.field import mod_p


class Transcript:
    """
    Running transcript for Fiat-Shamir transform.

    Absorbs labeled values and squeezes pseudorandom challenges.
    Uses Keccak256 for hashing.
    """

    def __init__(self, label: bytes = b"VERILLM_TRANSCRIPT_V1"):
        """Initialize transcript with domain label."""
        # Use sha3_256 (Keccak256)
        self._state = hashlib.sha3_256(label).digest()

    def absorb(self, label: bytes, value: int) -> None:
        """
        Absorb a labeled integer value into the transcript.

        Args:
            label: Domain label for this value
            value: Integer value to absorb
        """
        # Pack value as little-endian 64-bit
        value_bytes = struct.pack("<Q", value % (1 << 64))

        # Update state
        h = hashlib.sha3_256()
        h.update(self._state)
        h.update(label)
        h.update(value_bytes)
        self._state = h.digest()

    def absorb_bytes(self, label: bytes, data: bytes) -> None:
        """
        Absorb labeled bytes into the transcript.

        Args:
            label: Domain label
            data: Raw bytes to absorb
        """
        h = hashlib.sha3_256()
        h.update(self._state)
        h.update(label)
        h.update(struct.pack("<Q", len(data)))  # Length prefix
        h.update(data)
        self._state = h.digest()

    def absorb_many(self, label: bytes, values: List[int]) -> None:
        """
        Absorb multiple values with a single label.

        Args:
            label: Domain label
            values: List of integer values
        """
        h = hashlib.sha3_256()
        h.update(self._state)
        h.update(label)
        h.update(struct.pack("<Q", len(values)))  # Count prefix

        for v in values:
            h.update(struct.pack("<Q", v % (1 << 64)))

        self._state = h.digest()

    def squeeze(self, label: bytes) -> int:
        """
        Squeeze a pseudorandom field element from the transcript.

        Args:
            label: Domain label for this challenge

        Returns:
            Pseudorandom element in F_p
        """
        # Hash with squeeze label
        h = hashlib.sha3_256()
        h.update(self._state)
        h.update(b"squeeze:")
        h.update(label)
        digest = h.digest()

        # Update state with digest for chaining
        h2 = hashlib.sha3_256()
        h2.update(self._state)
        h2.update(digest)
        self._state = h2.digest()

        # Extract field element from first 8 bytes
        raw = int.from_bytes(digest[:8], "little")
        return mod_p(raw)

    def squeeze_n(self, label: bytes, n: int) -> List[int]:
        """
        Squeeze n pseudorandom field elements.

        Args:
            label: Domain label
            n: Number of elements to squeeze

        Returns:
            List of n field elements
        """
        result = []
        for i in range(n):
            # Each squeeze gets unique sublabel
            sublabel = label + b":" + struct.pack("<Q", i)
            result.append(self.squeeze(sublabel))
        return result

    def fork(self, label: bytes) -> "Transcript":
        """
        Create a forked transcript with current state.

        Useful for sub-proofs that should not affect main transcript.

        Args:
            label: Fork label

        Returns:
            New Transcript initialized from current state
        """
        forked = Transcript.__new__(Transcript)

        h = hashlib.sha3_256()
        h.update(self._state)
        h.update(b"fork:")
        h.update(label)
        forked._state = h.digest()

        return forked

    def get_state(self) -> bytes:
        """Get current transcript state (for debugging/testing)."""
        return self._state

    def clone(self) -> "Transcript":
        """Create exact clone of current transcript."""
        cloned = Transcript.__new__(Transcript)
        cloned._state = self._state
        return cloned


def create_transcript(label: str = "verallm") -> Transcript:
    """Create a new transcript with the given label."""
    return Transcript(f"VERILLM_{label.upper()}_V1".encode())
