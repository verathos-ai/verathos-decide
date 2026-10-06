"""SHA3-256 Fiat-Shamir transcript over F_p (p = 2^61 - 1); mirrored by the native block prover."""

import hashlib
import struct

from verathos.capacity.zk.field import mod_p


class Transcript:
    """
    Running transcript for Fiat-Shamir transform.

    Absorbs labeled values and squeezes pseudorandom challenges.
    """

    def __init__(self, label: bytes):
        """Initialize transcript with domain label."""
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

    def squeeze_n(self, label: bytes, n: int) -> list[int]:
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

    def get_state(self) -> bytes:
        return self._state

    def set_state(self, state: bytes) -> None:
        """Adopt the state reached by the native prover after it extended this transcript."""
        self._state = state

