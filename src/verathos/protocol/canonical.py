"""Canonical JSON encoding and digests.

Every signed or hashed object is encoded the same way: sorted keys, no insignificant whitespace, UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode ``value`` as canonical JSON bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256_hex(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    """Hex SHA-256 of the canonical JSON encoding of ``value``."""
    return sha256_hex(canonical_json(value))
