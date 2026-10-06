"""sr25519 signatures by Bittensor hotkeys.

Signatures cover a domain tag plus the canonical JSON of the signed object, so a signature for one object type
can never be replayed as another.
"""

from __future__ import annotations

from typing import Any

from bittensor_wallet import Keypair

from verathos.protocol.canonical import canonical_json


def signing_payload(domain: str, value: Any) -> bytes:
    return domain.encode() + b"\n" + canonical_json(value)


def sign(keypair: Keypair, domain: str, value: Any) -> str:
    """Hex signature of ``value`` under ``domain``."""
    return keypair.sign(signing_payload(domain, value)).hex()


def verify(ss58_address: Any, domain: str, value: Any, signature_hex: Any) -> bool:
    """Whether ``signature_hex`` is ``ss58_address``'s signature of ``value`` under ``domain``. Never raises: any
    malformed address, signature or value is simply not a valid signature."""
    if not isinstance(ss58_address, str) or not isinstance(signature_hex, str):
        return False
    try:
        signature = bytes.fromhex(signature_hex)
        if len(signature) != 64:
            return False
        return Keypair(ss58_address=ss58_address).verify(signing_payload(domain, value), signature)
    except (ValueError, TypeError):
        return False
