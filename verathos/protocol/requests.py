"""Signed HTTP requests between neurons (validator or gateway -> miner).

The caller signs, with its sr25519 hotkey:

    verathos/request/v1 \\n METHOD \\n PATH \\n sha256(body) \\n timestamp_ms \\n nonce \\n receiver_hotkey

Binding the receiver's hotkey means a request signed for one miner cannot be replayed against another; the nonce
plus the timestamp window means it cannot be replayed against the same miner.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

from bittensor_wallet import Keypair

DOMAIN = "verathos/request/v1"
HDR_HOTKEY = "X-Verathos-Hotkey"
HDR_SIGNATURE = "X-Verathos-Signature"
HDR_TIMESTAMP = "X-Verathos-Timestamp"
HDR_NONCE = "X-Verathos-Nonce"
HDR_RECEIVER = "X-Verathos-Receiver"
MAX_SKEW_MS = 60_000


def signing_message(method: str, path: str, body: bytes, timestamp_ms: str, nonce: str, receiver: str) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return "\n".join([DOMAIN, method.upper(), path, body_hash, timestamp_ms, nonce, receiver]).encode()


def sign_request(keypair: Keypair, method: str, path: str, body: bytes, receiver: str,
                 *, timestamp_ms: int | None = None) -> dict[str, str]:
    """Headers authenticating ``keypair`` as the caller of this request to ``receiver``."""
    ts = str(int(time.time() * 1000) if timestamp_ms is None else timestamp_ms)
    nonce = secrets.token_hex(16)
    signature = keypair.sign(signing_message(method, path, body, ts, nonce, receiver)).hex()
    return {HDR_HOTKEY: keypair.ss58_address, HDR_SIGNATURE: signature, HDR_TIMESTAMP: ts, HDR_NONCE: nonce,
            HDR_RECEIVER: receiver}


class NonceCache:
    """Nonces seen within the replay window."""

    def __init__(self, window_ms: int = 2 * MAX_SKEW_MS) -> None:
        self._window_ms = window_ms
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()

    def check_and_add(self, key: str, now_ms: int) -> bool:
        """True if ``key`` is new; records it. False if it was already seen inside the window."""
        with self._lock:
            if len(self._seen) > 100_000:
                cutoff = now_ms - self._window_ms
                self._seen = {k: t for k, t in self._seen.items() if t >= cutoff}
            if key in self._seen:
                return False
            self._seen[key] = now_ms
            return True


@dataclass(frozen=True)
class VerifiedCaller:
    hotkey: str
    nonce: str


class RequestRejected(Exception):
    pass


def verify_request(headers: dict[str, str], method: str, path: str, body: bytes, receiver: str,
                   nonces: NonceCache, *, now_ms: int | None = None) -> VerifiedCaller:
    """Check signature, receiver, freshness and replay. Raises RequestRejected with the reason."""
    h = {k.lower(): v for k, v in headers.items()}
    try:
        hotkey = h[HDR_HOTKEY.lower()]
        signature = h[HDR_SIGNATURE.lower()]
        ts = h[HDR_TIMESTAMP.lower()]
        nonce = h[HDR_NONCE.lower()]
        signed_for = h[HDR_RECEIVER.lower()]
    except KeyError as e:
        raise RequestRejected(f"missing header {e.args[0]}") from None
    if signed_for != receiver:
        raise RequestRejected("request was signed for another receiver")
    if not ts.isdecimal() or not (16 <= len(nonce) <= 64):
        raise RequestRejected("malformed timestamp or nonce")
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if abs(now - int(ts)) > MAX_SKEW_MS:
        raise RequestRejected("timestamp outside the allowed window")
    try:
        sig = bytes.fromhex(signature)
        ok = len(sig) == 64 and Keypair(ss58_address=hotkey).verify(signing_message(method, path, body, ts, nonce, receiver), sig)
    except Exception:
        ok = False
    if not ok:
        raise RequestRejected("bad signature")
    if not nonces.check_and_add(f"{hotkey}:{nonce}", now):
        raise RequestRejected("replayed request")
    return VerifiedCaller(hotkey=hotkey, nonce=nonce)
