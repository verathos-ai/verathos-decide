from __future__ import annotations

import time

import pytest
from bittensor_wallet import Keypair

from decide.protocol.requests import NonceCache, RequestRejected, sign_request, verify_request


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def test_valid_request_verifies_once() -> None:
    caller, miner = kp(), kp()
    body = b'{"x":1}'
    headers = sign_request(caller, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address)
    nonces = NonceCache()
    who = verify_request(headers, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address, nonces)
    assert who.hotkey == caller.ss58_address
    with pytest.raises(RequestRejected, match="replayed"):
        verify_request(headers, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address, nonces)


@pytest.mark.parametrize("mutate,reason", [
    (lambda a: {**a, "body": b'{"x":2}'}, "bad signature"),
    (lambda a: {**a, "path": "/e/gpu1/v1/systemone"}, "bad signature"),
    (lambda a: {**a, "receiver": kp().ss58_address}, "another receiver"),
])
def test_tampering_is_rejected(mutate, reason) -> None:
    caller, miner = kp(), kp()
    base = {"body": b'{"x":1}', "path": "/e/gpu0/v1/systemone", "receiver": miner.ss58_address}
    headers = sign_request(caller, "POST", base["path"], base["body"], miner.ss58_address)
    m = mutate(base)
    with pytest.raises(RequestRejected, match=reason):
        verify_request(headers, "POST", m["path"], m["body"], m["receiver"], NonceCache())


def test_stale_request_is_rejected() -> None:
    caller, miner = kp(), kp()
    old = int(time.time() * 1000) - 120_000
    headers = sign_request(caller, "POST", "/p", b"", miner.ss58_address, timestamp_ms=old)
    with pytest.raises(RequestRejected, match="window"):
        verify_request(headers, "POST", "/p", b"", miner.ss58_address, NonceCache())


def test_missing_headers_are_rejected() -> None:
    with pytest.raises(RequestRejected, match="missing"):
        verify_request({}, "POST", "/p", b"", kp().ss58_address, NonceCache())
