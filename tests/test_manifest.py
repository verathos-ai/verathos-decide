from __future__ import annotations

import pytest
from bittensor_wallet import Keypair

from decide.protocol.manifest import (
    Endpoint,
    Hardware,
    Manifest,
    ManifestRejected,
    SignedManifest,
    accept,
    commitment_value,
    committed_digest,
)

CATALOG = "0" * 64


def make(keypair: Keypair, **overrides) -> Manifest:
    fields = dict(
        netuid=405,
        mechid=1,
        hotkey=keypair.ss58_address,
        seq=1,
        issued_block=100,
        catalog_digest=CATALOG,
        endpoints=[
            Endpoint(endpoint_id="gpu0", url="https://203.0.113.5:8443", model_id="kev-4b",
                     hardware=Hardware(kind="cuda", device_id="GPU-aaaa"), max_concurrency=64),
            Endpoint(endpoint_id="gpu1", url="https://203.0.113.5:8444", model_id="kev-9b",
                     hardware=Hardware(kind="cuda", device_id="GPU-bbbb"), max_concurrency=32),
        ],
    )
    fields.update(overrides)
    return Manifest(**fields)


@pytest.fixture()
def keypair() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def test_accepts_signed_and_committed(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    assert committed_digest(commitment) == signed.manifest.digest()
    assert accept(signed, hotkey=keypair.ss58_address, netuid=405, commitment=commitment) == signed.manifest


def test_round_trip_keeps_digest_and_signature(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    again = SignedManifest.model_validate_json(signed.model_dump_json())
    assert again.manifest.digest() == signed.manifest.digest()
    assert again.signature_valid()


def test_rejects_tampered_manifest(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    tampered = signed.model_copy(deep=True)
    tampered.manifest.endpoints[0].url = "https://198.51.100.9:8443"
    with pytest.raises(ManifestRejected, match="signature"):
        accept(tampered, hotkey=keypair.ss58_address, netuid=405, commitment=commitment)


def test_rejects_uncommitted_version(keypair: Keypair) -> None:
    old = SignedManifest.create(make(keypair, seq=1), keypair)
    new = SignedManifest.create(make(keypair, seq=2), keypair)
    with pytest.raises(ManifestRejected, match="commitment"):
        accept(new, hotkey=keypair.ss58_address, netuid=405, commitment=commitment_value(old.manifest.digest()))


def test_rejects_other_hotkey(keypair: Keypair) -> None:
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    with pytest.raises(ManifestRejected, match="hotkey"):
        accept(signed, hotkey=other.ss58_address, netuid=405, commitment=commitment)
    with pytest.raises(ValueError):
        SignedManifest.create(make(keypair), other)


def test_one_device_per_endpoint(keypair: Keypair) -> None:
    dup = make(keypair).endpoints
    dup[1].hardware = Hardware(kind="cuda", device_id="GPU-aaaa")
    with pytest.raises(ValueError, match="hardware device"):
        make(keypair, endpoints=dup)
