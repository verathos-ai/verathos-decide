from __future__ import annotations

import pytest
from bittensor_wallet import Keypair

from verathos.protocol.manifest import (
    Endpoint,
    Hardware,
    Manifest,
    ManifestRejected,
    SignedManifest,
    accept,
    axon_base,
    commitment_value,
    committed_digest,
    endpoint_url,
)

CATALOG = "0" * 64
AXON = axon_base("203.0.113.5", 8091)


def make(keypair: Keypair, **overrides) -> Manifest:
    fields = dict(
        netuid=405,
        mechid=1,
        hotkey=keypair.ss58_address,
        catalog_digest=CATALOG,
        endpoints=[
            Endpoint(endpoint_id="gpu0", url=endpoint_url(AXON, "gpu0"), model_id="kev-4b",
                     hardware=Hardware(kind="cuda", device_id="GPU-aaaa"), max_concurrency=64),
            Endpoint(endpoint_id="gpu1", url=endpoint_url(AXON, "gpu1"), model_id="kev-9b",
                     hardware=Hardware(kind="cuda", device_id="GPU-bbbb"), max_concurrency=32),
        ],
    )
    fields.update(overrides)
    return Manifest(**fields)


def check(document, keypair: Keypair, *, commitment: str, hotkey: str | None = None, mechid: int = 1,
          axon: str = AXON, catalogs=(CATALOG,)) -> Manifest:
    return accept(document, hotkey=hotkey or keypair.ss58_address, netuid=405, mechid=mechid, axon=axon,
                  commitment=commitment, catalog_digests=catalogs)


@pytest.fixture()
def keypair() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def test_accepts_signed_and_committed(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    assert committed_digest(commitment) == signed.manifest.digest()
    assert check(signed.model_dump(mode="json"), keypair, commitment=commitment) == signed.manifest


def test_digest_and_signature_cover_fields_this_code_does_not_know(keypair: Keypair) -> None:
    """A newer miner may add manifest fields; older validators still verify the document as received."""
    from verathos.protocol import signing
    from verathos.protocol.canonical import digest
    from verathos.protocol.manifest import MANIFEST_DOMAIN

    raw = {**make(keypair).model_dump(mode="json"), "future_field": [1, 2]}
    document = {"manifest": raw, "signature": signing.sign(keypair, MANIFEST_DOMAIN, raw)}
    assert check(document, keypair, commitment=commitment_value(digest(raw))).endpoints[0].endpoint_id == "gpu0"


@pytest.mark.parametrize("document", [[], "x", {"manifest": []}, {"manifest": {}, "signature": 5},
                                      {"manifest": {"a": float("nan")}, "signature": "00" * 64}])
def test_malformed_documents_are_rejected_not_raised(keypair: Keypair, document) -> None:
    with pytest.raises(ManifestRejected):
        check(document, keypair, commitment=commitment_value("0" * 64))


def test_rejects_tampered_manifest(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    document = signed.model_dump(mode="json")
    document["manifest"]["endpoints"][0]["max_concurrency"] = 4096
    with pytest.raises(ManifestRejected, match="signature"):
        check(document, keypair, commitment=commitment_value(signed.manifest.digest()))


def test_rejects_uncommitted_version(keypair: Keypair) -> None:
    old = SignedManifest.create(make(keypair), keypair)
    new = SignedManifest.create(make(keypair, endpoints=make(keypair).endpoints[:1]), keypair)
    with pytest.raises(ManifestRejected, match="commitment"):
        check(new.model_dump(mode="json"), keypair, commitment=commitment_value(old.manifest.digest()))


def test_rejects_other_hotkey_and_mechanism(keypair: Keypair) -> None:
    other = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    with pytest.raises(ManifestRejected, match="hotkey"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment, hotkey=other.ss58_address)
    with pytest.raises(ManifestRejected, match="mechanism"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment, mechid=2)
    with pytest.raises(ValueError):
        SignedManifest.create(make(keypair), other)


def test_endpoints_must_be_on_the_hotkeys_axon(keypair: Keypair) -> None:
    """Endpoint URLs feed the audit's grouping, so a manifest cannot claim hosts other than its own axon."""
    signed = SignedManifest.create(make(keypair), keypair)
    with pytest.raises(ManifestRejected, match="axon"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()),
              axon=axon_base("198.51.100.9", 8091))


def test_one_device_per_endpoint(keypair: Keypair) -> None:
    dup = make(keypair).endpoints
    dup[1].hardware = Hardware(kind="cuda", device_id="GPU-aaaa")
    with pytest.raises(ValueError, match="hardware device"):
        make(keypair, endpoints=dup)


def test_accepts_previous_catalog_during_transition(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    with pytest.raises(ManifestRejected, match="another catalog"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment, catalogs=("1" * 64,))
    assert check(signed.model_dump(mode="json"), keypair, commitment=commitment, catalogs=("1" * 64, CATALOG))
