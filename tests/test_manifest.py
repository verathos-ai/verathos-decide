from __future__ import annotations

import pytest
from bittensor_wallet import Keypair

from tests.tlsfake import pin
from verathos.protocol.manifest import (
    Endpoint,
    Manifest,
    ManifestRejected,
    SignedManifest,
    accept,
    axon_base,
    commitment_value,
    committed_digest,
    endpoint_host,
    endpoint_url,
)

pytestmark = pytest.mark.usefixtures("lab_network")

CATALOG = "0" * 64
AXON = axon_base("203.0.113.5", 8091)
PIN = pin("axon")


def make(keypair: Keypair, **overrides) -> Manifest:
    fields = dict(
        netuid=405,
        mechid=1,
        hotkey=keypair.ss58_address,
        catalog_digest=CATALOG,
        endpoints=[
            Endpoint(endpoint_id="gpu0", url=endpoint_url(AXON, "gpu0"), model_id="kev-4b", capacity_rps=48.5),
            Endpoint(endpoint_id="gpu1", url=endpoint_url(AXON, "gpu1"), model_id="kev-9b", capacity_rps=21.0),
        ],
        tls={AXON: PIN},
    )
    fields.update(overrides)
    return Manifest(**fields)


def check(document, keypair: Keypair, *, commitment: str, hotkey: str | None = None, catalogs=None,
          seen: str | None = PIN) -> Manifest:
    """Accept as a validator that fetched ``document`` from the axon, which presented the certificate ``seen``."""
    return accept(document, hotkey=hotkey or keypair.ss58_address, netuid=405, commitment=commitment,
                  catalogs={CATALOG: 1} if catalogs is None else catalogs, axon=AXON, seen=seen)


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
    document["manifest"]["endpoints"][0]["capacity_rps"] = 4096.0
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
    with pytest.raises(ValueError):
        SignedManifest.create(make(keypair), other)


def test_a_manifest_is_for_the_mechanism_of_the_catalog_it_is_built_for(keypair: Keypair) -> None:
    """While validators accept the previous catalog too, each catalog's manifests are for its own mechanism."""
    previous = "1" * 64
    for catalog_digest, mechid in ((CATALOG, 0), (previous, 1)):
        signed = SignedManifest.create(make(keypair, catalog_digest=catalog_digest, mechid=mechid), keypair)
        assert check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()),
                     catalogs={CATALOG: 0, previous: 1}) == signed.manifest
    for catalog_digest, mechid in ((CATALOG, 1), (previous, 0)):
        signed = SignedManifest.create(make(keypair, catalog_digest=catalog_digest, mechid=mechid), keypair)
        with pytest.raises(ManifestRejected, match=f"manifest is for mechanism {mechid}, not {1 - mechid}"):
            check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()),
                  catalogs={CATALOG: 0, previous: 1})
    signed = SignedManifest.create(make(keypair, catalog_digest="2" * 64), keypair)
    with pytest.raises(ManifestRejected, match="another catalog"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()),
              catalogs={CATALOG: 1, previous: 1})


def test_endpoints_may_be_on_several_machines(keypair: Keypair) -> None:
    """One hotkey, GPUs on several machines: each endpoint is listed at its own machine's address, and the manifest
    names each machine's certificate."""
    worker = axon_base("198.51.100.9", 20017)
    endpoints = [*make(keypair).endpoints,
                 Endpoint(endpoint_id="w1-gpu0", url=endpoint_url(worker, "w1-gpu0"), model_id="kev-4b",
                          capacity_rps=30.0)]
    signed = SignedManifest.create(make(keypair, endpoints=endpoints, tls={AXON: PIN, worker: pin("worker")}),
                                   keypair)
    accepted = check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()))
    assert {endpoint_host(e.url, e.endpoint_id) for e in accepted.endpoints} == {AXON, worker}
    assert accepted.pin(worker) == pin("worker") and accepted.pin(AXON) == PIN
    signed = SignedManifest.create(make(keypair, endpoints=endpoints), keypair)
    with pytest.raises(ManifestRejected, match=f"w1-gpu0: the manifest names no certificate for {worker}"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()))


def test_the_axon_must_present_the_certificate_its_manifest_names(keypair: Keypair) -> None:
    """The manifest is fetched accepting any certificate; it is accepted only if the axon presented the one it
    names, so every later connection to the axon is pinned to the signed, committed pin."""
    signed = SignedManifest.create(make(keypair), keypair)
    document, commitment = signed.model_dump(mode="json"), commitment_value(signed.manifest.digest())
    for seen in (pin("someone else"), None):
        with pytest.raises(ManifestRejected, match="the axon's TLS certificate"):
            check(document, keypair, commitment=commitment, seen=seen)
    signed = SignedManifest.create(make(keypair, endpoints=[], tls={axon_base("198.51.100.9", 1): PIN}), keypair)
    with pytest.raises(ManifestRejected, match="names no certificate for its axon"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()))


@pytest.mark.parametrize("base", ["http://203.0.113.5:8091", "https://203.0.113.5:8091/", "https://172.16.0.1:1",
                                  "https://miner.example:8091", "https://[2001:DB8::1]:8091"])
def test_certificates_are_named_for_public_base_urls_only(keypair: Keypair, base: str) -> None:
    signed = SignedManifest.create(make(keypair, tls={AXON: PIN, base: PIN}), keypair)
    with pytest.raises(ManifestRejected, match="names a certificate for"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment_value(signed.manifest.digest()))


@pytest.mark.parametrize("change", [lambda raw: raw.pop("tls"), lambda raw: raw.update(tls={AXON: "AB" * 32}),
                                    lambda raw: raw.update(tls={AXON: "ab" * 31})])
def test_a_manifest_without_valid_pins_is_malformed(keypair: Keypair, change) -> None:
    """Endpoints over plain http without pins are not a manifest of this protocol."""
    from verathos.protocol import signing
    from verathos.protocol.canonical import digest
    from verathos.protocol.manifest import MANIFEST_DOMAIN

    raw = make(keypair).model_dump(mode="json")
    change(raw)
    document = {"manifest": raw, "signature": signing.sign(keypair, MANIFEST_DOMAIN, raw)}
    with pytest.raises(ManifestRejected, match="malformed manifest"):
        check(document, keypair, commitment=commitment_value(digest(raw)))


@pytest.mark.parametrize("url", [
    "http://203.0.113.5:8091/e/gpu0", "https://203.0.113.5/e/gpu0", "https://203.0.113.5:8091/e/gpu0/",
    "https://203.0.113.5:8091/e/gpu1", "https://203.0.113.5:8091/e/gpu0?x=1", "https://u@203.0.113.5:8091/e/gpu0",
    "https://miner.example:8091/e/gpu0", "https://172.16.0.1:8091/e/gpu0", "https://203.0.113.5:0/e/gpu0",
    "https://203.0.113.5:65536/e/gpu0", "https://203.0.113.5:08091/e/gpu0", "https://[2001:DB8::1]:8091/e/gpu0",
    "https://[2001:db8:0::1]:8091/e/gpu0", "https://203.0.113.5:8091/x/e/gpu0"])
def test_a_manifest_with_any_other_endpoint_url_is_rejected(keypair: Keypair, url: str) -> None:
    """Exactly https://<public IP>:<port>/e/<endpoint_id>, as the chain writes the address; nothing else."""
    if url.startswith("http://"):
        with pytest.raises(ValueError, match="https"):
            Endpoint(endpoint_id="gpu0", url=url, model_id="kev-4b", capacity_rps=1.0)
        endpoint = Endpoint.model_construct(endpoint_id="gpu0", url=url, model_id="kev-4b", tier="open",
                                            capacity_rps=1.0)
    else:
        endpoint = Endpoint(endpoint_id="gpu0", url=url, model_id="kev-4b", capacity_rps=1.0)
    signed = SignedManifest.create(make(keypair).model_copy(update={"endpoints": [endpoint]}), keypair)
    with pytest.raises(ManifestRejected, match="URL|malformed"):
        check(signed.model_dump(mode="json", warnings=False), keypair,
              commitment=commitment_value(signed.manifest.digest()))


def test_the_hosts_of_a_manifest_are_bounded(keypair: Keypair) -> None:
    from verathos.protocol import manifest

    bases = [axon_base(f"198.51.100.{i}", 80) for i in range(1, 4)]
    endpoints = [Endpoint(endpoint_id=f"gpu{i}", url=endpoint_url(base, f"gpu{i}"), model_id="kev-4b",
                          capacity_rps=1.0) for i, base in enumerate(bases, 1)]
    signed = SignedManifest.create(make(keypair, endpoints=endpoints, tls={b: PIN for b in [AXON, *bases]}), keypair)
    commitment = commitment_value(signed.manifest.digest())
    assert len(check(signed.model_dump(mode="json"), keypair, commitment=commitment).tls) == 4
    with pytest.raises(ValueError):
        make(keypair, tls={axon_base(f"198.51.100.{i}", 80): PIN for i in range(manifest.MAX_HOSTS + 1)})


@pytest.mark.parametrize("capacity", [-1.0, 100_001.0, float("nan")])
def test_declared_capacity_must_be_in_range(capacity: float) -> None:
    with pytest.raises(ValueError):
        Endpoint(endpoint_id="gpu0", url=endpoint_url(AXON, "gpu0"), model_id="kev-4b", capacity_rps=capacity)


def test_endpoint_ids_are_unique(keypair: Keypair) -> None:
    endpoints = make(keypair).endpoints
    with pytest.raises(ValueError, match="unique"):
        make(keypair, endpoints=[endpoints[0], endpoints[0]])


def test_accepts_previous_catalog_during_transition(keypair: Keypair) -> None:
    signed = SignedManifest.create(make(keypair), keypair)
    commitment = commitment_value(signed.manifest.digest())
    with pytest.raises(ManifestRejected, match="another catalog"):
        check(signed.model_dump(mode="json"), keypair, commitment=commitment, catalogs={"1" * 64: 1})
    assert check(signed.model_dump(mode="json"), keypair, commitment=commitment, catalogs={"1" * 64: 1, CATALOG: 1})


def test_ipv6_public_hosts_are_normalised_as_the_chain_returns_them() -> None:
    from verathos.protocol.manifest import public_ip

    assert public_ip("2001:DB8:0:0::1") == "2001:db8::1"
    assert public_ip("203.0.113.5") == "203.0.113.5"
    assert axon_base("2001:db8::1", 8091) == "https://[2001:db8::1]:8091"
