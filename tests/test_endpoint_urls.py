"""Where the network connects: an endpoint URL names a public unicast address, with no allowance for tests."""

from __future__ import annotations

import ipaddress

import pytest
from bittensor_wallet import Keypair

from verathos.protocol.gateway import Route, Routes
from verathos.protocol.manifest import (
    Endpoint,
    Manifest,
    ManifestRejected,
    SignedManifest,
    accept,
    commitment_value,
    endpoint_host,
    is_public,
    public_unicast,
)


@pytest.mark.parametrize("address", ["8.8.8.8", "1.2.3.4", "9.9.9.9", "2606:4700::1111", "2a01:4f8::1"])
def test_public_unicast_addresses(address: str) -> None:
    assert public_unicast(ipaddress.ip_address(address)) and is_public(address)


@pytest.mark.parametrize("address", [
    "10.0.0.1", "172.16.0.1", "192.168.1.1", "100.64.0.1",  # private, shared (CGNAT)
    "127.0.0.1", "::1",  # loopback
    "169.254.1.1", "fe80::1",  # link-local
    "224.0.0.1", "239.1.2.3", "ff0e::1",  # multicast
    "240.0.0.1", "255.255.255.255", "::808:808",  # reserved
    "0.0.0.0", "::",  # unspecified
    "192.0.2.1", "198.51.100.7", "203.0.113.5", "2001:db8::1", "fc00::1", "198.18.0.1",  # documentation, ULA, bench
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:8.8.8.8",  # IPv4-mapped, of private and public addresses alike
    "64:ff9b::a00:1", "64:ff9b::808:808", "2002:a00:1::1",  # NAT64 and 6to4 forms of IPv4 addresses
])
def test_non_public_addresses(address: str) -> None:
    assert not public_unicast(ipaddress.ip_address(address)) and not is_public(address)


@pytest.mark.parametrize("value", ["", "localhost", "miner.example", "8.8.8.8:80", "[2606:4700::1111]"])
def test_only_address_literals_are_public(value: str) -> None:
    assert not is_public(value)


def test_endpoint_hosts() -> None:
    assert endpoint_host("http://8.8.8.8:8091/e/gpu0", "gpu0") == "http://8.8.8.8:8091"
    assert endpoint_host("http://[2606:4700::1111]:1/e/w-gpu7", "w-gpu7") == "http://[2606:4700::1111]:1"
    for url in ("http://10.0.0.1:8091/e/gpu0", "http://127.0.0.1:8091/e/gpu0", "http://[::ffff:8.8.8.8]:80/e/gpu0",
                "http://[::ffff:808:808]:80/e/gpu0", "http://203.0.113.5:8091/e/gpu0", "http://8.8.8.8:8091/e/gpu1"):
        assert endpoint_host(url, "gpu0") is None, url


def test_a_manifest_naming_a_non_public_host_is_rejected() -> None:
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    for host in ("8.8.8.8", "10.0.0.1"):
        signed = SignedManifest.create(Manifest(
            netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest="0" * 64, endpoints=[
                Endpoint(endpoint_id="gpu0", url="http://8.8.8.8:8091/e/gpu0", model_id="kev-4b", capacity_rps=1.0),
                Endpoint(endpoint_id="gpu1", url=f"http://{host}:8091/e/gpu1", model_id="kev-4b",
                         capacity_rps=1.0)]), keypair)
        document = signed.model_dump(mode="json")
        commitment = commitment_value(signed.manifest.digest())
        if host == "8.8.8.8":
            accept(document, hotkey=keypair.ss58_address, netuid=405, mechid=1, commitment=commitment,
                   catalog_digests={"0" * 64})
            continue
        with pytest.raises(ManifestRejected, match="gpu1: its URL is not"):
            accept(document, hotkey=keypair.ss58_address, netuid=405, mechid=1, commitment=commitment,
                   catalog_digests={"0" * 64})


def test_the_gateway_routes_to_public_hosts_only() -> None:
    """Routes on any public machine of a miner are used; not even a table the auditor signed can send the gateway
    into a private network."""
    from verathos.gateway.router import Router  # after the chain SDK's import, which silences the loggers there are

    router = Router(share=1.0, max_latency_s=2.0)
    urls = {"a": "http://8.8.8.8:8091/e/gpu0", "b": "http://1.2.3.4:20017/e/gpu0", "c": "http://10.0.0.1:8091/e/gpu0",
            "d": "http://127.0.0.1:8091/e/gpu0", "e": "http://[::ffff:10.0.0.1]:8091/e/gpu0",
            "f": "http://metadata.internal/e/gpu0"}
    router.update(Routes(netuid=405, auditor="a", epoch=1, issued_ms=1, catalog_digest="0" * 64, routes=[
        Route(hotkey=h, endpoint_id="gpu0", model_id="kev-4b", url=url, credited_rps=10.0, latency_ms=100.0)
        for h, url in urls.items()]))
    assert {e.route.hotkey for e in router._pools["kev-4b"].endpoints} == {"a", "b"}
