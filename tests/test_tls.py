"""TLS between machines: identities kept in a state directory, pinned clients, and the services that use them over
real TLS on loopback ports."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import FastAPI, Request

from tests.mock_engine import MockEngine
from tests.tlsfake import free_port, serving
from verathos import catalog
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.protocol import tls
from verathos.protocol.http import get_json, request_document
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def counting_app() -> tuple[FastAPI, list[str]]:
    """An app that records the paths it was asked for."""
    app, asked = FastAPI(), []

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def anything(path: str, request: Request):
        asked.append(path)
        return {"path": path}

    return app, asked


def test_an_identity_is_created_once_kept_private_and_rotated_by_deleting_it(tmp_path) -> None:
    first = tls.identity(tmp_path / "state")
    assert first.pin == tls.stored_pin(tmp_path / "state") and tls.is_pin(first.pin)
    for path in (first.key_path, first.cert_path):
        assert path.stat().st_mode & 0o777 == 0o600
    assert tls.identity(tmp_path / "state") == first  # reused across restarts
    first.cert_path.unlink()  # a certificate is made again for the same key: the same pin
    assert tls.identity(tmp_path / "state").pin == first.pin
    first.cert_path.write_text("not a certificate")
    assert tls.identity(tmp_path / "state").pin == first.pin
    first.key_path.unlink()  # rotation: a new key, a new certificate, a new pin
    rotated = tls.identity(tmp_path / "state")
    assert rotated.pin != first.pin and tls.stored_pin(tmp_path / "state") == rotated.pin
    assert tls.stored_pin(tmp_path / "nothing") is None and not (tmp_path / "nothing").exists()
    assert tls.curl_pin("00" * 32) == "sha256//" + "A" * 43 + "="


def test_a_pinned_client_accepts_exactly_the_pinned_certificate(tmp_path) -> None:
    """Another self-signed certificate at the same address is refused before any byte of the request is sent; a
    connection checked for one pin never carries a request for another."""
    served, other = tls.identity(tmp_path / "served"), tls.identity(tmp_path / "other")
    app, asked = counting_app()

    async def run(base: str):
        async with tls.client() as client:
            ok = await client.get(base + "/a", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match=f"is not the pinned {other.pin}"):
                await client.get(base + "/b", extensions=tls.use(other.pin))
            again = await client.get(base + "/c", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match="no TLS pin"):
                await client.get(base + "/d")
            with pytest.raises(httpx.UnsupportedProtocol, match="only https"):
                await client.get(base.replace("https", "http") + "/e", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match="an IP address has no CA certificate"):
                await client.get(base + "/f", extensions=tls.use(tls.CA))
            document, seen = await request_document(client, "GET", base + "/g", limit=1024, timeout=5, tls=tls.ANY)
            return ok.json(), again.json(), document, seen

    with serving(app, served) as base:
        ok, again, document, seen = asyncio.run(run(base))
        assert get_json(base + "/h", limit=1024, timeout=5, tls=served.pin) == {"path": "h"}
        with pytest.raises(httpx.ConnectError, match="is not the pinned"):
            get_json(base + "/i", limit=1024, timeout=5, tls=other.pin)
    assert ok == {"path": "a"} and again == {"path": "c"} and document == {"path": "g"} and seen == served.pin
    assert asked == ["a", "c", "g", "h"]  # nothing of the refused requests reached the server


def test_a_client_never_takes_its_proxies_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert not tls.client()._mounts and not tls.sync_client()._mounts


@pytest.mark.usefixtures("lab_network")
def test_a_manifest_is_accepted_over_tls_only_from_an_axon_presenting_the_pin_it_names(tmp_path) -> None:
    """Discovery fetches the manifest accepting any certificate, accepts it only if the axon presented the pinned
    one, and challenges the endpoint with that pin."""
    from verathos.validator.audit import audit_epoch
    from verathos.validator.compare import Tolerance
    from verathos.validator.discovery import Identities, Neuron, discover

    miner, validator = kp(), kp()
    served, other = tls.identity(tmp_path / "served"), tls.identity(tmp_path / "other")

    def manifest_app(port: int, named: str):
        config = MinerConfig.model_validate({
            "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "127.0.0.1",
            "port": port, "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b",
                                         "engine_url": "http://127.0.0.1:1"}]})
        signed = SignedManifest.create(Manifest(
            netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=catalog.load().digest(),
            endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                                capacity_rps=40.0)], tls={config.base_url: named}), miner)
        app = create_app(config=config, keypair=miner, manifest=lambda: signed, engines={"gpu0": MockEngine("kev-4b")},
                         authorized=lambda hk: hk == validator.ss58_address, auditors=set)
        return app, commitment_value(signed.manifest.digest())

    async def epoch(port: int, commitment: str):
        async with tls.client() as client:
            miners = await discover([Neuron(1, miner.ss58_address, "127.0.0.1", port, commitment)], 405, client,
                                    {catalog.load().digest(): 1})
            results = await audit_epoch([e for m in miners for e in m.endpoints()], epoch=3, secret=b"k" * 32,
                                        keypair=validator, references={"kev-4b": MockEngine("kev-4b")},
                                        tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                        netuid=405, pool_size=2, per_endpoint=2, identities=Identities())
            return miners, results

    for named, accepted in ((served.pin, True), (other.pin, False)):
        port = free_port()
        app, commitment = manifest_app(port, named)
        with serving(app, served, port) as base:
            miners, results = asyncio.run(epoch(port, commitment))
        if accepted:
            (found,) = miners
            assert found.manifest.pin(base) == served.pin
            (result,) = results.values()
            assert result.identity_ok and result.answered == result.sent == 2
        else:
            assert miners == []


def test_the_gateway_reaches_a_miner_over_pinned_tls_and_blames_it_for_another_certificate(tmp_path) -> None:
    from tests.test_gateway import miner
    from verathos.gateway.upstream import Upstream, UpstreamFailure
    from verathos.protocol.api import SystemOneRequest
    from verathos.protocol.canonical import digest
    from verathos.protocol.gateway import Route

    gateway = kp()
    keypair, app = miner("203.0.113.5", gateway, MockEngine("kev-4b"))
    served, other = tls.identity(tmp_path / "served"), tls.identity(tmp_path / "other")
    request = {"model": "kev-4b", "state": "Hi, we were billed twice for March.",
               "questions": {"churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}}}
    body = json.dumps(request).encode()

    async def call(base: str, pin: str):
        route = Route(hotkey=keypair.ss58_address, endpoint_id="gpu0", model_id="kev-4b", url=f"{base}/e/gpu0",
                      tls=pin, credited_rps=10.0, latency_ms=100.0)
        async with tls.client() as client:
            return await Upstream(gateway, client, netuid=405).call(
                route, request, body, digest(request), SystemOneRequest.model_validate(request), 10.0,
                max_tokens=len(body))

    with serving(app, served) as base:
        exchange = asyncio.run(call(base, served.pin))
        with pytest.raises(UpstreamFailure) as failure:
            asyncio.run(call(base, other.pin))
    assert exchange.receipt["receipt"]["miner_hotkey"] == keypair.ss58_address
    assert (failure.value.kind, failure.value.blame) == ("connect", "miner")


def test_a_light_validator_reads_its_source_over_tls_with_the_catalogs_pin(tmp_path) -> None:
    """The auditor's network document over real TLS: accepted with the pin the catalog names for the auditor,
    refused with any other certificate."""
    from tests.test_light import Chain, config
    from tests.test_light import kp as light_kp
    from verathos.protocol.status import NetworkStatus
    from verathos.validator import light
    from verathos.validator.service import add_network_route

    owner, own = light_kp(), light_kp()
    served = tls.identity(tmp_path / "auditor")
    cat = catalog.load()
    block = 100 * cat.epoch_blocks + light.LightValidator.offset(cat) + 5

    class Board:
        revision = 1

        @staticmethod
        def versioned_network():
            body = NetworkStatus(epoch=100, epoch_blocks=cat.epoch_blocks, weights={3: 1.0}).model_dump(mode="json")
            return 1, json.dumps(body)

    app = FastAPI()
    add_network_route(app, Board(), owner, 405)  # signs what the board holds, as the auditor's service does
    set_weights: list[dict] = []
    with serving(app, served) as base:
        host, port = base.removeprefix("https://").rsplit(":", 1)
        for named, sets in ((served.pin, True), (tls.identity(tmp_path / "other").pin, False)):
            chain = Chain(owner.ss58_address, own.ss58_address, block)
            chain.neurons[owner.ss58_address].axon_info = SimpleNamespace(ip=host, port=int(port))
            followed = catalog.Catalog.parse({**cat.document(), "auditor": {"tls": named}})
            v = light.LightValidator(config(tmp_path), chain, own.ss58_address, SimpleNamespace(catalog=followed),
                                     set_weights=lambda w, mechid: set_weights.append(w) or True,
                                     state_path=tmp_path / f"light-{sets}.json")
            v.step()
            if sets:
                assert set_weights == [{3: 1.0}]
            else:
                assert len(set_weights) == 1 and "is not the pinned" in v.why


def test_catalog_mirrors_and_engines_off_this_machine_are_reached_over_tls_only() -> None:
    from verathos.miner.engine import HttpEngine, engine_tls
    from verathos.validator.config import ValidatorConfig

    assert engine_tls("http://127.0.0.1:8008", None) is None and engine_tls("http://localhost:8008", None) is None
    assert engine_tls("https://gpu.example.org", None) == tls.CA and engine_tls("https://1.2.3.4", "ab" * 32)
    for url in ("http://10.0.0.2:8008", "http://gpu.example.org", "ftp://1.2.3.4"):
        with pytest.raises(ValueError, match="https"):
            HttpEngine(url)
    base = {"netuid": 405, "wallet_name": "w", "hotkey_name": "h"}
    with pytest.raises(ValueError, match="catalog mirror"):
        ValidatorConfig.model_validate({**base, "catalog_mirrors": ["http://mirror.example/{digest}.json"]})
    with pytest.raises(ValueError, match="catalog mirror"):
        MinerConfig.model_validate({**base, "public_host": "1.2.3.4", "port": 1,
                                    "catalog_mirrors": ["http://mirror.example/{digest}.json"],
                                    "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b",
                                                   "engine_url": "http://127.0.0.1:1"}]})
    assert tls.authenticated("https://1.2.3.4:8091/v1/catalog/x") == tls.ANY  # a digest-checked catalog at an IP
    assert tls.authenticated("https://mirror.example/x") == tls.CA


def test_the_chain_is_reached_over_wss_or_on_this_machine_only() -> None:
    from verathos.gateway.config import GatewayConfig
    from verathos.install.common import network_and_netuid
    from verathos.validator.config import ValidatorConfig

    for network in ("finney", "test", "archive", "local", "wss://node.example.org:443", "wss://203.0.113.5:9944",
                    "ws://127.0.0.1:9944", "ws://localhost:9944"):
        assert tls.chain_network(network) == network
    for network in ("ws://203.0.113.5:9944", "ws://node.example.org", "http://127.0.0.1:9933", "https://node.example",
                    "203.0.113.5:9944", "wss://", "unknown-network"):
        with pytest.raises(ValueError, match="every connection between machines is TLS"):
            tls.chain_network(network)
    plain = "ws://203.0.113.5:9944"
    base = {"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "network": plain}
    with pytest.raises(ValueError, match="plain ws:// only on this machine"):
        ValidatorConfig.model_validate(base)
    with pytest.raises(ValueError, match="plain ws:// only on this machine"):
        GatewayConfig.model_validate({**base, "database_url": "sqlite+aiosqlite:///x.db"})
    with pytest.raises(ValueError, match="plain ws:// only on this machine"):
        MinerConfig.model_validate({**base, "public_host": "1.2.3.4", "port": 1,
                                    "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b",
                                                   "engine_url": "http://127.0.0.1:1"}]})
    assert ValidatorConfig.model_validate({**base, "network": "test"}).network == "test"
    with pytest.raises(SystemExit, match="plain ws:// only on this machine"):
        network_and_netuid(SimpleNamespace(network=plain, netuid=None), {})
    with pytest.raises(SystemExit, match="plain ws:// only on this machine"):
        network_and_netuid(SimpleNamespace(network=None, netuid=None), {"network": plain})
    assert network_and_netuid(SimpleNamespace(network="test", netuid=None), {}) == ("test", 405)
