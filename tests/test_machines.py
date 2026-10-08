"""One hotkey, endpoints on several machines: the primary lists every machine's endpoints in the manifest, and
validators, the auditor and gateways reach each endpoint at its own machine.

Machines are miner services on the test host, at lab addresses or loopback ports (``lab_network``)."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import secrets
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fake_loadgen import make_runner
from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner import __main__ as miner_main
from verathos.miner import chain as miner_chain
from verathos.miner.chain import build_host_endpoints, own_endpoints
from verathos.miner.config import MinerConfig, peer_url
from verathos.miner.peers import Peers, merged
from verathos.protocol.manifest import (
    ENDPOINTS_PATH,
    Endpoint,
    HostEndpoints,
    ManifestRejected,
    accept_host,
    commitment_value,
    endpoint_host,
)
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import sign_request
from verathos.throughput import auditor as auditor_module
from verathos.throughput import measure
from verathos.throughput.auditor import Auditor, Checks
from verathos.throughput.policy import HDR_REQUEST_ID, AuditPolicy, ScheduledTest, Session, session_headers
from verathos.validator.audit import audit_epoch
from verathos.validator.compare import Tolerance
from verathos.validator.discovery import Identities, Neuron, discover
from verathos.validator.throughput import Throughput

pytestmark = pytest.mark.usefixtures("lab_network")
CATALOG = catalog.load()
REQUEST = {"model": "kev-4b", "state": "Hi, we were billed twice for March.",
           "questions": {"churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}}}


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def config(host: str, port: int, endpoints: list[str], tmp_path, **fields) -> MinerConfig:
    return MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": port,
        "listen_host": host, "state_dir": str(tmp_path / f"{host}-{port}"), **fields,
        "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9",
                       "max_concurrency": 4} for e in endpoints]})


def test_peers_are_machines_base_urls_and_a_worker_has_none(tmp_path) -> None:
    assert peer_url("154.54.100.137:20017") == "http://154.54.100.137:20017"
    assert peer_url(" http://[2606:4700:0::1111]:8091/") == "http://[2606:4700::1111]:8091"
    for bad in ("154.54.100.137", "miner.example:80", "1.2.3.4:0", "1.2.3.4:65536", "http://1.2.3.4:80/x", ""):
        with pytest.raises(ValueError):
            peer_url(bad)
    primary = config("203.0.113.5", 8091, ["gpu0"], tmp_path, peers=["198.51.100.7:20017"])
    assert primary.role == "primary" and primary.peers == ["http://198.51.100.7:20017"]
    for fields, why in (({"role": "worker", "peers": ["198.51.100.7:1"]}, "a worker has no peers"),
                        ({"peers": ["203.0.113.5:8091"]}, "not this one"),
                        ({"peers": ["198.51.100.7:1", "http://198.51.100.7:1"]}, "unique")):
        with pytest.raises(ValueError, match=why):
            config("203.0.113.5", 8091, ["gpu0"], tmp_path, **fields)


class FakeSubtensor:
    """The metagraph the miner reads permits and stakes from, and the subnet owner (the auditor)."""

    def __init__(self, permitted: dict[str, float], auditor: str, *args, **kwargs) -> None:
        self.permitted, self.auditor = permitted, auditor

    def metagraph(self, netuid: int, lite: bool = True):
        return SimpleNamespace(hotkeys=[*self.permitted, "unpermitted"], validator_permit=[True] * len(self.permitted)
                               + [False], S=[*self.permitted.values(), 1e6], block=5 * 180 + 7)

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return self.auditor


def test_a_worker_runs_the_full_miner_service_and_reveals_no_more_than_the_manifest(tmp_path, monkeypatch) -> None:
    """Callers are checked on a worker as on any miner: validators with a permit from the metagraph, the catalog's and
    the configured gateways, the auditor's signed test sessions, nobody else; its endpoints document is read-only,
    signed by the hotkey and holds only what the manifest publishes."""
    miner, validator, small, gateway, auditor, stranger = kp(), kp(), kp(), kp(), kp(), kp()
    floor = CATALOG.scoring.validator_min_stake
    chain = FakeSubtensor({validator.ss58_address: floor, small.ss58_address: floor - 1}, auditor.ss58_address)
    monkeypatch.setattr(miner_chain.bt, "Subtensor", lambda *a, **k: chain)
    cfg = config("198.51.100.7", 20017, ["w-gpu0"], tmp_path, role="worker", gateway_hotkeys=[gateway.ss58_address])
    publisher = miner_main.Publisher(cfg, SimpleNamespace(hotkey=miner), CATALOG, publish_on_chain=True,
                                     subtensor=chain)
    assert not publisher.publish_on_chain and publisher.signed is None and publisher.local is None
    authorizer = miner_main.permits(cfg, publisher)
    client = TestClient(miner_main.build_app(cfg, miner, publisher, authorizer, {"w-gpu0": MockEngine("kev-4b")}))
    assert client.get(ENDPOINTS_PATH).status_code == 404  # nothing measured yet
    publisher.local = build_host_endpoints(cfg, own_endpoints(cfg, cfg.endpoints, {"w-gpu0": 18.9}), miner,
                                           CATALOG.digest())

    assert client.get("/v1/manifest").status_code == 404  # the primary serves the hotkey's manifest
    document = client.get(ENDPOINTS_PATH).json()
    host = accept_host(document, hotkey=miner.ss58_address, netuid=405, mechid=1, url="http://198.51.100.7:20017")
    assert set(document["host"]) == set(HostEndpoints.model_fields)
    assert [set(e) for e in document["host"]["endpoints"]] == [set(Endpoint.model_fields)]
    assert host.endpoints[0].url == "http://198.51.100.7:20017/e/w-gpu0" and "127.0.0.1" not in json.dumps(document)
    assert {client.post(ENDPOINTS_PATH).status_code, client.put(ENDPOINTS_PATH).status_code} == {405}

    def call(caller: Keypair, path: str = "/e/w-gpu0/v1/systemone", payload: dict = REQUEST):
        body = json.dumps(payload).encode()
        signed = sign_request(caller, "POST", path, body, miner.ss58_address)
        return client.post(path, content=body, headers={"Content-Type": "application/json", **signed})

    for caller in (validator, gateway):
        r = call(caller)
        assert r.status_code == 200 and SignedReceipt.model_validate(r.json()["receipt"]).signature_valid()
    assert call(validator, "/e/w-gpu0/v1/identity", {"challenge": "ab" * 16}).status_code == 200
    assert call(auditor, "/e/w-gpu0/v1/identity", {"challenge": "ab" * 16}).status_code == 200  # whatever its stake
    for caller in (stranger, small):  # no permit; a permit but less stake than validator_min_stake
        assert call(caller).status_code == 403
        assert call(caller, "/e/w-gpu0/v1/identity", {"challenge": "ab" * 16}).status_code == 403
    assert call(validator, payload={**REQUEST, "state": "x" * 100_000}).status_code == 413  # the per-caller limits
    session = Session(netuid=405, auditor=auditor.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="w-gpu0",
                      test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    r = client.post("/e/w-gpu0/v1/systemone", json=REQUEST,
                    headers={**session_headers(session, auditor), HDR_REQUEST_ID: secrets.token_hex(16)})
    assert r.status_code == 200 and publisher.testing()


def signed_host(keypair: Keypair, url: str, endpoints: dict[str, str], *, signer: Keypair | None = None,
                catalog_digest: str = CATALOG.digest(), at: str | None = None) -> dict:
    """An endpoints document of the machine at ``url``: endpoint id -> model, each listed at ``at`` (default
    ``url``)."""
    from verathos.protocol import signing
    from verathos.protocol.manifest import ENDPOINTS_DOMAIN

    host = HostEndpoints(netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=catalog_digest, url=url,
                         endpoints=[Endpoint(endpoint_id=e, url=f"{at or url}/e/{e}", model_id=m, capacity_rps=10.0)
                                    for e, m in endpoints.items()]).model_dump(mode="json")
    return {"host": host, "signature": signing.sign(signer or keypair, ENDPOINTS_DOMAIN, host)}


def test_the_primary_takes_each_peers_endpoints_only_as_the_hotkey_signed_them_for_that_machine() -> None:
    hotkey, other = kp(), kp()
    urls = {name: f"http://10.0.6.{i}:80" for i, name in enumerate(("good", "foreign", "elsewhere", "posing"), 1)}
    documents = {
        "good": signed_host(hotkey, urls["good"], {"a-gpu0": "kev-4b", "gpu0": "kev-4b", "a-gpu1": "kev-27b"}),
        "foreign": signed_host(hotkey, urls["foreign"], {"b-gpu0": "kev-4b"}, signer=other),
        "elsewhere": signed_host(hotkey, urls["elsewhere"], {"c-gpu0": "kev-4b"}, at="http://10.0.6.9:80"),
        "posing": signed_host(hotkey, urls["good"], {"d-gpu0": "kev-4b"}),
    }
    apps = {}
    for name, url in urls.items():
        app = apps[httpx.URL(url).host] = FastAPI()
        app.get(ENDPOINTS_PATH)(lambda name=name: documents[name])

    class Transport(httpx.AsyncBaseTransport):
        down: set[str] = set()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host in self.down:
                raise httpx.ConnectError("down")
            return await httpx.ASGITransport(app=apps[request.url.host]).handle_async_request(request)

    peers = Peers([urls["good"], urls["foreign"], urls["elsewhere"], urls["posing"]], hotkey=hotkey.ss58_address,
                  netuid=405, mechid=1)

    def poll(hold_s: float = 3600.0, digest: str = CATALOG.digest()) -> None:
        async def run():
            async with httpx.AsyncClient(transport=Transport()) as client:
                await peers.poll(client, digest, hold_s)

        asyncio.run(run())

    poll()
    own = [Endpoint(endpoint_id="gpu0", url="http://10.0.6.10:80/e/gpu0", model_id="kev-4b", capacity_rps=10.0)]
    listed, left_out = merged(own, peers.peers, CATALOG)
    assert [e.endpoint_id for e in listed] == ["gpu0", "a-gpu0"]
    assert {endpoint_host(e.url, e.endpoint_id) for e in listed} == {"http://10.0.6.10:80", urls["good"]}
    assert any("gpu0 of peer http://10.0.6.1:80 left out: this machine lists an endpoint gpu0" in line
               for line in left_out)
    assert any("does not serve kev-27b" in line for line in left_out)
    assert [p.said.split(": ", 1)[1] for p in peers.peers[1:]] == [
        "not signed by this hotkey", "endpoint c-gpu0 is not listed at http://10.0.6.3:80/e/c-gpu0 on a public address",
        "it says it is http://10.0.6.1:80"]

    Transport.down.add("10.0.6.1")  # the peer is out: its endpoints stay listed for an epoch, then are withdrawn
    poll()
    assert [e.endpoint_id for e in peers.peers[0].endpoints] == ["a-gpu0", "gpu0", "a-gpu1"]
    assert "stay listed until it has been out for an epoch" in peers.peers[0].said
    poll(hold_s=0.0)
    assert peers.peers[0].endpoints == [] and "withdrawn" in peers.peers[0].said
    Transport.down.clear()
    poll(digest="1" * 64)  # back, but its endpoints were built for another catalog than the primary's
    assert peers.peers[0].endpoints == [] and "for catalog" in peers.peers[0].said
    with pytest.raises(ManifestRejected, match="not signed"):
        accept_host({"host": documents["good"]["host"], "signature": "00" * 64}, hotkey=hotkey.ss58_address,
                    netuid=405, mechid=1, url=urls["good"])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class CapacityEngine:
    """Answers like the reference, at most ``rps`` requests per second."""

    def __init__(self, rps: float, concurrency: int = 4) -> None:
        self.engine, self.service_s, self.calls = MockEngine("kev-4b"), concurrency / rps, 0
        self._slots = asyncio.Semaphore(concurrency)

    async def answer(self, request):
        self.calls += 1
        async with self._slots:
            await asyncio.sleep(self.service_s)
            return await self.engine.answer(request)


def test_a_validator_gets_its_request_budget_per_endpoint_and_epoch(tmp_path, monkeypatch) -> None:
    """Canaries and identity challenges, counted per validator, endpoint and epoch; beyond the catalog's budget 429
    until the next epoch. Gateways and the auditor's test traffic are not counted."""
    from verathos.miner.admission import RequestBudget

    clock = [(7, 30.4)]
    budget = RequestBudget(lambda: 2, lambda: clock[0])
    assert [budget.take("v", "gpu0") for _ in range(3)] == [None, None, 30.4]
    assert budget.take("v", "gpu1") is None and budget.take("w", "gpu0") is None
    clock[0] = (8, 2160.0)
    assert budget.take("v", "gpu0") is None

    miner, validator, gateway, auditor = kp(), kp(), kp(), kp()
    chain = FakeSubtensor({validator.ss58_address: 1e6}, auditor.ss58_address)
    monkeypatch.setattr(miner_chain.bt, "Subtensor", lambda *a, **k: chain)
    cfg = config("198.51.100.7", 20017, ["gpu0"], tmp_path, gateway_hotkeys=[gateway.ss58_address])
    tight = CATALOG.model_copy(update={"scoring": CATALOG.scoring.model_copy(
        update={"validator_requests_per_epoch": 3})})
    assert CATALOG.scoring.request_budget() == 4 * CATALOG.scoring.canary_pool_size
    assert tight.scoring.request_budget() == 3
    publisher = miner_main.Publisher(cfg, SimpleNamespace(hotkey=miner), tight, publish_on_chain=False,
                                     subtensor=chain)
    authorizer = miner_main.permits(cfg, publisher)
    epoch, left_s = authorizer.epoch(tight.epoch_blocks)  # the chain's head at block 907
    assert epoch == 5 and 0 < left_s <= (6 * 180 - 907) * 12
    client = TestClient(miner_main.build_app(cfg, miner, publisher, authorizer, {"gpu0": MockEngine("kev-4b")}))

    def call(caller: Keypair, path: str = "/e/gpu0/v1/systemone", payload: dict = REQUEST):
        body = json.dumps(payload).encode()
        signed = sign_request(caller, "POST", path, body, miner.ss58_address)
        return client.post(path, content=body, headers={"Content-Type": "application/json", **signed})

    assert call(validator, "/e/gpu0/v1/identity", {"challenge": "ab" * 16}).status_code == 200
    assert [call(validator).status_code for _ in range(3)] == [200, 200, 429]
    refused = call(validator, "/e/gpu0/v1/identity", {"challenge": "ab" * 16})
    assert refused.status_code == 429 and 0 < int(refused.headers["Retry-After"]) <= left_s + 1
    assert all(call(gateway).status_code == 200 for _ in range(5))
    session = Session(netuid=405, auditor=auditor.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                      test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    assert all(client.post("/e/gpu0/v1/systemone", json=REQUEST, headers={
        **session_headers(session, auditor), HDR_REQUEST_ID: secrets.token_hex(16)}).status_code == 200
        for _ in range(5))
    authorizer._head = (6 * 180, time.monotonic())  # the next epoch
    assert call(validator).status_code == 200


def test_two_machines_end_to_end(tmp_path, monkeypatch) -> None:
    """A primary on one port and a worker on another, each its own miner service: the primary lists the worker's
    endpoints, a validator challenges and sends canaries to every endpoint at its own machine, and the auditor tests
    them all there, at once."""
    monkeypatch.setattr(miner_main, "measure_and_store", functools.partial(measure.measure_and_store, run_s=0.2))
    monkeypatch.setattr(miner_main, "wait_ready", functools.partial(measure.wait_ready, timeout_s=2.0))
    monkeypatch.setattr(auditor_module, "SERVED_WAIT_S", 0.0)  # no gateway reports these users' answers
    hotkey, validator, auditor_kp = kp(), kp(), kp()
    chain = FakeSubtensor({validator.ss58_address: 1e6}, auditor_kp.ss58_address)
    monkeypatch.setattr(miner_chain.bt, "Subtensor", lambda *a, **k: chain)
    ports = {"primary": free_port(), "worker": free_port()}
    configs = {"worker": config("127.0.0.1", ports["worker"], ["w-gpu0", "w-gpu1"], tmp_path, role="worker"),
               "primary": config("127.0.0.1", ports["primary"], ["gpu0"], tmp_path,
                                 peers=[f"127.0.0.1:{ports['worker']}"])}
    engines = {"gpu0": CapacityEngine(80), "w-gpu0": CapacityEngine(80), "w-gpu1": CapacityEngine(80)}
    publishers, servers = {}, []
    for role, cfg in configs.items():
        publishers[role] = miner_main.Publisher(cfg, SimpleNamespace(hotkey=hotkey), CATALOG, publish_on_chain=False,
                                                subtensor=chain)
        app = miner_main.build_app(cfg, hotkey, publishers[role], miner_main.permits(cfg, publishers[role]),
                                   {e.endpoint_id: engines[e.endpoint_id] for e in cfg.endpoints})
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=cfg.port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
    while not all(s.started for s in servers):
        time.sleep(0.05)
    try:
        for role in ("worker", "primary"):
            assert publishers[role].adopt(CATALOG)
        primary = publishers["primary"]
        primary.poll_peers()
        manifest = primary.signed.manifest
        assert {e.endpoint_id: endpoint_host(e.url, e.endpoint_id) for e in manifest.endpoints} == {
            "gpu0": f"http://127.0.0.1:{ports['primary']}", "w-gpu0": f"http://127.0.0.1:{ports['worker']}",
            "w-gpu1": f"http://127.0.0.1:{ports['worker']}"}
        assert primary.own_ids() == ["gpu0"] and publishers["worker"].own_ids() == ["w-gpu0", "w-gpu1"]
        assert "the manifest lists 3 endpoints on 2 machines" in miner_main.serving_line(configs["primary"], primary)
        neurons = [Neuron(1, hotkey.ss58_address, "127.0.0.1", ports["primary"],
                          commitment_value(manifest.digest()))]

        async def validate():
            async with httpx.AsyncClient() as client:
                miners = await discover(neurons, 405, 1, client, {CATALOG.digest()})
                return await audit_epoch([e for m in miners for e in m.endpoints()], epoch=5, secret=b"k" * 32,
                                         keypair=validator, references={"kev-4b": MockEngine("kev-4b")},
                                         tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                         netuid=405, mechid=1, pool_size=4, per_endpoint=2,
                                         identities=Identities())

        before = {e: engine.calls for e, engine in engines.items()}
        results = asyncio.run(validate())
        assert {r.endpoint.endpoint.endpoint_id: (r.identity_ok, r.sent, r.answered) for r in results.values()} == {
            e: (True, 2, 2) for e in engines}
        assert all(engines[e].calls >= before[e] + 2 for e in engines)  # each machine answered its own canaries

        policy = dataclasses.replace(AuditPolicy(), max_latency_s=0.5, warmup_s=0.5, test_s=2.0, samples_per_test=4)
        from tests.test_throughput import FakeChain, block_hash

        fake_chain = FakeChain(head=5 * 180 + 40)
        auditor = Auditor(netuid=405, mechid=1, keypair=auditor_kp, chain=fake_chain, policy=policy,
                          catalog=lambda: CATALOG, references={}, db_path=tmp_path / "throughput.sqlite",
                          runner=make_runner(httpx.AsyncHTTPTransport()))
        throughput = Throughput(netuid=405, mechid=1, policy=policy, auditor_hotkey=auditor_kp.ss58_address,
                                chain=fake_chain, auditor=auditor)

        async def refresh():
            async with httpx.AsyncClient() as client:
                return await throughput.refresh(neurons, {CATALOG.digest()}, 5, client)

        throughput.update_slots(asyncio.run(refresh()), neurons, 5)
        assert sorted(t.slot.url for t in auditor.targets(policy)) == sorted(e.url for e in manifest.endpoints)
        selection = fake_chain.head_block - policy.lead_blocks
        auditor.follower.poll()
        asyncio.run(auditor.run_test(ScheduledTest(5, selection, block_hash(selection).hex(), fake_chain.head_block),
                                     Checks({"kev-4b": MockEngine("kev-4b")}, {})))
        outcomes = {o.endpoint_id: o for o in auditor.store.outcomes(hotkey.ss58_address, 0)}
        assert set(outcomes) == set(engines)
        assert all(o.outcome == "passed" and o.credited_rps > 0 for o in outcomes.values()), outcomes
    finally:
        for server in servers:
            server.should_exit = True
