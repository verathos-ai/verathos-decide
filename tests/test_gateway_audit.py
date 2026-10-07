"""The auditor's side of the gateway: catalog gateways at the miner, sampled gateway answers proving wrong answers
and feeding a drift tally of their own, and the signed routes."""

from __future__ import annotations

import asyncio
import json

import httpx
from bittensor_wallet import Keypair

from tests.mock_engine import MockEngine
from tests.test_miner_app import REQUEST
from tests.test_throughput import KEV, _auditor, block_hash, kp
from verathos import catalog
from verathos.miner.app import CALLER_CONCURRENCY, create_app
from verathos.miner.config import MinerConfig
from verathos.protocol.gateway import GatewaySample
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest
from verathos.protocol.requests import sign_request
from verathos.throughput.auditor import Checks, Result, Slot, Target
from verathos.throughput.feed import GATEWAY_EVIDENCE
from verathos.throughput.policy import AuditPolicy, ScheduledTest
from verathos.validator.throughput import Throughput

PATH = "/e/gpu0/v1/systemone"


def miner(keypair: Keypair, engine: MockEngine, gateway: Keypair):
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5",
        "port": 8091, "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://unused"}]})
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)]), keypair)
    return create_app(config=config, keypair=keypair, manifest=lambda: signed, engines={"gpu0": engine},
                      authorized=lambda hk: hk == gateway.ss58_address, auditors=lambda: set(),
                      gateways=lambda: {gateway.ss58_address})


async def exchange(app, caller: Keypair, receiver: Keypair, request: dict) -> httpx.Response:
    body = json.dumps(request).encode()
    headers = {"Content-Type": "application/json", **sign_request(caller, "POST", PATH, body, receiver.ss58_address)}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
        return await client.post(PATH, content=body, headers=headers)


def request(i: int) -> dict:
    return {**REQUEST, "state": f"{REQUEST['state']} Ticket {i}."}


def test_a_catalog_gateway_is_not_held_to_the_per_caller_limit() -> None:
    miner_kp, gateway = kp(), kp()

    class Slow(MockEngine):
        async def answer(self, request):
            await asyncio.sleep(0.2)
            return await super().answer(request)

    app = miner(miner_kp, Slow("kev-4b"), gateway)

    async def run():
        return await asyncio.gather(*(exchange(app, gateway, miner_kp, request(i))
                                      for i in range(CALLER_CONCURRENCY + 4)))

    assert {r.status_code for r in asyncio.run(run())} == {200}


def test_sampled_gateway_answers_prove_wrong_answers(tmp_path) -> None:
    gateway, honest_kp, wrong_kp = kp(), kp(), kp()
    honest = miner(honest_kp, MockEngine("kev-4b"), gateway)
    wrong = miner(wrong_kp, MockEngine("kev-4b", salt="a cheaper model"), gateway)

    async def collect():
        out = []
        for app, keypair in ((honest, honest_kp), (wrong, wrong_kp)):
            for i in range(3):
                r = await exchange(app, gateway, keypair, request(i))
                assert r.status_code == 200
                out.append(GatewaySample(hotkey=keypair.ss58_address, endpoint_id="gpu0", model_id="kev-4b",
                                         request=request(i), response=r.text))
        return out

    samples = asyncio.run(collect())
    auditor = _auditor(tmp_path, gateway_checks_per_epoch=2)
    auditor.follower.poll()
    # only answers issued to the gateway that submits them count, and at most 2 per endpoint and epoch
    assert auditor.accept_samples(kp().ss58_address, samples) == 0
    assert auditor.accept_samples(gateway.ss58_address, samples) == 4
    assert auditor.accept_samples(gateway.ss58_address, samples) == 0
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))

    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    (evidence,) = auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == wrong_kp.ss58_address and evidence.test_id == GATEWAY_EVIDENCE

    validator = Throughput(netuid=405, mechid=1, policy=AuditPolicy(), auditor_hotkey=auditor.hotkey,
                           chain=auditor.chain)
    with_gateway = catalog.Catalog.parse({**catalog.load().document(), "gateways": [gateway.ss58_address]})
    reference = {"kev-4b": MockEngine("kev-4b")}
    assert asyncio.run(validator.proven_invalid([evidence], reference, with_gateway)) == [evidence]
    # a receipt issued to a caller that is not a gateway of the catalog proves nothing
    assert asyncio.run(validator.proven_invalid([evidence], reference, catalog.load())) == []


def test_drifted_gateway_answers_lose_the_credit_whatever_the_tests_show(tmp_path) -> None:
    """Correct test answers do not dilute drifted answers to users: the two are tallied apart."""
    auditor = _auditor(tmp_path)
    slot, tol = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 20.0), KEV.tolerance
    test = ScheduledTest(3, 560, block_hash(560).hex(), 565)
    auditor.store.add_test(test)
    auditor.store.add_results(test.test_id, [Result(Target(slot, 20.0), "passed", "", 20.0, 20.0, 400.0,
                                                    drift_sum=0.0, drift_count=400)])
    auditor.store.set_state(test.test_id, "confirmed")
    (credit_,) = auditor.store.credits(2, 3, auditor.policy, lambda m: tol)
    assert not credit_.drift
    for i in range(3):
        auditor.store.add_sample(sample_id=f"s{i}", epoch=3, gateway="gw", hotkey="hk", endpoint_id="gpu0",
                                 model_id="kev-4b", request_id=f"n{i}", request={}, response="{}", per_endpoint=4)
        auditor.store.finish_sample(f"s{i}", "checked", drift_sum=40 * 2 * tol.mean_abs, drift_count=40)
    (credit_,) = auditor.store.credits(2, 3, auditor.policy, lambda m: tol)
    assert credit_.drift and credit_.credited_rps == 0.0


def test_routes_list_the_credited_endpoints_a_gateway_may_use(tmp_path) -> None:
    auditor = _auditor(tmp_path)
    slots = {name: Slot(f"hk-{name}", "gpu0", f"http://{name}/e/gpu0", "kev-4b", 30.0)
             for name in ("good", "probation", "wrong", "untested")}
    test = ScheduledTest(3, 560, block_hash(560).hex(), 565)
    auditor.store.add_test(test)
    auditor.store.add_results(test.test_id, [
        Result(Target(slots[n], 30.0), "passed", "", 30.0, 25.0, 350.0) for n in ("good", "probation", "wrong")])
    auditor.store.set_state(test.test_id, "confirmed")
    auditor.store.add_sample(sample_id="s", epoch=3, gateway="gw", hotkey="hk-wrong", endpoint_id="gpu0",
                             model_id="kev-4b", request_id="n", request={}, response="{}", per_endpoint=2)
    from verathos.throughput.feed import Evidence
    auditor.store.finish_sample("s", "checked", "wrong", Evidence(
        hotkey="hk-wrong", endpoint_id="gpu0", model_id="kev-4b", test_id=GATEWAY_EVIDENCE, reason="wrong",
        request_id="n", request={}, response="{}"))
    auditor.update_slots(list(slots.values()))
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()

    signed = auditor.routes(lambda: ["hk-probation/gpu0"])
    assert signed.verify(auditor.hotkey, 405) and not signed.verify(kp().ss58_address, 405)
    (route,) = signed.routes.routes
    assert (route.hotkey, route.url, route.credited_rps, route.latency_ms) == ("hk-good", "http://good/e/gpu0",
                                                                               25.0, 350.0)


def test_the_auditor_service_takes_samples_only_from_catalog_gateways(tmp_path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from verathos.protocol.gateway import ROUTES_PATH, SAMPLES_PATH, GatewaySamples
    from verathos.validator.service import add_gateway_routes

    gateway, miner_kp = kp(), kp()
    honest = miner(miner_kp, MockEngine("kev-4b"), gateway)
    response = asyncio.run(exchange(honest, gateway, miner_kp, request(1)))
    body = GatewaySamples(samples=[GatewaySample(hotkey=miner_kp.ss58_address, endpoint_id="gpu0",
                                                 model_id="kev-4b", request=request(1), response=response.text)])
    raw = body.model_dump_json().encode()
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    app = FastAPI()
    add_gateway_routes(app, auditor, probation=list, gateways=lambda: [gateway.ss58_address])
    client = TestClient(app)

    def post(caller: Keypair, content: bytes, signed: bytes | None = None):
        headers = sign_request(caller, "POST", SAMPLES_PATH, signed or content, auditor.hotkey)
        return client.post(SAMPLES_PATH, content=content, headers=headers)

    assert post(kp(), raw).status_code == 403
    assert post(gateway, raw, signed=b"something else").status_code == 401
    assert post(gateway, b'{"samples": []}').status_code == 422
    assert post(gateway, raw).json() == {"kept": 1}
    assert client.get(ROUTES_PATH).json()["routes"]["routes"] == []  # nothing credited yet
