"""The auditor's side of the gateway: catalog gateways at the miner, sampled gateway answers proving wrong answers
and feeding a drift tally of their own, and the signed routes."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
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
    # only answers issued to the gateway that submits them count, for endpoints the auditor knows
    assert auditor.accept_samples(gateway.ss58_address, samples) == 0
    auditor.update_slots([Slot(k.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)
                          for k in (honest_kp, wrong_kp)])
    assert auditor.accept_samples(kp().ss58_address, samples) == 0
    assert auditor.accept_samples(gateway.ss58_address, samples) == 6
    assert auditor.accept_samples(gateway.ss58_address, samples) == 0  # known already
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    states = auditor.store._query("SELECT hotkey, state FROM samples")
    assert sorted(st for hk, st in states if hk == honest_kp.ss58_address) == ["checked", "checked", "skipped"]

    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    published = auditor.verdicts(4)
    (evidence,) = published.verdicts.invalid
    assert evidence.hotkey == wrong_kp.ss58_address and evidence.from_gateway
    # user requests are not published, nor kept once checked: the miner's receipt is all that remains
    assert "billed twice" not in published.model_dump_json() and evidence.request == {}
    assert set(json.loads(evidence.response)) == {"receipt"}
    assert all(r == ("{}", "") for r in auditor.store._query("SELECT request, response FROM samples"))

    validator = Throughput(netuid=405, mechid=1, policy=AuditPolicy(), auditor_hotkey=auditor.hotkey,
                           chain=auditor.chain)
    with_gateway = catalog.Catalog.parse({**catalog.load().document(), "gateways": [gateway.ss58_address]})
    reference = {"kev-4b": MockEngine("kev-4b")}
    assert asyncio.run(validator.proven_invalid([evidence], reference, with_gateway)) == [evidence]
    # a receipt issued to a caller that is not a gateway of the catalog proves nothing
    assert asyncio.run(validator.proven_invalid([evidence], reference, catalog.load())) == []


def test_input_tokens_signed_beyond_the_references_count_are_evidence(tmp_path) -> None:
    """The users' input tokens set the models' emission shares; a miner signing more than the reference counts is
    caught on its sampled answers, as a wrong answer would be."""
    class Inflating(MockEngine):
        async def answer(self, request):
            response = await super().answer(request)
            return {**response, "usage": {"input_tokens": 3 * response["usage"]["input_tokens"] + 10}}

    gateway, honest_kp, greedy_kp = kp(), kp(), kp()
    honest = miner(honest_kp, MockEngine("kev-4b"), gateway)
    greedy = miner(greedy_kp, Inflating("kev-4b"), gateway)

    async def collect():
        out = []
        for app, keypair in ((honest, honest_kp), (greedy, greedy_kp)):
            r = await exchange(app, gateway, keypair, request(0))
            assert r.status_code == 200
            out.append(GatewaySample(hotkey=keypair.ss58_address, endpoint_id="gpu0", model_id="kev-4b",
                                     request=request(0), response=r.text))
        return out

    auditor = _auditor(tmp_path, gateway_checks_per_epoch=2)
    auditor.follower.poll()
    auditor.update_slots([Slot(k.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)
                          for k in (honest_kp, greedy_kp)])
    assert auditor.accept_samples(gateway.ss58_address, asyncio.run(collect())) == 2
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    (evidence,) = auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == greedy_kp.ss58_address and evidence.reason.startswith("input tokens misreported")


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
                                 model_id="kev-4b", request_id=f"n{i}", request={}, response="{}", proof=False)
        auditor.store.finish_sample(f"s{i}", "checked", drift_sum=40 * 2 * tol.mean_abs, drift_count=40,
                                    checked_epoch=3)
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
                             model_id="kev-4b", request_id="n", request={}, response="{}", proof=False)
    from verathos.throughput.feed import Evidence
    auditor.store.finish_sample("s", "checked", "wrong", Evidence(
        hotkey="hk-wrong", endpoint_id="gpu0", model_id="kev-4b", test_id=f"{GATEWAY_EVIDENCE}-n", reason="wrong",
        request_id="n", request={}, response="{}"), checked_epoch=3)
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    assert auditor.verdicts(4) is not None
    assert auditor.routes(lambda: []) is None  # no miners discovered since the start: no table to replace a good one
    auditor.update_slots(list(slots.values()))

    signed = auditor.routes(lambda: ["hk-probation/other-endpoint"])  # probation covers the whole hotkey
    assert signed.verify(auditor.hotkey, 405) and not signed.verify(kp().ss58_address, 405)
    (route,) = signed.routes.routes
    assert (route.hotkey, route.url, route.credited_rps, route.latency_ms) == ("hk-good", "http://good/e/gpu0",
                                                                               25.0, 350.0)


def test_the_auditor_service_takes_samples_only_from_catalog_gateways(tmp_path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from verathos.protocol.gateway import ROUTES_PATH, SAMPLES_PATH
    from verathos.validator.service import add_gateway_routes

    gateway, miner_kp = kp(), kp()
    honest = miner(miner_kp, MockEngine("kev-4b"), gateway)
    response = asyncio.run(exchange(honest, gateway, miner_kp, request(1)))
    sample = GatewaySample(hotkey=miner_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b", request=request(1),
                           response=response.text)
    raw = ('{"samples": [' + sample.model_dump_json() + "]}").encode()
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    auditor.update_slots([Slot(miner_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    app = FastAPI()
    add_gateway_routes(app, auditor, probation=list, gateways=lambda: [gateway.ss58_address])
    client = TestClient(app)

    def post(caller: Keypair, content: bytes, signed: bytes | None = None):
        headers = sign_request(caller, "POST", SAMPLES_PATH, signed or content, auditor.hotkey)
        return client.post(SAMPLES_PATH, content=content, headers=headers)

    assert post(kp(), raw).status_code == 403
    assert post(gateway, raw, signed=b"something else").status_code == 401
    assert post(gateway, b'{"samples": {}}').status_code == 422
    assert post(gateway, b'{"samples": [{"not": "a sample"}]}').json() == {"kept": 0}  # judged one by one
    assert post(gateway, raw).json() == {"kept": 1}
    unsized = client.post(SAMPLES_PATH, content=iter([raw]), headers=sign_request(gateway, "POST", SAMPLES_PATH, raw,
                                                                                  auditor.hotkey))
    assert unsized.status_code == 413  # a declared length is required
    assert client.get(ROUTES_PATH).json()["routes"]["routes"] == []  # nothing credited yet


def test_what_an_endpoint_answers_the_gateway_during_a_test_counts_as_delivered(tmp_path) -> None:
    """Miners serve users before test traffic: a busy endpoint loses nothing, refusing users gains nothing."""
    import time as time_module

    from tests.test_throughput import _result
    from verathos.protocol.gateway import ServedCount

    auditor = _auditor(tmp_path)
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 20.0)
    auditor.update_slots([slot])
    start = int(time_module.time()) - 30
    window = (start + 0.5, start + 16.5)  # test_s 16
    counts = [ServedCount(hotkey="hk", endpoint_id="gpu0", second=start + i, answered=10) for i in range(17)]
    assert auditor.accept_served(counts) == 17
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)

    async def evaluate(completed: int):
        return await auditor.evaluate(test, Target(slot, 20.0), _result("hk/gpu0", completed=completed),
                                      Checks({}, {}).for_test(), auditor.policy, window)

    result = asyncio.run(evaluate(160))  # 10/s of the test's own and 10/s to the gateway
    assert result.outcome == "passed" and result.delivered_rps == pytest.approx(20.0)
    assert json.loads(result.detail)["gateway_answered"] == pytest.approx(160.0)
    assert auditor.accept_served([ServedCount(hotkey="unknown", endpoint_id="gpu0", second=start, answered=5)]) == 0


def test_a_test_leaves_room_for_the_users_an_endpoint_serves(tmp_path) -> None:
    """Users go first: a test sends only what the endpoint's recent user traffic leaves of the tested rate, so the
    endpoint is never asked for more than its tested rate."""
    import time as time_module

    from verathos.protocol.gateway import ServedCount
    from verathos.throughput.auditor import USER_RATE_LAG_S, USER_RATE_S

    auditor = _auditor(tmp_path)
    busy, quiet = (Slot("hk", e, f"http://x/e/{e}", "kev-4b", 20.0) for e in ("gpu0", "gpu1"))
    auditor.update_slots([busy, quiet])
    now = time_module.time()
    first = int(now - USER_RATE_LAG_S - USER_RATE_S)
    assert auditor.accept_served([ServedCount(hotkey="hk", endpoint_id="gpu0", second=first + i, answered=12)
                                  for i in range(int(USER_RATE_S) + 1)]) == int(USER_RATE_S) + 1
    assert auditor.test_rate(Target(busy, 20.0), now, auditor.policy) == pytest.approx(8.0, abs=0.3)
    assert auditor.test_rate(Target(busy, 10.0), now, auditor.policy) == pytest.approx(1.0)  # at least 10%
    assert auditor.test_rate(Target(quiet, 20.0), now, auditor.policy) == 20.0
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    plan = auditor.plan(test, [Target(busy, 20.0), Target(quiet, 20.0)], int(now * 1000), auditor.policy)
    assert [t["rate_rps"] for t in plan["targets"]] == [pytest.approx(8.0, abs=0.3), 20.0]


def test_what_users_got_from_an_endpoint_is_published_with_its_credit(tmp_path) -> None:
    import time as time_module

    from pydantic import ValidationError

    from verathos.protocol.gateway import ServedCount, latency_bucket

    auditor = _auditor(tmp_path)
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 30.0)
    test = ScheduledTest(3, 560, block_hash(560).hex(), 565)
    auditor.store.add_test(test)
    auditor.store.add_results(test.test_id, [Result(Target(slot, 30.0), "passed", "", 30.0, 25.0, 350.0)])
    auditor.store.set_state(test.test_id, "confirmed")
    auditor.update_slots([slot])
    auditor.chain.head_block = 3 * 180 + 1
    auditor.follower.poll()
    now = int(time_module.time())
    counts = [ServedCount(hotkey="hk", endpoint_id="gpu0", second=now - 1, answered=3, offered=8, refused=2,
                          latency={latency_bucket(0.2): 2, latency_bucket(0.9): 1}),
              ServedCount(hotkey="hk", endpoint_id="gpu0", second=now, offered=2, refused=1)]  # refused, no answer
    assert auditor.accept_served(counts) == 2
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    (credit,) = auditor.verdicts(4).verdicts.credits
    assert (credit.user_offers, credit.user_refusals, credit.user_answers) == (10, 3, 3)
    assert credit.user_latency_ms == pytest.approx(200, rel=0.06)
    assert auditor.store.users(3)[("hk", "gpu0", "kev-4b")].offered == 10  # kept across a restart
    for bad in ({}, {"answered": 1, "latency": {latency_bucket(0.2): 2}}, {"answered": 1, "latency": {500: 1}}):
        with pytest.raises(ValidationError):
            ServedCount(hotkey="hk", endpoint_id="gpu0", second=now, **bad)


def test_a_test_is_judged_against_what_the_endpoint_was_asked(tmp_path) -> None:
    """Users sending less than the test left room for cost the endpoint nothing; users it refused count as asked. A
    report posted again is taken once."""
    import time as time_module

    from tests.test_throughput import _result
    from verathos.protocol.gateway import ServedCount

    auditor = _auditor(tmp_path)
    dip, refuser = (Slot("hk", e, f"http://x/e/{e}", "kev-4b", 20.0) for e in ("dip", "refuser"))
    auditor.update_slots([dip, refuser])
    start = int(time_module.time()) - 30
    window = (start + 0.5, start + 16.5)  # test_s 16; the test left 10/s for users and sent 10/s itself
    assert auditor.accept_served([ServedCount(hotkey="hk", endpoint_id="dip", second=start + i, attempts=6,
                                              answered=6) for i in range(17)], report_id="r1") == 17
    assert auditor.accept_served([ServedCount(hotkey="hk", endpoint_id="dip", second=start, attempts=6,
                                              answered=6)], report_id="r1") == 0  # the same report again
    auditor.accept_served([ServedCount(hotkey="hk", endpoint_id="refuser", second=start + i, attempts=10,
                                       answered=5) for i in range(17)], report_id="r2")
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)

    def evaluate(slot: Slot):
        return asyncio.run(auditor.evaluate(test, Target(slot, 20.0), _result(slot.key, completed=160),
                                            Checks({}, {}).for_test(), auditor.policy, window))

    users_dipped = evaluate(dip)
    assert users_dipped.outcome == "passed" and users_dipped.credited_rps == pytest.approx(20.0)
    refused = evaluate(refuser)  # delivered 15 of the 20 requests/s it was asked
    assert refused.outcome == "partial" and refused.credited_rps == pytest.approx(20 * 0.75 * 0.75)


def test_a_discovery_that_finds_no_miner_keeps_the_endpoints_known(tmp_path) -> None:
    from verathos.validator.discovery import Neuron
    from verathos.validator.throughput import Throughput

    auditor = _auditor(tmp_path)
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 20.0)
    auditor.update_slots([slot])
    throughput = Throughput(netuid=405, mechid=1, policy=auditor.policy, auditor_hotkey=auditor.hotkey,
                            chain=auditor.chain, auditor=auditor)
    throughput.update_slots([], [Neuron(1, "hk", "1.2.3.4", 8091, "commitment")])  # every fetch failed
    assert auditor._slots == [slot]


def test_signed_documents_of_a_newer_auditor_still_verify() -> None:
    from verathos.protocol import signing
    from verathos.protocol.gateway import ROUTES_DOMAIN, Routes, SignedRoutes
    from verathos.throughput.feed import VERDICTS_DOMAIN, SignedVerdicts, Verdicts

    keypair = kp()
    verdicts = Verdicts.model_validate({"netuid": 405, "auditor": keypair.ss58_address, "epoch": 4, "credits": [
        {"hotkey": "hk", "endpoint_id": "gpu0", "model_id": "kev-4b", "credited_rps": 1.0, "latency_ms": 1.0,
         "test_epoch": 3, "qualified": True, "a_later_field": 7}], "invalid": [], "another": {"x": 1}})
    received = SignedVerdicts.model_validate_json(SignedVerdicts.create(verdicts, keypair).model_dump_json())
    assert signing.verify(keypair.ss58_address, VERDICTS_DOMAIN, received.verdicts.model_dump(mode="json"),
                          received.signature)
    routes = Routes.model_validate({"netuid": 405, "auditor": keypair.ss58_address, "epoch": 4, "issued_ms": 1,
                                    "catalog_digest": "0" * 64, "routes": [], "later": True})
    received_routes = SignedRoutes.model_validate_json(SignedRoutes.create(routes, keypair).model_dump_json())
    assert received_routes.verify(keypair.ss58_address, 405)
    assert signing.verify(keypair.ss58_address, ROUTES_DOMAIN, received_routes.routes.model_dump(mode="json"),
                          received_routes.signature)


def test_served_verdicts_are_the_same_after_a_restart(tmp_path) -> None:
    from verathos.throughput.auditor import Auditor

    auditor = _auditor(tmp_path)
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    first = auditor.verdicts(4)
    auditor.store.add_sample(sample_id="late", epoch=3, gateway="gw", hotkey="hk", endpoint_id="gpu0",
                             model_id="kev-4b", request_id="n", request={}, response="{}", proof=False)
    restarted = Auditor(netuid=405, mechid=1, keypair=auditor.keypair, chain=auditor.chain, policy=auditor.policy,
                        catalog=catalog.load, references={}, db_path=tmp_path / "t.sqlite")
    restarted.follower.poll()
    assert restarted.verdicts(4).model_dump_json() == first.model_dump_json()


def test_samples_to_check_are_drawn_per_endpoint_and_proofs_always_go(tmp_path) -> None:
    import random

    auditor = _auditor(tmp_path)
    store = auditor.store
    for i in range(8):
        store.add_sample(sample_id=f"a{i}", epoch=3, gateway="gw", hotkey="hk", endpoint_id="gpu0",
                         model_id="kev-4b", request_id=f"n{i}", request={}, response="{}", proof=False)
    assert not store.add_sample(sample_id="a9", epoch=3, gateway="gw", hotkey="hk", endpoint_id="gpu0",
                                model_id="kev-4b", request_id="n9", request={}, response="{}", proof=False)
    assert store.add_sample(sample_id="p", epoch=3, gateway="gw", hotkey="hk", endpoint_id="gpu0",
                            model_id="kev-4b", request_id="np", request={}, response="{}", proof=True)
    chosen = {row[0] for row in store.samples_to_check(3, 2, random.Random(1))}
    assert "p" in chosen and len(chosen) == 3
    skipped = store._query("SELECT COUNT(*) FROM samples WHERE state = 'skipped'")[0][0]
    assert skipped == 6
    picked = next(s for s in chosen if s != "p")
    for _ in range(3):
        store.failed_check(picked)
    assert store._query("SELECT state FROM samples WHERE sample_id = ?", (picked,))[0][0] == "unchecked"
    store.add_sample(sample_id="old", epoch=1, gateway="gw", hotkey="hk", endpoint_id="gpu1", model_id="kev-4b",
                     request_id="no", request={}, response="{}", proof=False)
    store.samples_to_check(3, 2, random.Random(1))
    assert store._query("SELECT state FROM samples WHERE sample_id = 'old'")[0][0] == "unchecked"


def test_sample_checks_wait_while_a_test_runs(tmp_path) -> None:
    gateway, miner_kp = kp(), kp()
    honest = miner(miner_kp, MockEngine("kev-4b"), gateway)
    response = asyncio.run(exchange(honest, gateway, miner_kp, request(1)))
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    auditor.update_slots([Slot(miner_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    sample = GatewaySample(hotkey=miner_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b", request=request(1),
                           response=response.text)
    assert auditor.accept_samples(gateway.ss58_address, [sample]) == 1
    auditor._tests_running = 1
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    assert auditor.store._query("SELECT state FROM samples")[0][0] == "pending"
    auditor._tests_running = 0
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    assert auditor.store._query("SELECT state FROM samples")[0][0] == "checked"


def test_nothing_is_served_before_the_chain_head_is_known_and_miners_are_discovered(tmp_path) -> None:
    auditor = _auditor(tmp_path)  # its follower has not polled yet
    assert auditor.routes(list) is None and auditor.verdicts(0) is None
    auditor.follower.poll()
    assert auditor.routes(list) is None
    auditor.update_slots([])
    assert auditor.routes(list) is not None


def test_user_demand_per_model_is_published_with_the_verdicts(tmp_path) -> None:
    import time as time_module

    from verathos.protocol.gateway import ServedCount

    auditor = _auditor(tmp_path)
    auditor.follower.poll()  # epoch 3
    auditor.update_slots([Slot("a", "gpu0", "http://a/e/gpu0", "kev-4b", 50.0),
                          Slot("b", "gpu0", "http://b/e/gpu0", "kev-9b", 50.0)])
    now = int(time_module.time())
    epoch_s = 180 * 12.0
    auditor.accept_served([ServedCount(hotkey="a", endpoint_id="gpu0", second=now, answered=1,
                                       input_tokens=int(epoch_s * 20)),
                           ServedCount(hotkey="b", endpoint_id="gpu0", second=now, answered=1,
                                       input_tokens=int(epoch_s * 5))])
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    auditor.close_demand(4)
    verdicts = auditor.verdicts(4).verdicts
    assert verdicts.demand == {"kev-4b": pytest.approx(0.1 * 20), "kev-9b": pytest.approx(0.1 * 5)}  # alpha 0.1
    auditor.chain.head_block = 5 * 180 + 1  # an epoch without traffic: demand decays
    auditor.follower.poll()
    auditor.close_demand(5)
    assert auditor.store.demand_ema(4) == {"kev-4b": pytest.approx(0.9 * 2.0), "kev-9b": pytest.approx(0.9 * 0.5)}


def test_an_auditor_database_one_version_old_is_migrated_not_dropped(tmp_path) -> None:
    import sqlite3

    from verathos.throughput.auditor import SCHEMA_VERSION, Store

    Store(tmp_path / "old.sqlite")  # a version 4 database: today's without the demand and users tables
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.executescript("DROP TABLE demand; DROP TABLE users; "
                     "INSERT INTO tests VALUES ('kept', 3, 560, 'h', 565, 'confirmed', ''); "
                     "PRAGMA user_version = 4;")
    db.close()
    store = Store(tmp_path / "old.sqlite")
    assert store._query("SELECT test_id FROM tests") == [("kept",)]
    assert store._query("PRAGMA user_version")[0][0] == SCHEMA_VERSION
    assert store.demand_ema(10) == {}
