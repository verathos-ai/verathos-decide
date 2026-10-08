"""The auditor's side of the gateway: catalog gateways at the miner, sampled gateway answers proving wrong answers
and feeding a drift tally of their own, and the signed routes."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from bittensor_wallet import Keypair

from tests.lineups import two_active_and_a_candidate
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

pytestmark = pytest.mark.usefixtures("lab_network")

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


def add_test_samples(auditor, test: ScheduledTest, slots) -> None:
    """The gateways' sample of each endpoint's answers to users in the test: well formed, and no reference to
    recompute them."""
    answers = asyncio.run(MockEngine("kev-4b").answer(REQUEST))["answers"]
    for s in slots:
        assert auditor.store.add_sample(sample_id=f"t-{s.key}", epoch=test.epoch, gateway="gw", hotkey=s.hotkey,
                                        endpoint_id=s.endpoint_id, model_id=s.model_id, request_id="n",
                                        request=REQUEST, response=json.dumps({"answers": answers}),
                                        proof=False, test_id=test.test_id)


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

    validator = Throughput(netuid=405, policy=AuditPolicy(), auditor_hotkey=auditor.hotkey,
                           chain=auditor.chain)
    with_gateway = catalog.Catalog.parse({**catalog.load().document(), "gateways": [gateway.ss58_address]})
    reference = {"kev-4b": MockEngine("kev-4b")}
    assert asyncio.run(validator.proven_invalid([evidence], reference, with_gateway)) == [evidence]
    # a receipt issued to a caller that is not a gateway of the catalog proves nothing
    assert asyncio.run(validator.proven_invalid([evidence], reference, catalog.load())) == []


def test_a_reference_on_other_weights_than_the_catalog_pins_checks_nothing(tmp_path) -> None:
    """A lagging or misconfigured reference cannot prove honest miners wrong: an engine naming other weights than the
    catalog pins is not used, and after the catalog pins other weights in place, the model's answers go unchecked for
    ``revision_transition_epochs`` while miners and references update."""
    gateway, wrong_kp = kp(), kp()
    wrong = miner(wrong_kp, MockEngine("kev-4b", salt="a cheaper model"), gateway)

    def samples(i: int) -> list[GatewaySample]:
        r = asyncio.run(exchange(wrong, gateway, wrong_kp, request(i)))
        return [GatewaySample(hotkey=wrong_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b",
                              request=request(i), response=r.text)]

    def reference(revision: str) -> Checks:
        return Checks({"kev-4b": MockEngine("kev-4b", served={"data": [{"id": "kev-4b", "revision": revision}]})}, {})

    auditor = _auditor(tmp_path, gateway_checks_per_epoch=1)
    auditor.follower.poll()
    auditor.update_slots([Slot(wrong_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    assert auditor.accept_samples(gateway.ss58_address, samples(1)) == 1
    asyncio.run(auditor.check_samples(reference("0" * 40)))
    assert auditor.store._query("SELECT state, reason FROM samples") == [("unchecked", "no reference for the model")]

    document = catalog.load().document()
    bumped = catalog.Catalog.parse({**document, "models": [
        {**m, "hf_revision": "1" * 40} if m["model_id"] == "kev-4b" else m for m in document["models"]]})
    auditor.catalog = lambda: bumped
    assert auditor.accept_samples(gateway.ss58_address, samples(2)) == 1
    asyncio.run(auditor.check_samples(reference("1" * 40)))  # the reference updated at once, the miner not yet
    assert auditor.store._query("SELECT COUNT(*) FROM samples WHERE evidence IS NOT NULL") == [(0,)]
    auditor.chain.head_block += 3 * 180  # the transition is over
    auditor.follower.poll()
    assert auditor.accept_samples(gateway.ss58_address, samples(3)) == 1
    asyncio.run(auditor.check_samples(reference("1" * 40)))
    assert auditor.store._query("SELECT COUNT(*) FROM samples WHERE evidence IS NOT NULL") == [(1,)]


def test_what_a_reference_serves_and_when_a_model_is_in_transition() -> None:
    from verathos.validator.references import served_revisions, transitioning, wrong_revision

    kev = catalog.load().get("kev-4b")
    clef_doc = {"data": [{"id": "Cloudflare/clef-flash", "revision": "a" * 40, "dtype": "float32"}]}
    kev_doc = {"models": [{"name": "kev-latest", "run": "models/kev-4b", "dtype": "float32"}]}
    assert served_revisions(clef_doc) == {"a" * 40} and served_revisions(kev_doc) == set()
    pinned_kev = {"models": [{"name": "kev-latest", "run": f"jaredpalmer/kev-4b@{'b' * 40}"}]}  # as the setup runs it
    assert served_revisions(pinned_kev) == {"b" * 40}
    assert served_revisions(None) == served_revisions({"data": "x"}) == set()
    assert "the catalog pins" in asyncio.run(wrong_revision(MockEngine("kev-4b", served=clef_doc), kev))
    assert asyncio.run(wrong_revision(MockEngine("kev-4b", served=kev_doc), kev)) is None  # cannot tell
    on_pin = {"data": [{"revision": kev.hf_revision}]}
    assert asyncio.run(wrong_revision(MockEngine("kev-4b", served=on_pin), kev)) is None

    cat, pins = catalog.load(), {}
    assert transitioning(pins, cat, 10) == set()  # first seen: not in transition
    bumped = catalog.Catalog.parse({**cat.document(), "models": [
        {**m, "engine": {**m["engine"], "revision": "2" * 40}} if m["model_id"] == "kev-9b" else m
        for m in cat.document()["models"]]})
    assert [transitioning(pins, bumped, e) for e in (11, 12, 13, 14)] == [{"kev-9b"}] * 3 + [set()]
    still = bumped.model_copy(update={"scoring": bumped.scoring.model_copy(update={"revision_transition_epochs": 0})})
    assert transitioning(dict(pins), still, 12) == set()


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


def test_input_tokens_signed_below_the_references_count_are_evidence(tmp_path) -> None:
    """Users are billed the signed input tokens: a miner signing fewer than the reference counts (serving its users
    for free) is caught as one signing more would be."""
    class Deflating(MockEngine):
        async def answer(self, request):
            response = await super().answer(request)
            return {**response, "usage": {"input_tokens": response["usage"]["input_tokens"] // 3}}

    gateway, cheap_kp = kp(), kp()
    cheap = miner(cheap_kp, Deflating("kev-4b"), gateway)
    r = asyncio.run(exchange(cheap, gateway, cheap_kp, request(0)))
    assert r.status_code == 200
    auditor = _auditor(tmp_path, gateway_checks_per_epoch=2)
    auditor.follower.poll()
    auditor.update_slots([Slot(cheap_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    sample = GatewaySample(hotkey=cheap_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b", request=request(0),
                           response=r.text)
    assert auditor.accept_samples(gateway.ss58_address, [sample]) == 1
    asyncio.run(auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    (evidence,) = auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == cheap_kp.ss58_address and evidence.reason.startswith("input tokens misreported")


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


def test_routes_outlive_epochs_without_a_confirmed_test(tmp_path) -> None:
    """An auditor that missed two tests (it was down, or they were void) scores no endpoint for a while, but keeps
    routing users to the endpoints of its latest confirmed test, without hotkeys proven wrong since."""
    from verathos.throughput.feed import Evidence

    auditor = _auditor(tmp_path)
    slots = [Slot(f"hk-{name}", "gpu0", f"http://{name}/e/gpu0", "kev-4b", 30.0) for name in ("good", "wrong")]
    test = ScheduledTest(3, 560, block_hash(560).hex(), 565)
    auditor.store.add_test(test)
    auditor.store.add_results(test.test_id, [Result(Target(s, 30.0), "passed", "", 30.0, 25.0, 350.0) for s in slots])
    auditor.store.set_state(test.test_id, "confirmed")
    auditor.store.add_sample(sample_id="s", epoch=4, gateway="gw", hotkey="hk-wrong", endpoint_id="gpu0",
                             model_id="kev-4b", request_id="n", request={}, response="{}", proof=False)
    auditor.store.finish_sample("s", "checked", "wrong", Evidence(
        hotkey="hk-wrong", endpoint_id="gpu0", model_id="kev-4b", test_id=f"{GATEWAY_EVIDENCE}-n", reason="wrong",
        request_id="n", request={}, response="{}"), checked_epoch=4)
    auditor.update_slots(slots)
    auditor.chain.head_block = 7 * 180 + 1  # epochs 5 and 6 had no confirmed test
    auditor.follower.poll()
    assert auditor.verdicts(7).verdicts.credits == []  # nothing is scored on a test that old
    (route,) = auditor.routes(lambda: []).routes.routes
    assert (route.hotkey, route.credited_rps) == ("hk-good", 25.0)


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
    add_test_samples(auditor, test, [slot])

    async def evaluate(completed: int):
        return await auditor.evaluate(test, Target(slot, 20.0), _result("hk/gpu0", completed=completed),
                                      Checks({}, {}).for_test(), auditor.policy, window)

    result = asyncio.run(evaluate(160))  # 10/s of the test's own and 10/s to the gateway
    assert result.outcome == "passed" and result.delivered_rps == pytest.approx(20.0) == result.credited_rps
    assert json.loads(result.detail)["gateway_answered"] == pytest.approx(160.0)
    assert auditor.accept_served([ServedCount(hotkey="unknown", endpoint_id="gpu0", second=start, answered=5)]) == 0


def test_answers_to_users_in_a_test_count_once_the_gateways_sample_of_them_is_checked(tmp_path) -> None:
    """An endpoint cannot answer users with junk while a test runs and give the test its GPU: its answers to users in
    the test count as delivered only once the gateways' sample of them passes the check; a wrong one is evidence, and
    without a sample only its replies to the test are credited."""
    import time as time_module

    from tests.test_throughput import _result
    from verathos.protocol.gateway import ServedCount

    gateway = kp()
    keys = {name: kp() for name in ("honest", "junk", "unsampled")}
    apps = {"honest": miner(keys["honest"], MockEngine("kev-4b"), gateway),
            "junk": miner(keys["junk"], MockEngine("kev-4b", salt="uniform junk"), gateway)}
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    slots = {name: Slot(k.ss58_address, "gpu0", f"http://{name}/e/gpu0", "kev-4b", 20.0) for name, k in keys.items()}
    auditor.update_slots(list(slots.values()))
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    auditor.store.add_test(test)

    async def sampled(name: str) -> GatewaySample:
        r = await exchange(apps[name], gateway, keys[name], request(1))
        return GatewaySample(hotkey=keys[name].ss58_address, endpoint_id="gpu0", model_id="kev-4b",
                             request=request(1), response=r.text, test_id=test.test_id)

    samples = [asyncio.run(sampled(name)) for name in apps]
    assert auditor.accept_samples(gateway.ss58_address, samples) == 2
    assert auditor.accept_samples(gateway.ss58_address, samples) == 0  # once per endpoint and test
    start = int(time_module.time()) - 30
    window = (start + 0.5, start + 16.5)
    assert auditor.accept_served([ServedCount(hotkey=k.ss58_address, endpoint_id="gpu0", second=start + i,
                                              attempts=10, answered=10) for k in keys.values() for i in range(17)])

    def evaluate(name: str):
        r = _result(slots[name].key, sent=160, completed=160)  # 10/s of the test's own, 10/s to users
        return asyncio.run(auditor.evaluate(test, Target(slots[name], 20.0), r,
                                            Checks({"kev-4b": MockEngine("kev-4b")}, {}).for_test(), auditor.policy,
                                            window))

    honest, junk, unsampled = evaluate("honest"), evaluate("junk"), evaluate("unsampled")
    assert honest.outcome == "passed" and honest.credited_rps == pytest.approx(20.0)
    assert junk.outcome == "invalid" and junk.credited_rps == 0.0 and "answer to a user" in junk.reason
    assert unsampled.outcome == "passed" and unsampled.credited_rps == pytest.approx(10.0)
    assert "unchecked" in unsampled.reason
    auditor.store.set_state(test.test_id, "confirmed")
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    (evidence,) = auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == keys["junk"].ss58_address and evidence.from_gateway


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
    """Users sending less than usual cost an endpoint nothing when all the model's users dipped; an endpoint whose
    users went elsewhere (it was slow) and one refusing users are judged against what users usually send it. A report
    posted again is taken once."""
    import time as time_module

    from tests.test_throughput import _result
    from verathos.protocol.gateway import ServedCount

    auditor = _auditor(tmp_path)
    names = ("dip", "other", "slow", "refuser")
    slots = {n: Slot("hk", n, f"http://x/e/{n}", "kev-4b", 20.0) for n in names}
    auditor.update_slots(list(slots.values()))
    start = int(time_module.time()) - 30
    window = (start + 0.5, start + 16.5)  # test_s 16; the test left 10/s for users and sent 10/s itself
    sent_answered = {"dip": (6, 6), "other": (6, 6), "slow": (2, 2), "refuser": (10, 5)}  # per second in the window
    counts = [ServedCount(hotkey="hk", endpoint_id=n, second=start - 80 + i, attempts=10, answered=10)
              for n in names for i in range(60)]  # before the test: 10/s each
    counts += [ServedCount(hotkey="hk", endpoint_id=n, second=start + i, attempts=a, answered=b)
               for n, (a, b) in sent_answered.items() for i in range(17)]
    assert auditor.accept_served(counts, report_id="r1") == len(counts)
    assert auditor.accept_served(counts[:1], report_id="r1") == 0  # the same report again
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    add_test_samples(auditor, test, slots.values())
    scale = auditor.user_scales(window)["kev-4b"]

    def evaluate(name: str):
        return asyncio.run(auditor.evaluate(test, Target(slots[name], 20.0), _result(slots[name].key, completed=160),
                                            Checks({}, {}).for_test(), auditor.policy, window, scale))

    users_dipped = evaluate("dip")
    assert users_dipped.outcome == "passed" and users_dipped.credited_rps == pytest.approx(20.0)
    assert evaluate("slow").outcome == "partial"  # 2/s while users of the model sent each endpoint about 6/s
    refused = evaluate("refuser")
    assert refused.outcome == "partial" and refused.credited_rps < 15


def test_users_above_their_usual_rate_displace_test_requests_at_no_cost(tmp_path) -> None:
    """A burst of users during a test (a customer, or test traffic relayed through the gateway) takes the place of
    test requests one for one, as miners serve users first: an endpoint that answers them all and the test's rest up
    to the tested rate passes. Refusing users still counts against it."""
    import time as time_module

    from tests.test_throughput import _result
    from verathos.protocol.gateway import ServedCount

    auditor = _auditor(tmp_path)
    slots = {n: Slot("hk", n, f"http://x/e/{n}", "kev-4b", 20.0) for n in ("busy", "refuser")}
    auditor.update_slots(list(slots.values()))
    start = int(time_module.time()) - 30
    window = (start + 0.5, start + 16.5)  # test_s 16; users usually send 6/s, so the test sent 14/s
    counts = [ServedCount(hotkey="hk", endpoint_id=n, second=start - 80 + i, attempts=6, answered=6)
              for n in slots for i in range(60)]
    counts += [ServedCount(hotkey="hk", endpoint_id=n, second=start + i, attempts=16, answered=16 if n == "busy" else 8)
               for n in slots for i in range(17)]
    assert auditor.accept_served(counts) == len(counts)
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    add_test_samples(auditor, test, slots.values())
    scale = auditor.user_scales(window)["kev-4b"]

    def evaluate(name: str, completed_per_s: int):
        r = _result(slots[name].key, sent=14 * 16, completed=completed_per_s * 16)
        return asyncio.run(auditor.evaluate(test, Target(slots[name], 20.0), r, Checks({}, {}).for_test(),
                                            auditor.policy, window, scale))

    busy = evaluate("busy", 5)  # 21/s in all: 16 to users, 5 of the test's
    assert busy.outcome == "passed" and busy.credited_rps == pytest.approx(20.0)
    refuser = evaluate("refuser", 12)  # 20/s in all, but 8 users/s refused while it served the test
    assert refuser.outcome == "partial" and refuser.credited_rps < 12


def test_test_traffic_and_canaries_relayed_through_the_gateway_are_recognised(tmp_path, caplog) -> None:
    import random

    from verathos.throughput.workload import in_mix_form, make_request
    from verathos.validator.canaries import canary_like, canary_pool

    mix = [make_request("kev-4b", KEV.workload, random.Random(i))["state"] for i in range(20)]
    canaries = [c["state"] for e in range(20) for c in canary_pool(b"k" * 32, e, "kev-4b", 16)]
    users = [REQUEST["state"], "Ticket 0123456789abcdef. please refund my order.", "Hi team, a question.", ""]
    assert all(map(in_mix_form, mix)) and not any(map(in_mix_form, canaries + users))
    assert all(map(canary_like, canaries)) and not any(map(canary_like, mix + users))

    gateway, miner_kp = kp(), kp()
    honest = miner(miner_kp, MockEngine("kev-4b"), gateway)
    relayed = make_request("kev-4b", KEV.workload, random.Random(1))
    response = asyncio.run(exchange(honest, gateway, miner_kp, relayed))
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    auditor.update_slots([Slot(miner_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    sample = GatewaySample(hotkey=miner_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b", request=relayed,
                           response=response.text)
    with caplog.at_level("WARNING", logger="verathos.throughput.auditor"):
        assert auditor.accept_samples(gateway.ss58_address, [sample]) == 1
    assert "relays what it is sent through the gateway" in caplog.text


def test_a_discovery_that_finds_no_miner_keeps_the_endpoints_known(tmp_path) -> None:
    from verathos.validator.discovery import Neuron
    from verathos.validator.throughput import Throughput

    auditor = _auditor(tmp_path)
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 20.0)
    auditor.update_slots([slot])
    throughput = Throughput(netuid=405, policy=auditor.policy, auditor_hotkey=auditor.hotkey,
                            chain=auditor.chain, auditor=auditor)
    throughput.update_slots([], [Neuron(1, "hk", "1.2.3.4", 8091, "commitment")], 3)  # every fetch failed
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
    restarted = Auditor(netuid=405, keypair=auditor.keypair, chain=auditor.chain, policy=auditor.policy,
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


def test_the_paid_input_tokens_each_endpoint_answered_are_summed_over_the_gateways_and_published(tmp_path) -> None:
    """Every catalog gateway reports per endpoint and second; the auditor sums the paid input tokens per endpoint and
    epoch and states the epoch before's with the verdicts, signed: what the usage pool splits by."""
    import time as time_module

    from verathos.protocol.gateway import ServedCount
    from verathos.throughput.feed import PaidServed, SignedVerdicts, paid_served

    auditor = _auditor(tmp_path)
    auditor.follower.poll()  # epoch 3
    auditor.update_slots([Slot("a", "gpu0", "http://a/e/gpu0", "kev-4b", 50.0),
                          Slot("a", "gpu1", "http://a/e/gpu1", "kev-4b", 50.0),
                          Slot("b", "gpu0", "http://b/e/gpu0", "kev-9b", 50.0)])
    now = int(time_module.time())
    first_gateway = [ServedCount(hotkey="a", endpoint_id="gpu0", second=now, answered=3, input_tokens=900,
                                 paid_input_tokens=600),
                     ServedCount(hotkey="b", endpoint_id="gpu0", second=now, answered=1, input_tokens=300,
                                 paid_input_tokens=0),
                     ServedCount(hotkey="c", endpoint_id="gpu0", second=now, answered=9, input_tokens=900,
                                 paid_input_tokens=900)]  # an endpoint the auditor does not test: not counted
    second_gateway = [ServedCount(hotkey="a", endpoint_id="gpu0", second=now + 1, answered=1, input_tokens=100,
                                  paid_input_tokens=100),
                      ServedCount(hotkey="a", endpoint_id="gpu1", second=now, answered=2, input_tokens=400,
                                  paid_input_tokens=400),
                      ServedCount(hotkey="b", endpoint_id="gpu0", second=now, answered=2, input_tokens=500)]
    assert auditor.accept_served(first_gateway, report_id="g1") == 2
    assert auditor.accept_served(second_gateway, report_id="g2") == 3
    assert auditor.accept_served(second_gateway, report_id="g2") == 0  # the same report again adds nothing
    auditor.chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    signed = auditor.verdicts(4)
    assert paid_served(signed.verdicts) == [PaidServed(hotkey="a", endpoint_id="gpu0", model_id="kev-4b",
                                                       input_tokens=700),
                                            PaidServed(hotkey="a", endpoint_id="gpu1", model_id="kev-4b",
                                                       input_tokens=400)]
    assert signed.verdicts.model_extra["paid_demand"]["kev-4b"] > 0
    assert SignedVerdicts.model_validate_json(signed.model_dump_json()).verdicts == signed.verdicts  # as served
    assert auditor.store.paid(3, 3) == {("a", "gpu0", "kev-4b"): 700, ("a", "gpu1", "kev-4b"): 400}
    auditor.chain.head_block = 5 * 180 + 1  # an epoch without paid traffic
    auditor.follower.poll()
    assert paid_served(auditor.verdicts(5).verdicts) == []


def test_verdicts_without_paid_served_read_as_none_paid() -> None:
    from verathos.throughput.feed import Verdicts, paid_served

    plain = Verdicts(netuid=405, auditor="x", epoch=4, credits=[], invalid=[])
    assert paid_served(plain) == []
    malformed = Verdicts(netuid=405, auditor="x", epoch=4, credits=[], invalid=[], paid_served=[{"hotkey": "a"}])
    assert paid_served(malformed) == [] and malformed.model_extra["paid_served"]  # not well formed


def test_the_models_shares_follow_paid_demand_and_move_a_step_per_epoch(tmp_path) -> None:
    """A burst of demand for one model, paid or not, moves the published shares by at most ``max_share_step`` per
    epoch; once the gateways tell paid from unpaid, unpaid traffic moves them not at all."""
    import time as time_module

    from verathos.protocol.gateway import ServedCount
    from verathos.throughput.feed import published_shares
    from verathos.validator.__main__ import epoch_shares

    cat = two_active_and_a_candidate()  # anchors 50/50
    auditor = _auditor(tmp_path, cat)
    auditor.follower.poll()  # epoch 3
    auditor.update_slots([Slot("a", "gpu0", "http://a/e/gpu0", "kev-4b", 50.0),
                          Slot("b", "gpu0", "http://b/e/gpu0", "kev-9b", 50.0)])
    step, epoch_s = cat.scoring.max_share_step, 180 * 12.0

    def epoch(number: int, paid: int | None, unpaid: int) -> dict[str, float]:
        """The verdicts' shares for epoch ``number`` after users sent kev-9b ``paid`` and ``unpaid`` input tokens/s
        in the epoch before (``paid`` None: the gateway does not tell them apart)."""
        tokens = int(epoch_s * ((paid or 0) + unpaid))
        auditor.accept_served([ServedCount(hotkey="b", endpoint_id="gpu0", second=int(time_module.time()),
                                           answered=10_000, input_tokens=tokens,
                                           paid_input_tokens=None if paid is None else int(epoch_s * paid))])
        auditor.chain.head_block = number * 180 + 1
        auditor.follower.poll()
        return published_shares(auditor.verdicts(number).verdicts)

    first = epoch(4, None, 1_000_000)  # far beyond demand_full_tokens_s, none of it told paid or not
    assert first == pytest.approx({"kev-4b": 0.5 - step, "kev-9b": 0.5 + step})
    second = epoch(5, None, 0)
    assert second["kev-9b"] == pytest.approx(0.5 + 2 * step)
    third = epoch(6, 0, 1_000_000)  # the gateways tell paid from unpaid now: none of this is paid
    assert auditor.verdicts(6).verdicts.model_extra["paid_demand"] == {}
    assert third["kev-9b"] == pytest.approx(0.5 + step)  # back toward the anchors, a step at a time
    fourth = epoch(7, 1_000_000, 0)
    assert fourth["kev-9b"] == pytest.approx(0.5 + 2 * step)
    assert epoch_shares(auditor.verdicts(7).verdicts, cat) == fourth  # what every validator scores with


def test_shares_move_along_the_line_to_their_target() -> None:
    from verathos.validator.scoring import bounded_shares

    moved = bounded_shares({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.1, "b": 0.5, "c": 0.4}, 0.1)
    assert moved == pytest.approx({"a": 0.4, "b": 0.35, "c": 0.25}) and sum(moved.values()) == pytest.approx(1.0)
    assert bounded_shares({"a": 0.5, "b": 0.5}, {"a": 0.52, "b": 0.48}, 0.05) == {"a": 0.52, "b": 0.48}
    assert bounded_shares({"a": 1.0}, {"a": 0.6, "b": 0.4}, 0.05) == {"a": 0.6, "b": 0.4}  # a model activated


def test_validators_score_with_the_shares_the_auditor_publishes_or_its_demand() -> None:
    from verathos.throughput.feed import Verdicts
    from verathos.validator.__main__ import epoch_shares
    from verathos.validator.scoring import model_shares

    cat = two_active_and_a_candidate()
    demand = {"kev-4b": 30_000.0, "kev-9b": 10_000.0}
    old_auditor = Verdicts(netuid=405, auditor="aud", epoch=9, credits=[], invalid=[], demand=demand)
    assert epoch_shares(old_auditor, cat) == model_shares(demand, cat)
    published = Verdicts.model_validate({**old_auditor.model_dump(), "shares": {"kev-4b": 0.55, "kev-9b": 0.45}})
    assert epoch_shares(published, cat) == {"kev-4b": 0.55, "kev-9b": 0.45}
    for bad in ({"kev-4b": 0.9, "kev-9b": 0.9}, {"kev-4b": 1.0}, {"kev-4b": "1", "kev-9b": 0}):
        assert epoch_shares(Verdicts.model_validate({**old_auditor.model_dump(), "shares": bad}), cat) == \
            model_shares(demand, cat)
    assert epoch_shares(None, cat) == model_shares({}, cat)


def test_an_auditor_database_one_version_old_is_migrated_not_dropped(tmp_path) -> None:
    import sqlite3

    from verathos.throughput.auditor import SCHEMA_VERSION, Store

    Store(tmp_path / "old.sqlite", epoch_blocks=180)  # today's, less what versions 5 on added
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.executescript("DROP TABLE demand; DROP TABLE users; DROP TABLE meta; DROP TABLE shares; DROP TABLE paid; "
                     "DROP INDEX results_by_endpoint; DROP INDEX samples_by_test; "
                     "ALTER TABLE samples DROP COLUMN test_id; "
                     "INSERT INTO tests VALUES ('kept', 3, 560, 'h', 565, 'confirmed', ''); "
                     "PRAGMA user_version = 4;")
    db.close()
    store = Store(tmp_path / "old.sqlite", epoch_blocks=180)
    assert store._query("SELECT test_id FROM tests") == [("kept",)]
    assert store._query("PRAGMA user_version")[0][0] == SCHEMA_VERSION
    assert store.demand_ema(10) == {}
    assert "test_id" in {column[1] for column in store._query("PRAGMA table_info(samples)")}
    assert store.paid(0, 10) == {}


def test_verdicts_take_a_few_queries_whatever_the_number_of_endpoints(tmp_path) -> None:
    """The drift tallies are grouped queries: building the credits of 4,096 endpoints with 30 epochs of tests takes
    a handful of statements, not two per endpoint."""
    import time as time_module

    auditor = _auditor(tmp_path)
    store = auditor.store
    endpoints = [(f"hk{h}", f"gpu{e}") for h in range(128) for e in range(32)]
    tests = [ScheduledTest(epoch, epoch * 180 + 20, block_hash(epoch * 180 + 20).hex(), epoch * 180 + 25)
             for epoch in range(30)]
    with store._lock:
        store._db.execute("BEGIN")
        store._db.executemany("INSERT INTO tests VALUES (?, ?, ?, ?, ?, 'confirmed', '')",
                              [(t.test_id, t.epoch, t.selection_block, t.selection_hash, t.start_block) for t in tests])
        store._db.executemany(
            "INSERT INTO results VALUES (?, ?, ?, 'kev-4b', 20, 20, 20, 20, 'passed', '', 400, 900, '{}', NULL, ?, 40)",
            [(t.test_id, hk, ep, 0.4 if hk == "hk7" else 0.0) for t in tests for hk, ep in endpoints])
        store._db.execute("COMMIT")
    statements = []
    store._db.set_trace_callback(statements.append)
    started = time_module.monotonic()
    credits = store.credits(28, 29, auditor.policy, lambda m: KEV.tolerance)
    took = time_module.monotonic() - started
    store._db.set_trace_callback(None)
    assert len(credits) == len(endpoints) and len(statements) <= 6
    assert {c.hotkey for c in credits if c.drift} == {"hk7"}  # 320 probabilities off by 0.01 each: beyond mean_abs
    assert took < 3.0


def test_a_sample_check_never_blocks_the_auditors_event_loop_while_verdicts_are_built(tmp_path) -> None:
    import threading
    import time as time_module

    gateway, miner_kp = kp(), kp()
    honest = miner(miner_kp, MockEngine("kev-4b"), gateway)
    response = asyncio.run(exchange(honest, gateway, miner_kp, request(1)))
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    auditor.update_slots([Slot(miner_kp.ss58_address, "gpu0", "http://x/e/gpu0", "kev-4b", 40.0)])
    sample = GatewaySample(hotkey=miner_kp.ss58_address, endpoint_id="gpu0", model_id="kev-4b", request=request(1),
                           response=response.text)
    assert auditor.accept_samples(gateway.ss58_address, [sample]) == 1
    auditor._verdicts_lock.acquire()  # verdicts being built in another thread
    threading.Timer(0.5, auditor._verdicts_lock.release).start()

    async def run() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        started = time_module.monotonic()
        await auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {}))
        task.cancel()
        assert time_module.monotonic() - started >= 0.4  # the check waited for the verdicts ...
        return ticks

    assert asyncio.run(run()) > 20  # ... while the loop went on
    assert auditor.store._query("SELECT state FROM samples")[0][0] == "checked"
