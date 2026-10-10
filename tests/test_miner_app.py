"""Miner service end to end with a mock engine: discovery, auth, answers, receipts, identity, test sessions."""

from __future__ import annotations

import asyncio
import json
import secrets
import time

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.app import IDENTITY_DOMAIN, create_app, identity_payload
from verathos.miner.config import MinerConfig
from verathos.protocol import signing
from verathos.protocol.api import distributions
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, accept, commitment_value
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import sign_request
from verathos.throughput.policy import HDR_REQUEST_ID, Session, session_headers

pytestmark = pytest.mark.usefixtures("lab_network")

REQUEST = {
    "model": "kev-4b",
    "state": "Hi, we were billed twice for March. Please refund today or we cancel.",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    },
}


AUDITOR = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
PIN = "ab" * 32  # the miner machine's certificate, as its manifest names it


@pytest.fixture()
def setup():
    miner, validator, stranger = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(3))
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [
            {"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:1", "max_concurrency": 4},
        ],
    })
    cat_digest = catalog.load().digest()
    manifest = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=cat_digest,
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)], tls={config.base_url: PIN}), miner)
    app = create_app(config=config, keypair=miner, manifest=lambda: manifest, engines={"gpu0": MockEngine("kev-4b")},
                     authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: {AUDITOR.ss58_address},
                     max_state_tokens=lambda model_id: 200)
    return TestClient(app), miner, validator, stranger, manifest


def post(client: TestClient, caller: Keypair, miner: Keypair, path: str, payload: dict):
    body = json.dumps(payload).encode()
    return client.post(path, content=body, headers={"Content-Type": "application/json",
                                                    **sign_request(caller, "POST", path, body, miner.ss58_address)})


def test_manifest_is_public_and_acceptable(setup) -> None:
    client, miner, *_ , manifest = setup
    assert accept(client.get("/v1/manifest").json(), hotkey=miner.ss58_address, netuid=405,
                  commitment=commitment_value(manifest.manifest.digest()),
                  catalogs={manifest.manifest.catalog_digest: 1}, axon="https://203.0.113.5:8091", seen=PIN)


def test_answer_carries_binding_receipt(setup) -> None:
    client, miner, validator, *_ = setup
    r = post(client, validator, miner, "/e/gpu0/v1/systemone", REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    receipt = SignedReceipt.model_validate(body["receipt"])
    assert receipt.signature_valid()
    assert receipt.binds(REQUEST, body["answers"])
    assert receipt.receipt.caller_hotkey == validator.ss58_address
    assert set(distributions(body["answers"])) == {"team", "urgency", "churn"}


def test_unsigned_and_unpermitted_callers_are_refused(setup) -> None:
    from verathos.miner.app import MAX_REQUEST_BYTES
    from verathos.protocol.requests import HDR_HOTKEY

    client, miner, validator, stranger, _ = setup
    assert client.post("/e/gpu0/v1/systemone", json=REQUEST).status_code == 403
    assert post(client, stranger, miner, "/e/gpu0/v1/systemone", REQUEST).status_code == 403
    forged = client.post("/e/gpu0/v1/systemone", json=REQUEST, headers={HDR_HOTKEY: validator.ss58_address})
    assert forged.status_code == 401
    big = {**REQUEST, "state": "x" * MAX_REQUEST_BYTES}
    assert post(client, validator, miner, "/e/gpu0/v1/systemone", big).status_code == 413


def test_a_body_nested_deeper_than_the_parser_goes_is_an_invalid_request(setup) -> None:
    client, miner, validator, *_ = setup
    path = "/e/gpu0/v1/systemone"
    body = b'{"model": "kev-4b", "state": ' + b"[" * 100_000 + b"]" * 100_000 + b', "questions": {}}'
    r = client.post(path, content=body, headers={"Content-Type": "application/json",
                                                 **sign_request(validator, "POST", path, body, miner.ss58_address)})
    assert r.status_code == 422, r.text


def test_wrong_model_and_unknown_endpoint(setup) -> None:
    client, miner, validator, *_ = setup
    assert post(client, validator, miner, "/e/gpu0/v1/systemone", {**REQUEST, "model": "kev-9b"}).status_code == 422
    assert post(client, validator, miner, "/e/gpu9/v1/systemone", REQUEST).status_code == 404


def test_identity_challenge_proves_endpoint_control(setup) -> None:
    client, miner, validator, *_ = setup
    challenge = "ab" * 16
    r = post(client, validator, miner, "/e/gpu0/v1/identity", {"challenge": challenge})
    assert r.status_code == 200, r.text
    payload = identity_payload("gpu0", challenge, validator.ss58_address)
    assert signing.verify(miner.ss58_address, IDENTITY_DOMAIN, payload, r.json()["signature"])


def session_post(client: TestClient, miner: Keypair, *, auditor: Keypair = AUDITOR, endpoint_id: str = "gpu0",
              request_id: str | None = None, expires_in_ms: int = 30_000, payload: dict = REQUEST):
    session = Session(netuid=405, auditor=auditor.ss58_address, miner_hotkey=miner.ss58_address,
                      endpoint_id=endpoint_id, test_id="t" * 64, not_after_ms=int(time.time() * 1000) + expires_in_ms)
    headers = {**session_headers(session, auditor), HDR_REQUEST_ID: request_id or secrets.token_hex(16)}
    return client.post(f"/e/{endpoint_id}/v1/systemone", json=payload, headers=headers)


def test_test_traffic_carries_the_auditors_session(setup) -> None:
    client, miner, *_ = setup
    request_id = secrets.token_hex(16)
    r = session_post(client, miner, request_id=request_id)
    assert r.status_code == 200, r.text
    receipt = SignedReceipt.model_validate(r.json()["receipt"])
    assert receipt.signature_valid() and receipt.binds(REQUEST, r.json()["answers"])
    assert (receipt.receipt.caller_hotkey, receipt.receipt.request_nonce) == (AUDITOR.ss58_address, request_id)
    assert session_post(client, miner, request_id=request_id).json()["error"] == "replayed request"


def test_test_sessions_are_checked(setup) -> None:
    client, miner, validator, *_ = setup
    assert "another auditor" in session_post(client, miner, auditor=validator).json()["error"]
    assert session_post(client, miner, expires_in_ms=-30_000).status_code == 200  # within the clock skew
    assert "expired" in session_post(client, miner, expires_in_ms=-61_000).json()["error"]
    assert session_post(client, miner, request_id="not-hex").json()["error"] == "malformed request id"
    forged = Session(netuid=405, auditor=AUDITOR.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                     test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    headers = {**session_headers(forged, validator), HDR_REQUEST_ID: secrets.token_hex(16)}
    r = client.post("/e/gpu0/v1/systemone", json=REQUEST, headers=headers)
    assert (r.status_code, r.json()["error"]) == (401, "bad test session signature")


def test_the_miner_measures_itself_through_its_own_sessions(setup) -> None:
    client, miner, validator, *_ = setup
    assert session_post(client, miner, auditor=miner).status_code == 200
    session = Session(netuid=405, auditor=AUDITOR.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                      test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    headers = session_headers(session, AUDITOR)
    headers["X-Verathos-Session-Signature"] = headers["X-Verathos-Session-Signature"].upper()
    r = client.post("/e/gpu0/v1/systemone", json=REQUEST, headers={**headers, HDR_REQUEST_ID: secrets.token_hex(16)})
    assert (r.status_code, r.json()["error"]) == (401, "bad test session signature")


def test_sessions_cannot_outlast_the_replay_protection(setup) -> None:
    client, miner, *_ = setup
    assert "lasts too long" in session_post(client, miner, expires_in_ms=3_600_000).json()["error"]


def test_states_longer_than_the_model_accepts_are_refused(setup) -> None:
    client, miner, validator, *_ = setup
    r = post(client, validator, miner, "/e/gpu0/v1/systemone", {**REQUEST, "state": "word " * 200})
    assert (r.status_code, r.json()["error"]) == (413, "state, instructions and criteria longer than the model accepts")
    long_question = {"type": "noul", "instructions": "word " * 200}  # a short state: the questions count too
    r = post(client, validator, miner, "/e/gpu0/v1/systemone", {**REQUEST, "questions": {"q": long_question}})
    assert r.status_code == 413


def test_the_miner_signs_only_well_formed_answers() -> None:
    from verathos.protocol.api import SystemOneRequest, well_formed

    request = SystemOneRequest.model_validate(REQUEST)
    good = asyncio.run(MockEngine("kev-4b").answer(REQUEST))["answers"]
    assert well_formed(request, good)
    assert not well_formed(request, {**good, "churn": {"type": "noul", "noul": 1.02}})
    assert not well_formed(request, {k: v for k, v in good.items() if k != "churn"})
    assert not well_formed(request, {**good, "team": {"type": "choice", "probabilities": {"billing": 1.0}}})


def test_the_fields_users_read_must_agree_with_the_distribution() -> None:
    """choice, noul and score are functions of the distribution; an answer carries only its type's fields."""
    from verathos.protocol.api import SystemOneRequest, known_fields, well_formed

    request = SystemOneRequest.model_validate(REQUEST)
    good = asyncio.run(MockEngine("kev-4b").answer(REQUEST))["answers"]
    team, urgency, p = good["team"], good["urgency"], good["churn"]["noul"]
    least = min(team["probabilities"], key=team["probabilities"].get)
    assert team["probabilities"][least] < team["probabilities"][team["choice"]] - 0.01 and abs(1 - 2 * p) > 0.01
    for wrong in ({"team": {**team, "choice": least}},
                  {"team": {**team, "choice": "nobody"}},
                  {"churn": {"type": "noul", "noul": 1 - p, "probabilities": {"false": 1 - p, "true": p}}},
                  {"churn": {"type": "noul", "probabilities": {"false": 1 - p, "true": p}}},
                  {"urgency": {**urgency, "score": urgency["score"] + 0.5}},
                  {"urgency": {**urgency, "score": "1"}},
                  {"urgency": {**urgency, "legend": {"0": "Can wait"}}},
                  {"team": {**team, "type": "score"}},
                  {"team": {**team, "confidence": 1.5}},
                  {"team": {**team, "note": "visit example.com"}}):
        assert not well_formed(request, {**good, **wrong}), wrong
    tied = {"billing": 0.5, "technical": 0.5}
    for fine in ({"team": {**team, "confidence": 0.9}},
                 {"urgency": {**urgency, "legend": {"0": "Can wait", "1": "This week", "2": "Today"}}},
                 {"churn": {"type": "noul", "noul": p, "probabilities": {"false": 1 - p, "true": p}}},
                 {"team": {"type": "choice", "choice": "technical", "probabilities": tied}}):
        assert well_formed(request, {**good, **fine}), fine
    extra = {**good, "team": {**team, "note": "x"}}
    assert known_fields(request, extra) == good and well_formed(request, known_fields(request, extra))


def test_an_engines_extra_fields_are_not_signed(setup) -> None:
    _, miner, validator, *_ = setup

    class Chatty(MockEngine):
        async def answer(self, request):
            response = await super().answer(request)
            response["answers"]["team"]["explanation"] = "because"
            return response

    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:1"}]})
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=10.0)], tls={config.base_url: PIN}), miner)
    chatty = TestClient(create_app(config=config, keypair=miner, manifest=lambda: signed,
                                   engines={"gpu0": Chatty("kev-4b")}, authorized=lambda hk: True,
                                   auditors=lambda: set()))
    r = post(chatty, validator, miner, "/e/gpu0/v1/systemone", REQUEST)
    assert r.status_code == 200 and "explanation" not in r.json()["answers"]["team"]
    assert SignedReceipt.model_validate(r.json()["receipt"]).binds(REQUEST, r.json()["answers"])


class Slow(MockEngine):
    """Answers after ``delay_s``, recording the order of the states it answered."""

    def __init__(self, model_id: str, delay_s: float = 0.3) -> None:
        super().__init__(model_id)
        self.delay_s, self.answered = delay_s, []

    async def answer(self, request):
        await asyncio.sleep(self.delay_s)
        self.answered.append(request["state"])
        return await super().answer(request)


def capped_miner(engine: MockEngine, *, testing=lambda: False, admission_limit: int | None = None):
    """A miner whose validator and gateway are known, with the per-caller limits of a real miner."""
    from verathos.miner.admission import Admission
    from verathos.miner.app import caller_request_bytes

    miner, validator, gateway = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(3))
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:1"}]})
    manifest = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)], tls={config.base_url: PIN}), miner)
    cat = catalog.load()
    app = create_app(config=config, keypair=miner, manifest=lambda: manifest, engines={"gpu0": engine},
                     authorized=lambda hk: hk in (validator.ss58_address, gateway.ss58_address),
                     auditors=lambda: {AUDITOR.ss58_address}, gateways=lambda: {gateway.ss58_address},
                     admissions={"gpu0": Admission(admission_limit)} if admission_limit else None,
                     max_state_tokens=lambda m: cat.get(m).max_state_tokens,
                     max_caller_request=lambda m: caller_request_bytes(m, cat.get(m).workload), testing=testing)
    return app, miner, validator, gateway


async def signed_post(client: httpx.AsyncClient, caller: Keypair, miner: Keypair, payload: dict) -> httpx.Response:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               **sign_request(caller, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address)}
    return await client.post("/e/gpu0/v1/systemone", content=body, headers=headers)


def test_a_validator_cannot_crowd_out_a_test_with_many_requests() -> None:
    """Callers other than gateways get one request at once per endpoint."""
    app, miner, validator, gateway = capped_miner(Slow("kev-4b"))

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
            validators = await asyncio.gather(*(signed_post(client, validator, miner, REQUEST) for _ in range(5)))
            gateways = await asyncio.gather(*(signed_post(client, gateway, miner, REQUEST) for _ in range(5)))
            return [r.status_code for r in validators], [r.status_code for r in gateways]

    validators, gateways = asyncio.run(run())
    assert sorted(validators) == [200] + [429] * 4 and gateways == [200] * 5


def test_a_validators_requests_are_no_larger_than_the_models_standard_ones() -> None:
    """A validator cannot make each of its requests cost many test requests: its canaries pass, a request beyond the
    largest of the model's request mix does not, while a gateway's users may send states up to the model's limit."""
    from verathos.validator.canaries import canary_pool

    app, miner, validator, gateway = capped_miner(MockEngine("kev-4b"))
    long_state = {**REQUEST, "state": "word " * 13_000}
    padded = {**REQUEST, "questions": {**REQUEST["questions"], "pad": {"type": "noul", "instructions": "x" * 20_000}}}

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
            canaries = [(await signed_post(client, validator, miner, c)).status_code
                        for c in canary_pool(b"k" * 32, 7, "kev-4b", 16)]
            refused = [await signed_post(client, validator, miner, p) for p in (long_state, padded)]
            served = await signed_post(client, gateway, miner, long_state)
            return canaries, refused, served.status_code

    canaries, refused, served = asyncio.run(run())
    assert canaries == [200] * 16 and served == 200
    assert [(r.status_code, r.json()["error"]) for r in refused] == [
        (413, "request larger than the model's standard requests")] * 2


def test_canaries_and_test_requests_fit_the_bound_for_validators() -> None:
    import random

    from verathos.miner.app import caller_request_bytes
    from verathos.protocol.canonical import canonical_json
    from verathos.throughput.workload import Workload, make_request
    from verathos.validator.canaries import canary_pool

    kev = catalog.load().get("kev-4b")
    bound = caller_request_bytes("kev-4b", kev.workload)
    largest = max(len(canonical_json(c)) for e in range(200) for c in canary_pool(b"s" * 32, e, "kev-4b", 16))
    assert largest <= bound and largest < 1_000 < 16_000 < bound  # canaries are a few hundred characters
    rng = random.Random(1)
    assert all(len(canonical_json(make_request("kev-4b", kev.workload, rng))) <= bound for _ in range(200))
    tiny = Workload(state_words=(2, 3), questions={"q": {"type": "noul"}})
    assert caller_request_bytes("kev-4b", tiny) >= largest  # a small mix never shuts out canaries


def test_while_test_traffic_arrives_a_validators_request_waits_behind_it() -> None:
    engine = Slow("kev-4b", delay_s=0.1)
    app, miner, validator, gateway = capped_miner(engine, testing=lambda: True, admission_limit=1)
    session = Session(netuid=405, auditor=AUDITOR.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                      test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as client:
            def tagged(tag: str) -> dict:
                return {**REQUEST, "state": tag}

            async def test_request():
                headers = {**session_headers(session, AUDITOR), HDR_REQUEST_ID: secrets.token_hex(16)}
                return await client.post("/e/gpu0/v1/systemone", json=tagged("test"), headers=headers)

            first = asyncio.create_task(signed_post(client, gateway, miner, tagged("user")))
            await asyncio.sleep(0.02)
            held = asyncio.create_task(signed_post(client, validator, miner, tagged("canary")))
            await asyncio.sleep(0.02)
            test = asyncio.create_task(test_request())
            return [r.status_code for r in await asyncio.gather(first, held, test)]

    assert asyncio.run(run()) == [200, 200, 200]
    assert engine.answered == ["user", "test", "canary"]


class Recording(MockEngine):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.received: list[dict] = []

    async def answer(self, request):
        self.received.append(request)
        return await super().answer(request)


@pytest.fixture()
def seeing():
    """A miner with an endpoint of a model that reads images (gpu0) and one of a model that reads text only (gpu1)."""
    from verathos.protocol.media import ImageLimits

    miner, validator = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(2))
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "clef-flash", "engine_url": "http://127.0.0.1:1"},
                      {"endpoint_id": "gpu1", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:2"}]})
    manifest = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id=e, url=config.endpoint_url(e), model_id=m, capacity_rps=40.0)
                   for e, m in (("gpu0", "clef-flash"), ("gpu1", "kev-4b"))], tls={config.base_url: PIN}), miner)
    engines = {"gpu0": Recording("clef-flash"), "gpu1": Recording("kev-4b")}
    app = create_app(config=config, keypair=miner, manifest=lambda: manifest, engines=engines,
                     authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: set(),
                     gateways=lambda: {validator.ss58_address},
                     image_limits=lambda model_id: ImageLimits() if model_id == "clef-flash" else None)
    return TestClient(app), miner, validator, engines


IMAGE_REQUEST = {"model": "clef-flash", "state": "What is in the image?",
                 "questions": {"corner": {"type": "noul", "instructions": "Is there a green corner?"}}}


def test_canonical_images_reach_the_engine_of_a_model_that_reads_them(seeing) -> None:
    from tests.images import canonical

    client, miner, validator, engines = seeing
    request = {**IMAGE_REQUEST, "images": [canonical(256, 320), canonical(1024, 768, "JPEG")]}
    r = post(client, validator, miner, "/e/gpu0/v1/systemone", request)
    assert r.status_code == 200, r.text
    assert engines["gpu0"].received == [request]
    assert SignedReceipt.model_validate(r.json()["receipt"]).binds(request, r.json()["answers"])


def test_images_that_are_not_canonical_or_not_for_the_model_never_reach_an_engine(seeing) -> None:
    from tests.images import b64, canonical, data_url, encoded, item

    client, miner, validator, engines = seeing
    refused = [("http://169.254.169.254/latest/meta-data/", "images.0: not {"),
               ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "images.0: not {"),
               ("/etc/passwd", "images.0: not {"), (b64(encoded(256, 256)), "images.0: not {"),
               (data_url(encoded(256, 256), "image/png"), "images.0: not {"),
               (item(encoded(256, 256, "WEBP"), "image/webp"), "images.0: not {"),
               (item(encoded(1056, 1024), "image/png"), "images.0: 1056 x 1024 pixels, outside"),
               (item(encoded(256, 256, "JPEG"), "image/jpeg"), "images.0: not in the canonical form")]
    for image, why in refused:
        r = post(client, validator, miner, "/e/gpu0/v1/systemone", {**IMAGE_REQUEST, "images": [image]})
        assert r.status_code == 422 and why in r.json()["error"], r.text
    five = {**IMAGE_REQUEST, "images": [canonical(256, 256)] * 5}
    assert "at most 4 images" in post(client, validator, miner, "/e/gpu0/v1/systemone", five).json()["error"]
    text_only = {**REQUEST, "images": [canonical(256, 256)]}
    r = post(client, validator, miner, "/e/gpu1/v1/systemone", text_only)
    assert r.status_code == 422 and r.json()["error"] == "kev-4b reads text only"
    for refused_field in ({"videos": []}, {"media_kwargs": {"do_resize": True}}):
        assert post(client, validator, miner, "/e/gpu0/v1/systemone", {**IMAGE_REQUEST, **refused_field}
                    ).status_code == 422
    assert engines["gpu0"].received == [] and engines["gpu1"].received == []


def test_a_body_is_held_to_its_models_limit(seeing) -> None:
    from verathos.miner.app import MAX_REQUEST_BYTES

    client, miner, validator, engines = seeing
    long_state = "x" * (2 * MAX_REQUEST_BYTES)
    assert post(client, validator, miner, "/e/gpu1/v1/systemone", {**REQUEST, "state": long_state}).status_code == 413
    assert post(client, validator, miner, "/e/gpu0/v1/systemone",
                {**IMAGE_REQUEST, "state": long_state}).status_code == 200


def test_a_caller_that_leaves_mid_body_gets_no_500_and_the_miner_shows_no_api_documents(setup, caplog) -> None:
    """A claimed caller is enough to have the body read: one that leaves before it is whole would otherwise log an
    error with a traceback each time, which anyone can repeat. The miner serves no API documents either."""
    import logging

    from tests.disconnect import left_mid_body
    from verathos.protocol.requests import HDR_HOTKEY

    client, miner, validator, _, _ = setup
    with caplog.at_level(logging.ERROR):
        status = asyncio.run(left_mid_body(client.app, "/e/gpu0/v1/systemone",
                                           {HDR_HOTKEY: validator.ss58_address, "content-type": "application/json"}))
    assert status == 413 and not caplog.records
    assert [client.get(path).status_code for path in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]


def test_a_timestamp_longer_than_a_time_is_malformed_not_an_error() -> None:
    from verathos.protocol.requests import NonceCache, RequestRejected, verify_request

    caller, receiver = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(2))
    headers = sign_request(caller, "POST", "/e/gpu0/v1/systemone", b"{}", receiver.ss58_address)
    stamp = next(k for k in headers if k.lower().endswith("timestamp"))
    with pytest.raises(RequestRejected, match="malformed timestamp"):
        verify_request({**headers, stamp: "9" * 5000}, "POST", "/e/gpu0/v1/systemone", b"{}", receiver.ss58_address,
                       NonceCache())
