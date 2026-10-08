"""The gateway end to end against in-process miners and an in-process auditor, on SQLite: keys, limits, routing,
retries, breakers, receipts, usage, and sampled answers proving a wrong one."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tests.mock_engine import MockEngine
from tests.test_miner_app import REQUEST
from tests.test_throughput import HostRouter, _auditor, kp
from verathos import catalog
from verathos.gateway import db
from verathos.gateway.accounts import Accounts
from verathos.gateway.app import Gateway, client_address, create_app
from verathos.gateway.config import GatewayConfig
from verathos.gateway.limits import RateLimiter
from verathos.gateway.router import BREAKER_FAILURES, BUSY_S, OPEN_S, Outcome, Router
from verathos.gateway.routes import RouteTable
from verathos.gateway.sampler import Sampler
from verathos.gateway.served import ServedCounter
from verathos.gateway.upstream import Upstream
from verathos.gateway.usage import UsageMeter
from verathos.miner.app import create_app as miner_app
from verathos.miner.config import MinerConfig
from verathos.miner.engine import EngineError
from verathos.protocol.gateway import Route, Routes, SignedRoutes
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest
from verathos.protocol.receipts import SignedReceipt
from verathos.throughput.auditor import Checks, Slot


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Failing:
    async def answer(self, request):
        raise EngineError("down")


class Slow(MockEngine):
    def __init__(self, model: str, delay: float) -> None:
        super().__init__(model)
        self.delay = delay

    async def answer(self, request):
        await asyncio.sleep(self.delay)
        return await super().answer(request)


def miner(host: str, gateway: Keypair, engine) -> tuple[Keypair, FastAPI]:
    keypair = kp()
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": 8091,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://unused",
                       "max_concurrency": 64}]})
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)]), keypair)
    app = miner_app(config=config, keypair=keypair, manifest=lambda: signed, engines={"gpu0": engine},
                    authorized=lambda hk: hk == gateway.ss58_address, auditors=lambda: set(),
                    gateways=lambda: {gateway.ss58_address})
    return keypair, app


def route(keypair: Keypair, host: str, credited: float = 20.0) -> Route:
    return Route(hotkey=keypair.ss58_address, endpoint_id="gpu0", model_id="kev-4b", url=f"http://{host}:8091/e/gpu0",
                 credited_rps=credited, latency_ms=300.0)


class Setup:
    """A gateway on SQLite in front of the given miners (host -> engine), with an in-process auditor."""

    def __init__(self, tmp_path, engines: dict[str, object], *, credited: float = 20.0, rate: float = 100.0,
                 burst: int = 100, monthly_quota: int | None = None, max_attempts: int = 3,
                 attempt_timeout_s: float = 2.0) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.gateway_kp = kp()
        self.miners = {host: miner(host, self.gateway_kp, engine) for host, engine in engines.items()}
        url = f"sqlite+aiosqlite:///{tmp_path}/gateway.db"
        db.upgrade(url)
        self.engine = db.create_engine(url)
        self.config = GatewayConfig(netuid=405, wallet_name="w", hotkey_name="h", database_url=url,
                                    max_attempts=max_attempts, attempt_timeout_s=attempt_timeout_s)
        self.auditor = _auditor(tmp_path)
        self.auditor.follower.poll()
        self.auditor.update_slots([Slot(keypair.ss58_address, "gpu0", f"http://{host}:8091/e/gpu0", "kev-4b", 40.0)
                                   for host, (keypair, _) in self.miners.items()])
        auditor_app = FastAPI()
        from verathos.validator.service import add_gateway_routes
        add_gateway_routes(auditor_app, self.auditor, probation=list, gateways=lambda: [self.gateway_kp.ss58_address])
        apps = {host: app for host, (_, app) in self.miners.items()} | {"auditor": auditor_app}
        self.client = httpx.AsyncClient(transport=HostRouter(apps))
        self.meter = UsageMeter(self.engine)
        self.accounts = Accounts(self.engine, self.meter, ttl_s=0.0, negative_ttl_s=0.0)
        self.router = Router(share=1.0, max_latency_s=2.0, rng=random.Random(1))
        routes = [route(keypair, host, credited) for host, (keypair, _) in self.miners.items()]
        self.router_table = Routes(netuid=405, auditor=self.auditor.hotkey, epoch=3, issued_ms=1,
                                   catalog_digest=catalog.load().digest(), routes=routes)
        self.router.update(self.router_table)

        async def auditor_url():
            return "http://auditor"

        self.sampler = Sampler(self.gateway_kp, self.client, auditor_url=auditor_url,
                               auditor_hotkey=self.auditor.hotkey, window_s=600, rng=random.Random(2))
        self.served = ServedCounter(self.gateway_kp, self.client, auditor_url=auditor_url,
                                    auditor_hotkey=self.auditor.hotkey)
        self.router.observer = self.served
        cat = catalog.Catalog.parse({**catalog.load().document(), "gateways": [self.gateway_kp.ss58_address]})
        self.gateway = Gateway(
            config=self.config, hotkey=self.gateway_kp.ss58_address, catalog=lambda: cat, accounts=self.accounts,
            meter=self.meter, router=self.router, sampler=self.sampler, served=self.served,
            routes=lambda: self.router_table,
            upstream=Upstream(self.gateway_kp, self.client, netuid=405, mechid=1), limiter=RateLimiter())
        self.app = create_app(self.gateway)
        self.key = asyncio.run(self._account(rate, burst, monthly_quota))

    async def _account(self, rate, burst, quota) -> str:
        await self.accounts.create_account("acme", rate_rps=rate, burst=burst, monthly_quota=quota)
        return await self.accounts.create_key("acme", "prod")

    async def post(self, payload=None, key: str | None = "default", content_override: bytes | None = None,
                   **headers) -> httpx.Response:
        auth = {"Authorization": f"Bearer {self.key if key == 'default' else key}"} if key else {}
        body = content_override or json.dumps(REQUEST if payload is None else payload).encode()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gw") as c:
            return await c.post("/v1/systemone", content=body,
                                headers={"Content-Type": "application/json", **auth, **headers})

    def many(self, n: int, payload=None) -> list[httpx.Response]:
        async def run():
            return await asyncio.gather(*(self.post(payload) for _ in range(n)))
        return asyncio.run(run())


def test_a_request_is_answered_with_the_miners_verified_receipt_and_counted(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    r = asyncio.run(s.post(**{"X-Request-Id": "client-42"}))
    assert r.status_code == 200 and r.headers["X-Request-Id"] == "client-42"
    body = r.json()
    receipt = SignedReceipt.model_validate(body["receipt"])
    assert receipt.signature_valid() and receipt.binds(REQUEST, body["answers"])
    assert receipt.receipt.caller_hotkey == s.gateway_kp.ss58_address
    assert asyncio.run(MockEngine("kev-4b").answer(REQUEST))["answers"] == body["answers"]

    asyncio.run(s.meter.flush())
    (usage,) = asyncio.run(s.accounts.usage("acme", dt.date(2026, 1, 1)))
    assert (usage["model_id"], usage["requests"], usage["failed"]) == ("kev-4b", 1, 0)


def test_keys_and_rate_limits(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")}, rate=1.0, burst=2)
    assert asyncio.run(s.post(key=None)).json()["error"]["type"] == "unauthorized"
    assert asyncio.run(s.post(key="vd_" + "0" * 64)).status_code == 401
    assert sorted(r.status_code for r in s.many(3)) == [200, 200, 429]  # a burst of 2, then 1 per second
    limited = s.many(1)[0]
    assert limited.json()["error"]["type"] == "rate_limited" and int(limited.headers["Retry-After"]) >= 1

    asyncio.run(s.accounts.set_account("acme", status="suspended"))
    assert asyncio.run(s.post()).status_code == 401
    asyncio.run(s.accounts.set_account("acme", status="active", rate_rps=100.0))
    (key,) = asyncio.run(s.accounts.list_keys("acme"))
    assert asyncio.run(s.accounts.revoke_key("acme", key["prefix"])) == 1
    assert asyncio.run(s.post()).status_code == 401


def test_monthly_quota_and_bad_requests(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")}, monthly_quota=2)
    assert [asyncio.run(s.post()).status_code for _ in range(2)] == [200, 200]
    assert asyncio.run(s.post()).json()["error"]["type"] == "quota_exceeded"

    t = Setup(tmp_path / "other", {"10.0.0.1": MockEngine("kev-4b")})
    invalid = asyncio.run(t.post({"model": "kev-4b"})).json()["error"]
    assert (invalid["type"], invalid["message"]) == ("invalid_request",
                                                     "not a SystemOne request: state: Field required (and 1 more)")
    assert "pydantic" not in asyncio.run(t.post({"model": "kev-4b", "questions": {"q": {"type": "nope"}}})).text
    assert asyncio.run(t.post({**REQUEST, "model": "clef-flash"})).json()["error"]["type"] == "model_not_found"
    gone = asyncio.run(t.post({**REQUEST, "model": "kev-9b"}))  # active, but nobody serves it now: try again later
    assert gone.status_code == 503 and gone.json()["error"]["type"] == "unavailable" and "Retry-After" in gone.headers
    too_long = {**REQUEST, "state": "x" * (4 * catalog.load().get("kev-4b").max_state_tokens + 1)}
    assert asyncio.run(t.post(too_long)).json()["error"]["type"] == "too_large"


def test_failing_miners_are_retried_elsewhere_and_their_breaker_opens(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": Failing(), "10.0.0.2": MockEngine("kev-4b")})
    responses = s.many(12)
    assert {r.status_code for r in responses} == {200}
    failing = next(e for e in s.router._pools["kev-4b"].endpoints if e.route.url.startswith("http://10.0.0.1"))
    assert failing.breaker.is_open


class Unreachable(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("No route to host", request=request)


def test_a_model_whose_every_miner_fails_is_unavailable_not_busy(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b"), "10.0.0.2": MockEngine("kev-4b")})
    for host in ("10.0.0.1", "10.0.0.2"):  # the miners' host died
        s.client._transport._t[host] = Unreachable()
    first = s.many(1)[0]
    assert first.status_code == 503 and first.json()["error"]["type"] == "unavailable"
    for _ in range(BREAKER_FAILURES):
        s.many(1)
    down = s.many(1)[0]  # every breaker open: none is answering, which is not a capacity limit
    assert down.json()["error"] == {"type": "unavailable", "message": "no miner of kev-4b answers right now"}
    assert down.headers["Retry-After"] == "30"


def test_the_status_says_per_model_what_users_get(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b"), "10.0.0.2": MockEngine("kev-4b")})

    async def status():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://gw") as c:
            r = await c.get("/v1/status")
            return r, {m["id"]: m for m in r.json()["models"]}
    assert all(r.status_code == 200 for r in s.many(5))
    response, models = asyncio.run(status())
    kev4 = models["kev-4b"]
    assert response.headers["cache-control"] == "public, max-age=15"
    assert (kev4["state"], kev4["endpoints"], kev4["capacity_rps"]) == ("operational", 2, 40.0)
    assert kev4["latency_ms"]["p50"] >= 0 and kev4["latency_ms"]["window_s"] == 900
    assert kev4["success_rate_1h"] is None and kev4["requests_1h"] == 5  # under 20 requests: no rate yet
    assert models["kev-9b"]["state"] == "unavailable" and models["kev-9b"]["latency_ms"] is None
    for host in ("10.0.0.1", "10.0.0.2"):  # the miners' host dies
        s.client._transport._t[host] = Unreachable()
    for _ in range(BREAKER_FAILURES + 1):
        s.many(1)
    _, models = asyncio.run(status())
    assert models["kev-4b"]["state"] == "unavailable" and models["kev-4b"]["availability_24h"] == 0.0


def test_an_answer_without_a_binding_receipt_is_not_returned(tmp_path) -> None:
    gateway = kp()
    forger = FastAPI()

    @forger.post("/e/gpu0/v1/systemone")
    async def forged():
        answers = (await MockEngine("kev-4b").answer(REQUEST))["answers"]
        return JSONResponse({"model": "kev-4b", "answers": answers, "receipt": {"receipt": {}, "signature": "00"}})

    s = Setup(tmp_path, {"10.0.0.2": MockEngine("kev-4b")})
    s.client._transport._t["10.0.0.9"] = httpx.ASGITransport(app=forger)
    s.router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=2, catalog_digest="0" * 64,
                           routes=[route(gateway, "10.0.0.9", 1000.0)]))
    r = asyncio.run(s.post())
    assert r.status_code == 503 and r.json()["error"]["type"] == "unavailable"


class Tampering(httpx.AsyncBaseTransport):
    """A miner's responses, rewritten by ``tamper`` (the parsed body -> the body sent)."""

    def __init__(self, inner: httpx.AsyncBaseTransport, tamper) -> None:
        self.inner, self.tamper = inner, tamper

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        r = await self.inner.handle_async_request(request)
        return httpx.Response(r.status_code, headers={"Content-Type": "application/json"},
                              stream=httpx.ByteStream(self.tamper(json.loads(await r.aread()))))


def test_users_get_only_what_the_miner_signed_of_its_receipt(tmp_path) -> None:
    """Unsigned additions to a receipt (here a NaN, which no response can carry) never reach the user; the answer is
    delivered, billed and counted once."""
    def tamper(doc):
        doc["receipt"]["note"] = "NOTE"
        doc["receipt"]["receipt"]["unsigned"] = "\ud800"
        return json.dumps(doc).replace('"NOTE"', "NaN").encode()

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    s.client._transport._t["10.0.0.1"] = Tampering(s.client._transport._t["10.0.0.1"], tamper)
    r = asyncio.run(s.post())
    assert r.status_code == 200
    receipt = r.json()["receipt"]
    assert set(receipt) == {"receipt", "signature"} and "unsigned" not in receipt["receipt"]
    assert SignedReceipt.model_validate(receipt).signature_valid()
    asyncio.run(s.meter.flush())
    (usage,) = asyncio.run(s.accounts.usage("acme", dt.date(2026, 1, 1)))
    assert usage["requests"] == 1 and s.sampler.pending()["picked"] == 1
    spaced = {**receipt, "signature": " ".join(receipt["signature"])}  # still hex to bytes.fromhex
    with pytest.raises(ValueError):
        SignedReceipt.model_validate(spaced)


def signing_miner(keypair: Keypair, answers) -> FastAPI:
    """A miner that signs whatever ``answers`` (the request -> the answers) gives, with a valid receipt."""
    from verathos.protocol.canonical import digest
    from verathos.protocol.receipts import Receipt
    from verathos.protocol.requests import HDR_HOTKEY, HDR_NONCE

    app = FastAPI()

    @app.post("/e/gpu0/v1/systemone")
    async def answer(request: Request):
        raw = json.loads(await request.body())
        given = await answers(raw)
        receipt = Receipt(netuid=405, mechid=1, miner_hotkey=keypair.ss58_address, endpoint_id="gpu0",
                          model_id="kev-4b", catalog_digest=catalog.load().digest(),
                          caller_hotkey=request.headers[HDR_HOTKEY], request_nonce=request.headers[HDR_NONCE],
                          request_sha256=digest(raw), answers_sha256=digest(given), received_ms=0, responded_ms=0,
                          input_tokens=10)
        return {"answers": given, "receipt": SignedReceipt.create(receipt, keypair).model_dump(mode="json")}

    return app


def test_answers_whose_visible_fields_disagree_with_their_distribution_are_proof(tmp_path) -> None:
    """The right probabilities with the least likely option as the choice, and a noul of 1 - p(true): users never get
    them, and the signed answers prove the miner wrong."""
    async def lying(request):
        answers = (await MockEngine("kev-4b").answer(request))["answers"]
        team, p = answers["team"], answers["churn"]["noul"]
        team["choice"] = min(team["probabilities"], key=team["probabilities"].get)
        answers["churn"] = {"type": "noul", "noul": 1 - p, "probabilities": {"false": 1 - p, "true": p}}
        return answers

    s = Setup(tmp_path, {"10.0.0.2": MockEngine("kev-4b")})
    liar = kp()
    s.client._transport._t["10.0.0.7"] = httpx.ASGITransport(app=signing_miner(liar, lying))
    s.auditor.update_slots([*s.auditor._slots, Slot(liar.ss58_address, "gpu0", "http://10.0.0.7:8091/e/gpu0",
                                                    "kev-4b", 40.0)])
    s.router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=2, catalog_digest="0" * 64, routes=[
        route(liar, "10.0.0.7", 1000.0), route(s.miners["10.0.0.2"][0], "10.0.0.2", 20.0)]))
    honest = asyncio.run(MockEngine("kev-4b").answer(REQUEST))["answers"]
    for r in s.many(4):
        assert r.status_code == 200 and r.json()["answers"] == honest
    assert s.sampler.pending()["proofs"] >= 1
    asyncio.run(s.sampler.flush())
    asyncio.run(s.auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    s.auditor.chain.head_block = 4 * 180 + 1
    s.auditor.follower.poll()
    (evidence,) = s.auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == liar.ss58_address and evidence.reason == "malformed answers"


def test_no_free_capacity_is_busy_not_a_failure(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": Slow("kev-4b", 0.3)}, credited=0.5)  # takes one request at a time
    codes = sorted(r.status_code for r in s.many(3))
    assert codes == [200, 503, 503]
    busy = [r for r in s.many(2) if r.status_code == 503][0]
    assert busy.json()["error"]["type"] == "busy" and busy.headers["Retry-After"] == "1"


def test_a_slow_miner_times_out_and_the_request_goes_elsewhere(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": Slow("kev-4b", 5.0), "10.0.0.2": MockEngine("kev-4b")},
              attempt_timeout_s=0.3)
    assert {r.status_code for r in s.many(6)} == {200}


def test_a_wrong_answer_served_to_users_is_proven_by_its_sample(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b", salt="a cheaper model")})
    for i in range(5):
        assert asyncio.run(s.post({**REQUEST, "state": f"{REQUEST['state']} {i}"})).status_code == 200
    asyncio.run(s.sampler.flush())  # one of the five, drawn after the answers
    asyncio.run(s.auditor.check_samples(Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    s.auditor.chain.head_block = 4 * 180 + 1
    s.auditor.follower.poll()
    (evidence,) = s.auditor.verdicts(4).verdicts.invalid
    assert evidence.hotkey == s.miners["10.0.0.1"][0].ss58_address


# ---- parts ----

def test_the_router_spreads_by_capacity_and_respects_caps_and_breakers() -> None:
    clock = Clock()
    router = Router(share=1.0, max_latency_s=2.0, rng=random.Random(3), clock=clock)
    big, small = kp(), kp()
    router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=1, catalog_digest="0" * 64,
                         routes=[route(big, "b", 30.0), route(small, "s", 10.0)]))
    counts = {big.ss58_address: 0, small.ss58_address: 0}
    for _ in range(2000):
        e = router.pick("kev-4b")
        counts[e.route.hotkey] += 1
    assert 0.65 < counts[big.ss58_address] / 2000 < 0.85  # draws by capacity, the better of two
    assert router.pick("kev-4b", exclude={big.ss58_address, small.ss58_address}) is None

    (b, s) = router._pools["kev-4b"].endpoints
    for _ in range(s.cap):
        router.started(s)
    assert all(router.pick("kev-4b") is b for _ in range(50))  # the small one is full
    for _ in range(s.cap):
        router.finished(s, Outcome.ANSWERED)

    for _ in range(BREAKER_FAILURES):
        router.started(b)
        router.finished(b, Outcome.FAILED)
    assert all(router.pick("kev-4b") is s for _ in range(50))
    clock.now += OPEN_S
    probe = next(e for e in (router.pick("kev-4b") for _ in range(100)) if e is b)
    router.started(probe)
    assert all(router.pick("kev-4b") is s for _ in range(50))  # one probe at a time
    router.finished(probe, Outcome.FAILED)
    clock.now += OPEN_S
    assert all(router.pick("kev-4b") is s for _ in range(50))  # a failed probe doubles the open time
    clock.now += OPEN_S
    probe = next(e for e in (router.pick("kev-4b") for _ in range(100)) if e is b)
    router.started(probe)
    router.finished(probe, Outcome.ANSWERED)
    assert not b.breaker.is_open

    router.update(Routes(netuid=405, auditor="a", epoch=4, issued_ms=2, catalog_digest="0" * 64,
                         routes=[route(big, "b", 30.0)]))
    assert router._pools["kev-4b"].endpoints == [b]  # kept with its state


class Observer:
    def __init__(self) -> None:
        self.offers: dict[str, int] = {}
        self.refusals: dict[str, int] = {}

    def offered(self, route: Route, refused: bool) -> None:
        self.offers[route.hotkey] = self.offers.get(route.hotkey, 0) + 1
        self.refusals[route.hotkey] = self.refusals.get(route.hotkey, 0) + refused

    def attempted(self, route: Route) -> None:
        pass

    def refused(self, route: Route) -> None:
        self.refusals[route.hotkey] = self.refusals.get(route.hotkey, 0) + 1


def test_an_endpoint_that_refuses_users_or_keeps_them_waiting_is_reported() -> None:
    """Each time the router considers an endpoint for a user request it reports whether it was ready: not busy, not
    behind an open breaker, its latest answer within twice its latency bound; refusals of requests count too."""
    clock, observer = Clock(), Observer()
    router = Router(share=1.0, max_latency_s=2.0, rng=random.Random(3), clock=clock, observer=observer)
    fast, slow = kp().ss58_address, kp().ss58_address
    router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=1, catalog_digest="0" * 64, routes=[
        Route(hotkey=h, endpoint_id="gpu0", model_id="kev-4b", url=f"http://{h}/e/gpu0", credited_rps=20.0,
              latency_ms=300.0) for h in (fast, slow)]))
    f, s = router._pools["kev-4b"].endpoints

    def considered(endpoint_hotkey: str, picks: int = 200) -> tuple[int, int]:
        before = observer.offers.get(endpoint_hotkey, 0), observer.refusals.get(endpoint_hotkey, 0)
        for _ in range(picks):
            router.pick("kev-4b")
        return (observer.offers.get(endpoint_hotkey, 0) - before[0],
                observer.refusals.get(endpoint_hotkey, 0) - before[1])

    offers, refusals = considered(slow)
    assert offers > 50 and refusals == 0  # ready: considered, not always chosen, never held against it
    router.started(s)
    router.finished(s, Outcome.ANSWERED, 4.5)  # it kept a user waiting beyond twice its bound
    offers, refusals = considered(slow)
    assert offers == refusals > 50 and any(router.pick("kev-4b") is s for _ in range(50))  # still routed
    router.started(s)
    router.finished(s, Outcome.ANSWERED, 0.3)
    assert considered(slow)[1] == 0  # ready again
    before = observer.refusals[slow]
    router.started(s)
    router.finished(s, Outcome.BUSY, 0.1)
    assert observer.refusals[slow] == before + 1 and s.busy_until == clock.now + BUSY_S
    offers, refusals = considered(slow)
    assert offers == refusals > 50 and all(router.pick("kev-4b") is f for _ in range(50))
    clock.now += BUSY_S
    router.started(s)
    router.finished(s, None, 1.0)  # a quick unclear failure counts only once another miner answers (penalize)
    assert considered(slow)[1] == 0
    before = observer.refusals[slow]
    router.penalize(s)
    assert observer.refusals[slow] == before + 1
    router.started(s)
    router.finished(s, None, 4.0)  # an attempt that timed out at twice the bound: slow, so not ready
    assert s.slow and considered(slow)[1] > 0
    router.started(s)
    router.finished(s, Outcome.ANSWERED, 0.2)
    for _ in range(f.cap):
        router.started(f)
    assert considered(fast) == (0, 0)  # full at speed: neither way
    for _ in range(f.cap):
        router.finished(f, Outcome.ANSWERED, 0.1)
    # requests that are not user traffic are not reported
    snapshot = dict(observer.offers), dict(observer.refusals)
    for _ in range(20):
        router.pick("kev-4b", observe=False)
    router.started(f)
    router.finished(f, Outcome.FAILED, 0.1, observe=False)
    router.penalize(f, observe=False)
    assert (observer.offers, observer.refusals) == snapshot
    assert observer.refusals.get(fast, 0) == 0


def test_gcra_allows_the_burst_then_the_rate() -> None:
    limiter = RateLimiter()
    assert [limiter.check("k", 2.0, 3, 0.0) for _ in range(3)] == [0.0] * 3
    assert limiter.check("k", 2.0, 3, 0.0) == pytest.approx(0.5)
    assert limiter.check("k", 2.0, 3, 0.5) == 0.0
    assert limiter.check("other", 2.0, 3, 0.5) == 0.0


def test_routes_are_taken_only_from_the_auditor_and_newer(tmp_path) -> None:
    auditor = kp()

    def signed(issued: int, signer: Keypair = auditor, netuid: int = 405) -> SignedRoutes:
        return SignedRoutes.create(Routes(netuid=netuid, auditor=signer.ss58_address, epoch=3, issued_ms=issued,
                                          catalog_digest="0" * 64, routes=[route(kp(), "m")]), signer)

    served = {"doc": signed(5)}
    app = FastAPI()

    @app.get("/v1/routes")
    async def get_routes():
        return served["doc"].model_dump(mode="json")

    async def url():
        return "http://auditor"

    def table() -> RouteTable:
        return RouteTable(client=httpx.AsyncClient(transport=HostRouter({"auditor": app})), auditor_url=url,
                          auditor_hotkey=auditor.ss58_address, netuid=405, path=tmp_path / "routes.json")

    t = table()
    assert asyncio.run(t.refresh()) and t.table.issued_ms == 5
    served["doc"] = signed(4)
    assert not asyncio.run(t.refresh())  # older
    served["doc"] = signed(9, signer=kp())
    assert not asyncio.run(t.refresh())  # not the auditor's
    restarted = table()
    asyncio.run(restarted.load())
    assert restarted.table.issued_ms == 5  # from disk


def test_the_client_address_is_believed_only_from_trusted_proxies() -> None:
    import ipaddress

    from starlette.requests import Request

    trusted = [ipaddress.ip_network("127.0.0.1/32")]

    def address(peer: str, forwarded: str) -> str:
        scope = {"type": "http", "client": (peer, 1), "headers": [(b"x-forwarded-for", forwarded.encode())]}
        return client_address(Request(scope), trusted)

    assert address("127.0.0.1", "198.51.100.7") == "198.51.100.7"
    assert address("127.0.0.1", "203.0.113.1, 198.51.100.7, 127.0.0.1") == "198.51.100.7"
    assert address("198.51.100.9", "203.0.113.1") == "198.51.100.9"
    # IPv4 clients seen through an IPv6 socket (::ffff:a.b.c.d) are IPv4 clients, the proxy among them too
    assert address("::ffff:127.0.0.1", "::ffff:198.51.100.7") == "198.51.100.7"
    assert address("::ffff:198.51.100.9", "203.0.113.1") == "198.51.100.9"


def test_invalid_keys_are_limited_per_ipv4_address_also_when_mapped_into_ipv6() -> None:
    from verathos.gateway.app import address_group

    assert address_group("::ffff:1.2.3.4") == "1.2.3.4" != address_group("::ffff:203.0.113.9")
    assert address_group("2001:db8::1") == address_group("2001:db8::2") == "2001:db8::/64"
    assert address_group("not an address") == "not an address"


def test_the_admin_cli_manages_accounts_and_keys(tmp_path, capsys) -> None:
    import sys

    from verathos.gateway.__main__ import main

    url = f"sqlite+aiosqlite:///{tmp_path}/admin.db"
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "database_url": url}))

    def cli(*args: str):
        sys.argv = ["verathos-gateway", "--config", str(config), *args]
        main()
        out = capsys.readouterr().out
        return json.loads(out) if args[0] != "db" else out

    cli("db", "upgrade")
    assert cli("accounts", "create", "--name", "acme", "--rate", "5", "--burst", "10")["id"] == 1
    key = cli("keys", "create", "--account", "acme", "--name", "prod")["key"]
    (listed,) = cli("keys", "list", "--account", "acme")
    assert listed["prefix"] == key[:16] and listed["revoked_at"] is None
    assert cli("keys", "revoke", "--account", "other", "--prefix", key[:16]) == {"revoked": 0}
    assert cli("keys", "revoke", "--account", "acme", "--prefix", key[:16]) == {"revoked": 1}
    assert cli("accounts", "set", "--name", "acme", "--status", "suspended") == {"updated": True}
    assert cli("accounts", "set", "--name", "acme", "--prepaid", "yes") == {"updated": True}
    added = cli("credits", "add", "--account", "acme", "--usd", "2.5", "--kind", "topup", "--reference", "pay-7")
    assert added["created"] and added["credit"]["amount"] == 2_500_000
    again = cli("credits", "add", "--account", "acme", "--usd", "2.5", "--kind", "topup", "--reference", "pay-7")
    assert not again["created"]
    listed = cli("credits", "list", "--account", "acme")
    assert listed["balance"] == 2_500_000 and [c["reference"] for c in listed["credits"]] == ["pay-7"]
    (account,) = cli("accounts", "list")
    assert account["prepaid"] is True and account["balance"] == 2_500_000


# ---- requests and responses that must not hurt anyone ----

class Counting(MockEngine):
    calls = 0

    async def answer(self, request):
        Counting.calls += 1
        if "POISON" in str(request.get("state")):
            raise EngineError("this request breaks the engine")
        return await super().answer(request)


def test_requests_no_miner_could_answer_are_refused_before_any_miner(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": Counting("kev-4b")})
    Counting.calls = 0
    nan = asyncio.run(s.post(content_override=b'{"model": "kev-4b", "state": {"x": NaN}, "questions": '
                                              b'{"q": {"type": "noul"}}}'))
    assert nan.status_code == 400 and nan.json()["error"]["type"] == "invalid_request"
    huge = {"model": "kev-4b", "state": "x", "questions": {
        f"q{i}": {"type": "choice", "criteria": {f"option {j:03d} with a long name": "" for j in range(255)}}
        for i in range(20)}}
    assert asyncio.run(s.post(huge)).json()["error"]["type"] == "too_large"
    assert Counting.calls == 0


def test_a_request_that_breaks_every_miner_opens_no_breaker(tmp_path) -> None:
    s = Setup(tmp_path, {f"10.0.0.{i}": Counting("kev-4b") for i in (1, 2, 3)})
    poison = {**REQUEST, "state": "POISON"}
    for _ in range(4):
        assert asyncio.run(s.post(poison)).json()["error"]["type"] == "unavailable"
    assert not any(e.breaker.is_open for e in s.router._pools["kev-4b"].endpoints)
    assert asyncio.run(s.post()).status_code == 200


def test_hostile_responses_are_refused_and_leave_nothing_behind(tmp_path) -> None:
    import gzip

    from starlette.responses import Response

    s = Setup(tmp_path, {"10.0.0.2": MockEngine("kev-4b")})
    hostile, deep = FastAPI(), "[" * 5000 + "]" * 5000

    @hostile.post("/e/gpu0/v1/systemone")
    async def answer():
        body = gzip.compress(b'{"answers": {}}' + b" " * 1_000_000)
        return Response(body, headers={"Content-Encoding": "gzip"}, media_type="application/json")

    nested = FastAPI()

    @nested.post("/e/gpu0/v1/systemone")
    async def nested_answer():
        return Response('{"answers": ' + deep + ', "receipt": {}}', media_type="application/json")

    for host, app in (("10.0.0.8", hostile), ("10.0.0.9", nested)):
        s.client._transport._t[host] = httpx.ASGITransport(app=app)
    s.router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=2, catalog_digest="0" * 64,
                           routes=[route(kp(), "10.0.0.8", 1000.0), route(kp(), "10.0.0.9", 1000.0)]))
    responses = s.many(6)
    assert {r.json()["error"]["type"] for r in responses} <= {"unavailable", "busy"}  # busy once both are shut out
    assert all(e.in_flight == 0 and e.breaker.is_open for e in s.router._pools["kev-4b"].endpoints)


def test_usage_is_what_the_miner_signed_within_bounds(tmp_path) -> None:
    class Greedy(MockEngine):
        async def answer(self, request):
            return {**await super().answer(request), "usage": {"input_tokens": 10 ** 20}}

    s = Setup(tmp_path, {"10.0.0.1": Greedy("kev-4b")})
    r = asyncio.run(s.post())
    tokens = r.json()["usage"]["input_tokens"]
    assert r.status_code == 200 and 0 < tokens <= len(json.dumps(REQUEST))  # at most one per byte of the request
    asyncio.run(s.meter.flush())  # writes, whatever the miner claimed
    (usage,) = asyncio.run(s.accounts.usage("acme", dt.date(2026, 1, 1)))
    assert usage["input_tokens"] == tokens


def test_invalid_keys_are_limited_before_the_database(tmp_path) -> None:
    import secrets

    from verathos.gateway.app import FAILED_KEY_BURST

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    lookups = []
    load = s.accounts._load

    async def counted(h):
        lookups.append(h)
        return await load(h)

    s.accounts._load = counted
    codes = [asyncio.run(s.post(key="vd_" + secrets.token_hex(32))).status_code for _ in range(60)]
    assert codes.count(401) == FAILED_KEY_BURST and codes.count(429) == 60 - FAILED_KEY_BURST
    assert len(lookups) == FAILED_KEY_BURST


def test_what_miners_answer_the_gateway_reaches_the_auditor(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    for _ in range(3):
        assert asyncio.run(s.post()).status_code == 200
    s.served.clock = lambda: 4_000_000_000.0  # those seconds are over
    import time as time_module
    shift = int(time_module.time()) - max(second for _, _, second in s.served._counts)
    s.served._counts = {(hk, ep, second + shift): n for (hk, ep, second), n in s.served._counts.items()}
    asyncio.run(s.served.report())
    keypair = s.miners["10.0.0.1"][0]
    now = time_module.time()
    assert s.auditor.served_in(keypair.ss58_address, "gpu0", now - 60, now + 1) == pytest.approx(3.0)


def test_a_miner_refusing_requests_another_miner_answers_is_counted(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.2": MockEngine("kev-4b")})
    refuser, bad = FastAPI(), kp()

    @refuser.post("/e/gpu0/v1/systemone")
    async def refuse():
        return JSONResponse({"error": "not today"}, status_code=400)

    s.client._transport._t["10.0.0.7"] = httpx.ASGITransport(app=refuser)
    s.router.update(Routes(netuid=405, auditor="a", epoch=3, issued_ms=2, catalog_digest="0" * 64, routes=[
        route(bad, "10.0.0.7", 1000.0), route(s.miners["10.0.0.2"][0], "10.0.0.2", 20.0)]))
    assert all(r.status_code == 200 for r in s.many(5))
    assert sum(n.refused for (hk, _, _), n in s.served._counts.items() if hk == bad.ss58_address) >= 1


def test_served_counts_are_kept_while_the_auditor_cannot_be_found() -> None:
    async def unknown():
        raise RuntimeError("chain unreachable")

    counter = ServedCounter(kp(), httpx.AsyncClient(), auditor_url=unknown, auditor_hotkey="a", clock=lambda: 1000.0)
    r = Route(hotkey="hk", endpoint_id="gpu0", model_id="kev-4b", url="http://x/e/gpu0", credited_rps=10.0,
              latency_ms=300.0)
    counter.record(r, 10, 3.0)
    assert ("hk", "gpu0", 997) in counter._counts  # in the second its request was sent
    asyncio.run(counter.report(final=True))
    assert counter.pending() == 1


def test_what_users_get_from_each_endpoint_reaches_the_auditor(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b"), "10.0.0.2": MockEngine("kev-4b")})
    s.router.max_latency_s = 1e-9  # every answer is slower than twice this bound
    for _ in range(2):
        assert asyncio.run(s.post()).status_code == 200  # answered, and reported as keeping the user waiting
    s.router.max_latency_s = 2.0
    for _ in range(4):
        assert asyncio.run(s.post()).status_code == 200
    import time as time_module
    shift = int(time_module.time()) - 1 - max(second for _, _, second in s.served._counts)  # all seconds complete
    s.served._counts = {(hk, ep, second + shift): n for (hk, ep, second), n in s.served._counts.items()}
    asyncio.run(s.served.report())
    tallies = s.auditor._users[s.auditor.follower.head // 180]
    assert sum(t.answers for t in tallies.values()) == 6 and sum(t.refused for t in tallies.values()) >= 1
    assert all(t.offered >= t.refused for t in tallies.values())


# ---- the admin API ----

ADMIN_TOKEN = "a" * 40


def admin_setup(tmp_path):
    from verathos.gateway.admin import ByPort, admin_app

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b"), "10.0.0.2": Slow("kev-4b", 0.05)})
    s.app = ByPort(s.app, 9999, admin_app(s.gateway, token=ADMIN_TOKEN))
    return s


async def admin_call(s, method: str, path: str, *, port: int = 9999, token: str | None = ADMIN_TOKEN, json_=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url=f"http://admin:{port}") as c:
        return await c.request(method, path, headers=headers, json=json_)


def test_the_admin_api_exists_only_on_its_port_and_only_with_the_token(tmp_path) -> None:
    s = admin_setup(tmp_path)
    for method, path in (("GET", "/admin/v1/state"), ("GET", "/admin/v1/systemone"), ("POST", "/admin/v1/systemone")):
        assert asyncio.run(admin_call(s, method, path, port=80)).status_code == 404  # nothing on the public port
    assert asyncio.run(admin_call(s, "POST", "/v1/systemone")).status_code == 404  # nor public routes here
    assert asyncio.run(admin_call(s, "GET", "/admin/v1/state", token=None)).status_code == 401
    assert asyncio.run(admin_call(s, "GET", "/admin/v1/state", token="b" * 40)).status_code == 401
    state = asyncio.run(admin_call(s, "GET", "/admin/v1/state")).json()
    assert state["models"] == ["kev-4b"] and state["routes"]["endpoints"] == 2 and "usage_rows" in state["pending"]
    routes = asyncio.run(admin_call(s, "GET", "/admin/v1/routes")).json()["routes"]
    assert {r["url"] for r in routes} == {"http://10.0.0.1:8091/e/gpu0", "http://10.0.0.2:8091/e/gpu0"}
    assert all(r["in_flight"] == 0 and not r["breaker"]["open"] for r in routes)


def test_an_admin_request_goes_where_it_is_sent_and_is_never_billed(tmp_path) -> None:
    s = admin_setup(tmp_path)
    target_kp = s.miners["10.0.0.2"][0]
    target = {"hotkey": target_kp.ss58_address, "endpoint_id": "gpu0"}
    r = asyncio.run(admin_call(s, "POST", "/admin/v1/systemone", json_={"request": REQUEST, "target": target}))
    body = r.json()
    assert r.status_code == 200 and body["endpoint"] == f"{target_kp.ss58_address}/gpu0" and not body["counted"]
    assert body["attempts"][0]["outcome"] == "answered" and json.loads(body["response"])["receipt"]
    assert s.served.pending() == 0 and s.sampler.pending() == {"picked": 0, "proofs": 0} and s.meter.pending() == 0
    counted = asyncio.run(admin_call(s, "POST", "/admin/v1/systemone",
                                     json_={"request": REQUEST, "target": target, "count": True})).json()
    assert counted["counted"] and s.served.pending() == 1 and s.sampler.pending()["picked"] == 1
    assert s.meter.pending() == 0  # never billed
    routed = asyncio.run(admin_call(s, "POST", "/admin/v1/systemone", json_={"request": REQUEST})).json()
    assert routed["attempts"] and routed["answers"] and s.served.pending() == 1 and s.meter.pending() == 0
    unknown = asyncio.run(admin_call(s, "POST", "/admin/v1/systemone",
                                     json_={"request": REQUEST, "target": {"hotkey": "x", "endpoint_id": "gpu0"}}))
    assert unknown.status_code == 502 and unknown.json()["attempts"][0]["detail"].startswith("not a routable")
    for bad in ({"request": REQUEST, "target": "x"}, {"request": REQUEST, "count": "false"}, {"request": []}):
        assert asyncio.run(admin_call(s, "POST", "/admin/v1/systemone", json_=bad)).status_code == 400
    from verathos.protocol.gateway import latency_bucket
    latencies = {b for (hk, _, _), n in s.served._counts.items() for b in n.latency if hk == target["hotkey"]}
    assert latencies == {latency_bucket(counted["attempts"][0]["ms"] / 1000)}  # the targeted answer's own duration
    assert all(e.in_flight == 0 for e in s.router._pools["kev-4b"].endpoints)


def test_the_admin_api_is_never_public_and_needs_a_private_token(tmp_path) -> None:
    import os

    from verathos.gateway.admin import read_token

    base = {"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "database_url": "sqlite+aiosqlite:///x.db"}
    for public in ("8.8.8.8", "0.0.0.0", "::"):
        with pytest.raises(ValueError):
            GatewayConfig(**base, admin_host=public, admin_port=9100, admin_token_file="t")
    with pytest.raises(ValueError):
        GatewayConfig(**base, admin_port=9100)
    GatewayConfig(**base, admin_host="10.1.2.3", admin_port=9100, admin_token_file="t")
    token = tmp_path / "admin.token"
    token.write_text(ADMIN_TOKEN)
    os.chmod(token, 0o644)
    with pytest.raises(SystemExit):
        read_token(token)
    os.chmod(token, 0o600)
    assert read_token(token) == ADMIN_TOKEN
    token.write_text("short")
    with pytest.raises(SystemExit):
        read_token(token)


def test_a_prepaid_account_is_served_while_its_credit_lasts(tmp_path) -> None:
    from verathos.gateway.accounts import Conflict

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    price = catalog.load().get("kev-4b").price
    long = {**REQUEST, "state": "the customer asks about an invoice " * 40}  # costs whole micro-USD

    def balance() -> int:
        return asyncio.run(s.accounts.account("acme"))["balance"]

    first = asyncio.run(s.post(long))  # not prepaid: served without credit, its usage still costs
    assert first.status_code == 200
    asyncio.run(s.meter.flush())
    cost = round(first.json()["usage"]["input_tokens"] * price)
    (usage,) = asyncio.run(s.accounts.usage("acme", dt.date(2026, 1, 1)))
    assert cost > 0 and usage["cost"] == cost and balance() == -cost

    async def models():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://gw") as c:
            return (await c.get("/v1/models")).json()["data"]
    listed = {m["id"]: m for m in asyncio.run(models())}
    assert listed["kev-4b"]["price"] == {"usd_per_million_input_tokens": price, "usd_per_million_output_tokens": 0}
    assert listed["kev-4b"]["available"] and listed["kev-4b"]["weights"]["repo"]
    assert not listed["kev-9b"]["available"]  # active in the catalog, listed, but no miner routed

    asyncio.run(s.accounts.set_account("acme", prepaid=True))
    refused = asyncio.run(s.post(long))
    assert refused.status_code == 402 and refused.json()["error"]["type"] == "insufficient_credit"
    entry, created = asyncio.run(s.accounts.add_credit("acme", cost + 1, kind="grant", reference="signup:acme"))
    assert created and entry["amount"] == cost + 1 and balance() == 1
    assert asyncio.run(s.post(long)).status_code == 200  # new credit applies at once in this process
    assert asyncio.run(s.post(long)).status_code == 402  # charged, though not written yet
    asyncio.run(s.meter.flush())
    assert balance() == 1 - cost  # the answer in flight is charged: slightly below zero

    again, created = asyncio.run(s.accounts.add_credit("acme", cost + 1, kind="grant", reference="signup:acme"))
    assert not created and again["id"] == entry["id"] and balance() == 1 - cost  # recorded once
    with pytest.raises(Conflict):
        asyncio.run(s.accounts.add_credit("acme", 5, kind="grant", reference="signup:acme"))
    for amount, kind in ((-5, "topup"), (0, "adjust"), (5, "gift")):
        with pytest.raises(ValueError):
            asyncio.run(s.accounts.add_credit("acme", amount, kind=kind, reference=f"x{amount}{kind}"))
    with pytest.raises(LookupError):
        asyncio.run(s.accounts.add_credit("nobody", 5, kind="grant", reference="y"))
    asyncio.run(s.accounts.add_credit("acme", 1_000_000, kind="topup", reference="pay-1"))
    assert asyncio.run(s.post(long)).status_code == 200
    ledger = asyncio.run(s.accounts.credits("acme"))
    assert [(c["kind"], c["reference"]) for c in ledger] == [("topup", "pay-1"), ("grant", "signup:acme")]


def test_another_process_charges_and_credit_apply_within_the_balance_ttl(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    clock = Clock()
    s.meter.clock = clock
    asyncio.run(s.accounts.set_account("acme", prepaid=True))
    asyncio.run(s.accounts.add_credit("acme", 1_000, kind="grant", reference="g"))
    assert asyncio.run(s.post()).status_code == 200
    other = Accounts(s.engine)  # another process: its credit and charges reach this one through the database
    asyncio.run(other.add_credit("acme", -2_000, kind="adjust", reference="a"))
    assert asyncio.run(s.post()).status_code == 200  # read less than BALANCE_TTL_S ago
    clock.now += s.meter.balance_ttl_s
    assert asyncio.run(s.post()).status_code == 402
    asyncio.run(other.add_credit("acme", 5_000, kind="topup", reference="t"))
    assert asyncio.run(s.post()).status_code == 402
    clock.now += s.meter.balance_ttl_s
    assert asyncio.run(s.post()).status_code == 200


def test_the_admin_api_signs_users_up_with_keys_and_credit(tmp_path) -> None:
    s = admin_setup(tmp_path)

    def call(method: str, path: str, body=None) -> httpx.Response:
        return asyncio.run(admin_call(s, method, path, json_=body))

    new = {"name": "u_42", "rate_rps": 5, "burst": 10, "prepaid": True}
    created = call("POST", "/admin/v1/accounts", new)
    assert created.status_code == 201 and created.json()["balance"] == 0 and created.json()["prepaid"] is True
    assert call("POST", "/admin/v1/accounts", new).status_code == 409
    for bad in ({**new, "name": "a/b"}, {**new, "burst": "10"}, {**new, "admin": True}, {**new, "prepaid": "yes"}):
        assert call("POST", "/admin/v1/accounts", bad).status_code == 400
    made = call("POST", "/admin/v1/accounts/u_42/keys", {"name": "default"})
    key = made.json()["key"]
    assert made.status_code == 201 and key.startswith(made.json()["prefix"]) and made.json()["revoked_at"] is None
    assert [k["prefix"] for k in call("GET", "/admin/v1/accounts/u_42/keys").json()["keys"]] == [key[:16]]
    assert "key" not in call("GET", "/admin/v1/accounts/u_42/keys").json()["keys"][0]
    assert asyncio.run(s.post(key=key)).status_code == 402

    grant = {"amount": 1_000_000, "kind": "grant", "reference": "signup:u_42", "note": "welcome"}
    first = call("POST", "/admin/v1/accounts/u_42/credits", grant)
    assert first.status_code == 201 and first.json()["balance"] == 1_000_000 and first.json()["created"]
    repeat = call("POST", "/admin/v1/accounts/u_42/credits", grant)
    assert repeat.status_code == 200 and not repeat.json()["created"] and repeat.json()["balance"] == 1_000_000
    assert call("POST", "/admin/v1/accounts/u_42/credits", {**grant, "amount": 2}).status_code == 409
    for bad in ({**grant, "amount": "5"}, {**grant, "amount": -5, "reference": "n"}, {**grant, "kind": "gift"},
                {"amount": 5, "kind": "grant"}):
        assert call("POST", "/admin/v1/accounts/u_42/credits", bad).status_code == 400
    answered = asyncio.run(s.post(key=key))
    assert answered.status_code == 200
    asyncio.run(s.meter.flush())
    cost = round(answered.json()["usage"]["input_tokens"] * catalog.load().get("kev-4b").price)
    (usage,) = call("GET", "/admin/v1/accounts/u_42/usage?since=2026-01-01").json()["usage"]
    assert (usage["requests"], usage["cost"]) == (1, cost)
    ledger = call("GET", "/admin/v1/accounts/u_42/credits").json()
    assert ledger["balance"] == 1_000_000 - cost and [c["reference"] for c in ledger["credits"]] == ["signup:u_42"]
    assert call("GET", "/admin/v1/accounts/u_42/usage?since=May").status_code == 400

    changed = call("PATCH", "/admin/v1/accounts/u_42", {"rate_rps": 9, "monthly_quota": None}).json()
    assert changed["rate_rps"] == 9 and changed["monthly_quota"] is None
    assert call("PATCH", "/admin/v1/accounts/u_42", {"status": None}).status_code == 400
    call("PATCH", "/admin/v1/accounts/u_42", {"status": "suspended"})
    assert asyncio.run(s.post(key=key)).status_code == 401
    call("PATCH", "/admin/v1/accounts/u_42", {"status": "active"})
    assert call("DELETE", f"/admin/v1/accounts/u_42/keys/{key[:16]}").status_code == 200
    assert call("DELETE", f"/admin/v1/accounts/u_42/keys/{key[:16]}").status_code == 404
    assert asyncio.run(s.post(key=key)).status_code == 401
    for method, path, body in (("GET", "/admin/v1/accounts/nope", None), ("PATCH", "/admin/v1/accounts/nope", {}),
                               ("POST", "/admin/v1/accounts/nope/keys", {"name": "k"}),
                               ("GET", "/admin/v1/accounts/nope/keys", None),
                               ("GET", "/admin/v1/accounts/nope/usage", None),
                               ("POST", "/admin/v1/accounts/nope/credits", grant),
                               ("GET", "/admin/v1/accounts/nope/credits", None)):
        assert call(method, path, body).status_code == 404, (method, path)
    assert call("GET", "/admin/v1/accounts/u_42", None).status_code == 200
    assert asyncio.run(admin_call(s, "GET", "/admin/v1/accounts/u_42", port=80)).status_code == 404


def test_a_database_from_before_credit_is_upgraded_in_place(tmp_path) -> None:
    from alembic import command
    from sqlalchemy import text

    url = f"sqlite+aiosqlite:///{tmp_path}/old.db"
    command.upgrade(db.alembic_config(url), "0001")

    async def old_rows():
        engine = db.create_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text("INSERT INTO accounts (name, rate_rps, burst) VALUES ('old', 1, 1)"))
        await engine.dispose()

    asyncio.run(old_rows())
    db.upgrade(url)
    engine = db.create_engine(url)
    assert asyncio.run(db.schema_revision(engine)) == db.head_revision()
    account = asyncio.run(Accounts(engine).account("old"))
    assert (account["prepaid"], account["balance"]) == (False, 0)
    asyncio.run(engine.dispose())
