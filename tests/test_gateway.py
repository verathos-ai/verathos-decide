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
from fastapi import FastAPI
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
from verathos.gateway.router import BREAKER_FAILURES, OPEN_S, Outcome, Router
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
from verathos.throughput.auditor import Checks


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
        from verathos.throughput.auditor import Slot
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
    assert asyncio.run(t.post({"model": "kev-4b"})).json()["error"]["type"] == "invalid_request"
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
    assert r.status_code == 200 and r.json()["usage"] == {"input_tokens": 10_000_000}
    asyncio.run(s.meter.flush())  # writes, whatever the miner claimed
    (usage,) = asyncio.run(s.accounts.usage("acme", dt.date(2026, 1, 1)))
    assert usage["input_tokens"] == 10_000_000


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
    s.served._counts = {(hk, ep, int(time_module.time())): n for (hk, ep, _), n in s.served._counts.items()}
    asyncio.run(s.served.report())
    keypair = s.miners["10.0.0.1"][0]
    now = time_module.time()
    assert s.auditor.served_in(keypair.ss58_address, "gpu0", now - 60, now + 1) == pytest.approx(3.0)


# ---- the admin API ----

ADMIN_TOKEN = "a" * 40


def admin_setup(tmp_path):
    from verathos.gateway.admin import add_admin_routes

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b"), "10.0.0.2": MockEngine("kev-4b")})
    add_admin_routes(s.app, s.gateway, port=9999, token=ADMIN_TOKEN)
    return s


async def admin_call(s, method: str, path: str, *, port: int = 9999, token: str | None = ADMIN_TOKEN, json_=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url=f"http://admin:{port}") as c:
        return await c.request(method, path, headers=headers, json=json_)


def test_the_admin_api_exists_only_on_its_port_and_only_with_the_token(tmp_path) -> None:
    s = admin_setup(tmp_path)
    assert asyncio.run(admin_call(s, "GET", "/admin/v1/state", port=80)).status_code == 404  # the public port
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
