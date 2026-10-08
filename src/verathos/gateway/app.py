"""The gateway's HTTP API.

    POST /v1/systemone   a SystemOne request with ``Authorization: Bearer <key>`` -> the answers, the usage, the
                         miner's signed receipt and the request id
    GET  /v1/models      the catalog's active models: state limit, price, weights, and whether available now
    GET  /v1/status      per model: operational, degraded or unavailable; endpoints, capacity, latency, success
    GET  /health         the process is up
    GET  /ready          routes are loaded and the gateway is in the catalog (503 otherwise)

Every response carries ``X-Request-Id`` (the client's, if it sent a valid one). Errors are
``{"error": {"type": ..., "message": ...}}``; 429 and 503 carry ``Retry-After``.

    400 invalid_request     not JSON (NaN and infinities included), or not a SystemOne request
    401 unauthorized        missing, unknown, revoked or expired key, or a suspended account
    402 insufficient_credit a prepaid account's balance is used up
    404 model_not_found     not an active model of the catalog
    413 too_large           body, state, or the answers the questions would need, over the limits
    429 rate_limited        over the key's rate (or too many invalid keys from one address)
    429 quota_exceeded      over the account's monthly quota
    500 internal            a fault of the gateway
    503 busy                no capacity free for the model right now
    503 unavailable         no miner serves the model right now, the miners failed or refused the request within its
                            deadline, every miner of the model is failing, or accounts cannot be checked
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import ipaddress
import json
import logging
import math
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verathos import catalog as catalog_module
from verathos.gateway import metrics
from verathos.gateway.accounts import Accounts, Principal, well_formed_key
from verathos.gateway.config import GatewayConfig
from verathos.gateway.limits import RateLimiter
from verathos.gateway.logs import Throttle
from verathos.gateway.router import Endpoint, Outcome, Router
from verathos.gateway.sampler import Sampler
from verathos.gateway.served import ServedCounter
from verathos.gateway.stats import DEGRADED_SUCCESS, ModelStats
from verathos.gateway.upstream import Exchange, Upstream, UpstreamFailure
from verathos.gateway.usage import UsageMeter
from verathos.protocol.api import MAX_RESPONSE_BYTES, Choice, SystemOneRequest, question_keys
from verathos.protocol.canonical import canonical_json, digest
from verathos.protocol.gateway import Routes

log = logging.getLogger(__name__)
STATE_CHARS_PER_TOKEN = 4  # as the miners count it
# An answer is billed at least one token per this many characters of the state, whatever the miner signs; these
# models' tokens average far fewer characters, so an honest count is never raised.
STATE_CHARS_PER_BILLED_TOKEN = 8
FAILED_KEY_RATE, FAILED_KEY_BURST = 2.0, 20  # invalid keys per client address (IPv6: per /64)
ANSWER_ENVELOPE_BYTES = 4096  # the receipt, the usage and the response's own fields
ACCOUNTS_WAIT_S = 5.0  # a request's key or credit check, at most; the miners' deadline starts after them
FINAL_WRITE_S = 10.0  # each write at shutdown, at most
STATUS_TTL_S = 15.0  # the models' states and the status are computed at most this often (it is public)
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_throttled = Throttle()


@dataclass
class Gateway:
    """The parts one gateway process serves with."""

    config: GatewayConfig
    hotkey: str
    catalog: Callable[[], catalog_module.Catalog | None]
    accounts: Accounts
    meter: UsageMeter
    router: Router
    upstream: Upstream
    sampler: Sampler
    served: ServedCounter
    routes: Callable[[], Routes | None]
    limiter: RateLimiter
    clock: Callable[[], float] = time.monotonic
    stats: ModelStats = field(default_factory=ModelStats)

    def routes_state(self) -> dict[str, Any] | None:
        table = self.routes()
        return None if table is None else {"epoch": table.epoch, "issued_ms": table.issued_ms,
                                           "age_s": round(time.time() - table.issued_ms / 1000, 1),
                                           "endpoints": len(table.routes), "catalog": table.catalog_digest}

    def pending_state(self) -> dict[str, Any]:
        return {"usage_rows": self.meter.pending(), "samples": self.sampler.pending(),
                "served_seconds": self.served.pending()}


def bearer(header: str | None) -> str | None:
    if not header or not header.startswith("Bearer "):
        return None
    return header[7:].strip() or None


def ip_address(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``text`` as an address, an IPv4 address mapped into IPv6 (``::ffff:a.b.c.d``) as the IPv4 one; None if it is
    not one."""
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None
    return getattr(ip, "ipv4_mapped", None) or ip


def client_address(request: Request, trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> str:
    """The client's address (see ``ip_address``): the peer, or, behind trusted proxies, the last X-Forwarded-For hop
    not among them."""
    peer = request.client.host if request.client else ""
    address = ip_address(peer)
    if address is None:
        return peer
    if not any(address in network for network in trusted):
        return str(address)
    for hop in reversed([h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]):
        hop_address = ip_address(hop)
        if hop_address is None:
            return hop
        if not any(hop_address in network for network in trusted):
            return str(hop_address)
    return str(address)


def address_group(address: str) -> str:
    """The unit invalid keys are limited by: an IPv4 address (also one mapped into IPv6), or an IPv6 /64."""
    ip = ip_address(address)
    if ip is None:
        return address
    return str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else str(ip)


def answer_bytes(request: SystemOneRequest) -> int:
    """An upper estimate of a response's size for this request: one probability per option of every question."""
    size = ANSWER_ENVELOPE_BYTES
    for qid, q in request.questions.items():
        keys = [json.dumps(k) for k in question_keys(q)]
        size += len(json.dumps(qid)) + 128 + sum(len(k) + 24 for k in keys)
        if isinstance(q, Choice):
            size += max(len(k) for k in keys)  # the chosen option
    return size


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


class Rejected(Exception):
    """A request the gateway refuses, with the API's error type."""

    def __init__(self, status: int, kind: str, message: str, retry_after: float | None = None,
                 model: str = "-") -> None:
        super().__init__(message)
        self.status, self.kind, self.message, self.retry_after, self.model = status, kind, message, retry_after, model


class Unchecked(Exception):
    """The accounts could not be checked: the database failed or was too slow."""


@dataclass(frozen=True)
class Checked:
    raw: dict[str, Any]
    body: bytes
    parsed: SystemOneRequest
    request_sha256: str
    price: float  # the model's, USD per million input tokens
    min_tokens: int  # billed at least, whatever the miner signs


NO_ANSWER = {  # why a forwarded request got no answer -> the API error
    "rejected": (503, "unavailable", "the miners of {model} refused this request", 1),
    "busy": (503, "busy", "no capacity free for {model} right now", 1),
    "failed": (503, "unavailable", "no miner answered {model} in time", 1),
    "down": (503, "unavailable", "no miner of {model} answers right now", 30),
}


def validation_message(exc: ValidationError) -> str:
    """The first problem of a validation error, short: where and what (no input echo, no documentation link)."""
    first = exc.errors(include_url=False, include_input=False)[0]
    where = ".".join(str(part) for part in first["loc"])
    more = f" (and {exc.error_count() - 1} more)" if exc.error_count() > 1 else ""
    return f"{where + ': ' if where else ''}{first['msg']}{more}"[:300]


def check_request(g: Gateway, body: bytes) -> Checked:
    """A body the gateway can forward: strict JSON, a SystemOne request for an active model miners serve now, within
    the model's state limit and with answers that fit a response. Raises Rejected."""
    try:
        raw = json.loads(body, parse_constant=_no_constants)
        parsed = SystemOneRequest.model_validate(raw)
        request_sha256 = digest(raw)
        state = parsed.state if isinstance(parsed.state, str) else canonical_json(parsed.state).decode()
    except ValidationError as exc:
        raise Rejected(400, "invalid_request", f"not a SystemOne request: {validation_message(exc)}") from None
    except (ValueError, TypeError, RecursionError) as exc:
        raise Rejected(400, "invalid_request", f"not a SystemOne request: {exc}"[:500]) from None
    cat = g.catalog()
    model = cat.get(parsed.model) if cat is not None else None
    if model is None or model.status != "active":
        raise Rejected(404, "model_not_found", f"{parsed.model} is not served")
    if parsed.model not in g.router.models():
        raise Rejected(503, "unavailable", f"no miner serves {parsed.model} right now", 30, parsed.model)
    if len(state) > STATE_CHARS_PER_TOKEN * model.max_state_tokens:
        raise Rejected(413, "too_large", f"the state exceeds what {model.model_id} accepts", model=parsed.model)
    if answer_bytes(parsed) > MAX_RESPONSE_BYTES:
        raise Rejected(413, "too_large", "the answers to these questions would exceed a response's size limit",
                       model=parsed.model)
    return Checked(raw, body, parsed, request_sha256, model.price, len(state) // STATE_CHARS_PER_BILLED_TOKEN)


async def forward(g: Gateway, checked: Checked, started: float, attempts: list[dict[str, Any]] | None = None, *,
                  counted: bool = True) -> Exchange | str:
    """Up to ``max_attempts`` attempts, each at another miner, within the request's deadline. -> the exchange (with
    its attempt's duration), or why there is none: busy (no endpoint could take it), rejected (the miners refused the
    request, 4xx) or failed. A failure whose cause is unclear counts against a miner only once another miner has
    answered. A refusal is unclear too, but the gateway has checked everything a miner checks, so one miner refusing a
    request no other miner refused or answered is reported as refusing users (its breaker is not touched).
    ``attempts`` collects what each attempt did; ``counted``: the request is user traffic, so the miners' refusals of
    it are reported."""
    model_id, deadline = checked.parsed.model, started + g.config.request_timeout_s
    tried: set[str] = set()
    unclear: list[Endpoint] = []
    rejected: list[Endpoint] = []
    result = "busy"
    for attempt in range(g.config.max_attempts):
        remaining = deadline - g.clock()
        if remaining <= 0.05:
            break
        endpoint = g.router.pick(model_id, tried, observe=counted)
        if endpoint is None:
            if not attempt and g.router.down(model_id):
                result = "down"
            break
        if attempt:
            metrics.RETRIES.inc()
        tried.add(endpoint.route.hotkey)
        g.router.started(endpoint, observe=counted)
        outcome: Outcome | None = None
        attempt_started = g.clock()
        note = {"endpoint": endpoint.key}
        try:
            exchange = await g.upstream.call(endpoint.route, checked.raw, checked.body, checked.request_sha256,
                                             checked.parsed, min(g.config.attempt_timeout_s, remaining),
                                             min_tokens=checked.min_tokens)
            outcome = Outcome.ANSWERED
            note["outcome"] = "answered"
        except UpstreamFailure as failure:
            metrics.UPSTREAM_FAILURES.labels(failure.kind).inc()
            note.update(outcome="failed", kind=failure.kind, blame=failure.blame, detail=str(failure))
            if failure.exchange is not None:
                g.sampler.proof(failure.exchange)
            if failure.blame == "miner":
                outcome, result = Outcome.FAILED, "failed"
            elif failure.blame == "busy":
                outcome = Outcome.BUSY
            elif failure.blame == "gateway":  # this host's own fault: no miner is told off for it
                result = "failed"
            elif failure.blame == "rejected":
                unclear.append(endpoint)
                rejected.append(endpoint)
                result = "rejected" if result != "failed" else result
            else:
                unclear.append(endpoint)
                result = "failed"
            continue
        except Exception as exc:
            log.exception("an attempt at %s failed", endpoint.key)
            metrics.UPSTREAM_FAILURES.labels("unexpected").inc()
            note.update(outcome="failed", kind="unexpected", detail=type(exc).__name__)
            outcome, result = Outcome.FAILED, "failed"
            continue
        finally:
            seconds = g.clock() - attempt_started
            g.router.finished(endpoint, outcome, seconds, observe=counted)
            note["ms"] = round(1000 * seconds, 1)
            if attempts is not None:
                attempts.append(note)
        for other in unclear:
            g.router.penalize(other, observe=counted)
        metrics.UPSTREAM_SECONDS.labels(model_id).observe(seconds)
        return dataclasses.replace(exchange, seconds=seconds)
    if len(rejected) == 1:
        g.router.refused(rejected[0], observe=counted)
    return result


async def final_writes(g: Gateway) -> None:
    """At shutdown: usage, samples and served counts written, each within ``FINAL_WRITE_S``; what cannot be is logged
    as lost."""
    for what, write in (("usage", g.meter.flush), ("samples", g.sampler.flush),
                        ("served counts", lambda: g.served.report(final=True))):
        try:
            async with asyncio.timeout(FINAL_WRITE_S):
                await write()
        except Exception as exc:
            log.error("the final write of %s failed (%s: %s); not written: %s", what, type(exc).__name__, exc,
                      g.pending_state())


def model_state(g: Gateway, model_id: str) -> str:
    """unavailable: no routable endpoint, or every one failing; degraded: under DEGRADED_SUCCESS of the last 15
    minutes' requests answered, or more than half the endpoints failing; else operational."""
    health = g.router.health(model_id)
    if not health["endpoints"] or g.router.down(model_id):
        return "unavailable"
    recent = g.stats.success_rate(model_id, 15)
    if (recent is not None and recent < DEGRADED_SUCCESS) or 2 * health["failing"] > health["endpoints"]:
        return "degraded"
    return "operational"


def mark_states(g: Gateway) -> dict[str, str]:
    """The active models' states now, recorded for their availability."""
    cat = g.catalog()
    states: dict[str, str] = {}
    for m in cat.models if cat is not None else []:
        if m.status == "active":
            states[m.model_id] = state = model_state(g, m.model_id)
            g.stats.mark(m.model_id, unavailable=state == "unavailable")
    return states


async def mark_states_forever(g: Gateway, every_s: float = 60.0) -> None:
    """Each minute's state, also in minutes nobody asks for the status."""
    while True:
        mark_states(g)
        await asyncio.sleep(every_s)


def create_app(g: Gateway, *, lifespan: Callable | None = None) -> FastAPI:
    app = FastAPI(title="Verathos gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    trusted = [ipaddress.ip_network(n, strict=False) for n in g.config.trusted_proxies]
    in_flight = 0
    computed: dict[str, tuple[float, Any]] = {}

    def recent(name: str, compute: Callable[[], Any]) -> Any:
        """``compute()``, computed again only after ``STATUS_TTL_S``."""
        now = g.clock()
        if name not in computed or now - computed[name][0] >= STATUS_TTL_S:
            computed[name] = (now, compute())
        return computed[name][1]

    def reject(status: int, kind: str, message: str, request_id: str, retry_after: float | None = None,
               model: str = "-") -> JSONResponse:
        metrics.REQUESTS.labels(model, kind).inc()
        headers = {"X-Request-Id": request_id}
        if retry_after is not None:
            headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
        return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status, headers=headers)

    @app.exception_handler(Exception)
    async def internal(request: Request, exc: Exception) -> JSONResponse:
        log.error("request failed", exc_info=exc)
        return reject(500, "internal", "the gateway failed", getattr(request.state, "request_id", ""))

    def ready_reason() -> str | None:
        cat = g.catalog()
        if g.routes() is None or cat is None:
            return "no routes from the auditor yet"
        if g.hotkey not in cat.gateways:
            return "this gateway's hotkey is not among the catalog's gateways"
        return None

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        reason = ready_reason()
        if reason:
            return JSONResponse({"ready": False, "reason": reason}, status_code=503)
        return {"ready": True}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        cat = g.catalog()
        states = recent("states", lambda: mark_states(g))
        data = [] if cat is None else [{
            "id": m.model_id, "object": "model", "max_state_tokens": m.max_state_tokens,
            "price": {"usd_per_million_input_tokens": m.price, "usd_per_million_output_tokens": 0},
            "available": states.get(m.model_id, "unavailable") != "unavailable", "license": m.license,
            "weights": {"repo": m.hf_repo, "revision": m.hf_revision},
            "base_model": {"repo": m.base_repo, "revision": m.base_revision} if m.base_repo else None,
        } for m in cat.models if m.status == "active"]
        return {"object": "list", "data": data}

    def status_document() -> dict[str, Any]:
        models = []
        for model_id, state in mark_states(g).items():
            health = g.router.health(model_id)
            models.append({"id": model_id, "state": state, "endpoints": health["endpoints"],
                           "capacity_rps": health["capacity_rps"], **g.stats.summary(model_id)})
        utc = dt.UTC
        return {"updated_at": dt.datetime.now(utc).isoformat(timespec="seconds"),
                "since": dt.datetime.fromtimestamp(g.stats.since, utc).isoformat(timespec="seconds"), "models": models}

    @app.get("/v1/status")
    async def status():
        return JSONResponse(recent("status", status_document),
                            headers={"Cache-Control": f"public, max-age={STATUS_TTL_S:.0f}"})

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        nonlocal in_flight
        given = request.headers.get("x-request-id", "")
        request_id = given if _REQUEST_ID.fullmatch(given) else secrets.token_hex(12)
        request.state.request_id = request_id
        if in_flight >= g.config.max_in_flight:
            return reject(503, "busy", "the gateway is at capacity", request_id, 1)
        in_flight += 1
        metrics.IN_FLIGHT.set(in_flight)
        try:
            return await serve(request, request_id)
        finally:
            in_flight -= 1
            metrics.IN_FLIGHT.set(in_flight)

    async def serve(request: Request, request_id: str):
        started = g.clock()
        key = bearer(request.headers.get("authorization"))
        address = ("address", address_group(client_address(request, trusted)))
        if g.limiter.wait(address, FAILED_KEY_RATE, FAILED_KEY_BURST, started):
            return reject(429, "rate_limited", "too many requests with invalid keys", request_id, 1)
        try:
            principal = await bounded("key lookup", g.accounts.authenticate(key)) if well_formed_key(key) else None
        except Unchecked:
            return reject(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if principal is None:
            g.limiter.check(address, FAILED_KEY_RATE, FAILED_KEY_BURST, started)
            return reject(401, "unauthorized", "a valid API key is required (Authorization: Bearer ...)", request_id)
        wait = g.limiter.check(("key", principal.key_id), principal.rate_rps, principal.burst, started)
        if wait:
            return reject(429, "rate_limited", "over the key's request rate", request_id, wait)
        if not g.meter.within_quota(principal):
            return reject(429, "quota_exceeded", "the account's monthly quota is used up", request_id)
        try:
            funded = await bounded("balance lookup", g.meter.has_credit(principal))
        except Unchecked:
            return reject(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if not funded:
            return reject(402, "insufficient_credit", "the account's credit is used up", request_id)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > g.config.max_body_bytes:
                return reject(413, "too_large", f"the body exceeds {g.config.max_body_bytes} bytes", request_id)
        try:
            checked = check_request(g, bytes(body))
        except Rejected as r:
            return reject(r.status, r.kind, r.message, request_id, r.retry_after, model=r.model)
        most = len(checked.body) * checked.price  # the most the answer may cost: one token per byte of the request
        try:
            funded = await bounded("balance lookup", g.meter.reserve(principal, most))
        except Unchecked:
            return reject(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if not funded:
            return reject(402, "insufficient_credit", "the account's credit is used up", request_id)
        try:
            return await answer(principal, checked, started, request_id)
        finally:
            g.meter.release(principal, most)

    async def bounded(what: str, check: Awaitable[Any]) -> Any:
        """An account check within ``ACCOUNTS_WAIT_S``; raises Unchecked if the database fails or is slower."""
        try:
            async with asyncio.timeout(ACCOUNTS_WAIT_S):
                return await check
        except TimeoutError:
            if times := _throttled(f"{what} slow"):
                log.warning("%s took more than %.0f s (%d times)", what, ACCOUNTS_WAIT_S, times)
        except Exception:
            if times := _throttled(f"{what} failed"):
                log.exception("%s failed (%d times)", what, times)
        raise Unchecked

    async def answer(principal: Principal, checked: Checked, started: float, request_id: str):
        model = checked.parsed.model
        outcome = await forward(g, checked, g.clock())  # the miners' deadline starts now, after the gateway's checks
        metrics.REQUEST_SECONDS.labels(model).observe(g.clock() - started)
        if isinstance(outcome, Exchange):
            # rendered before anything is recorded: an answer that cannot be delivered is neither billed nor counted
            response = JSONResponse({"id": request_id, "model": model, "answers": outcome.answers,
                                     "usage": {"input_tokens": outcome.input_tokens}, "receipt": outcome.receipt},
                                    headers={"X-Request-Id": request_id})
            g.stats.record(model, answered=True, seconds=g.clock() - started)
            g.meter.record(principal, model, answered=True, input_tokens=outcome.input_tokens, price=checked.price)
            g.sampler.offer(outcome)
            g.served.record(outcome.route, outcome.input_tokens, outcome.seconds)
            metrics.REQUESTS.labels(model, "answered").inc()
            return response
        g.stats.record(model, answered=False)
        g.meter.record(principal, model, answered=False)
        status, kind, message, retry = NO_ANSWER[outcome]
        return reject(status, kind, message.format(model=model), request_id, retry, model=model)

    return app
