"""The gateway's HTTP API.

    POST /v1/systemone   a SystemOne request with ``Authorization: Bearer <key>`` -> the answers, the usage, the
                         miner's signed receipt and the request id
    GET  /v1/models      the models served now
    GET  /health         the process is up
    GET  /ready          routes are loaded and the gateway is in the catalog (503 otherwise)

Every response carries ``X-Request-Id`` (the client's, if it sent a valid one). Errors are
``{"error": {"type": ..., "message": ...}}``; 429 and 503 carry ``Retry-After``.

    400 invalid_request     not JSON, or not a SystemOne request
    401 unauthorized        missing, unknown, revoked or expired key
    404 model_not_found     not a model the gateway serves
    413 too_large           body or state over the limits
    429 rate_limited        over the account's rate (or too many failed keys from one address)
    429 quota_exceeded      over the account's monthly quota
    503 busy                no capacity free for the model right now
    503 unavailable         the miners failed within the request's deadline, or the gateway cannot serve yet
"""

from __future__ import annotations

import ipaddress
import json
import logging
import math
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verathos import catalog as catalog_module
from verathos.gateway import metrics
from verathos.gateway.accounts import Accounts
from verathos.gateway.config import GatewayConfig
from verathos.gateway.limits import RateLimiter
from verathos.gateway.router import Outcome, Router
from verathos.gateway.sampler import Sampler
from verathos.gateway.upstream import Exchange, Upstream, UpstreamFailure
from verathos.gateway.usage import UsageMeter
from verathos.protocol.api import SystemOneRequest
from verathos.protocol.canonical import canonical_json

log = logging.getLogger(__name__)
STATE_CHARS_PER_TOKEN = 4  # as the miners count it
FAILED_KEY_RATE, FAILED_KEY_BURST = 2.0, 20  # per client address
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


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
    routes_loaded: Callable[[], bool]
    limiter: RateLimiter
    clock: Callable[[], float] = time.monotonic


def error(status: int, kind: str, message: str, request_id: str, retry_after: float | None = None) -> JSONResponse:
    headers = {"X-Request-Id": request_id}
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status, headers=headers)


def bearer(header: str | None) -> str | None:
    if not header or not header.startswith("Bearer "):
        return None
    return header[7:].strip() or None


def client_address(request: Request, trusted: list[ipaddress.IPv4Network | ipaddress.IPv6Network]) -> str:
    """The client's address: the peer, or, behind trusted proxies, the last X-Forwarded-For hop not among them."""
    peer = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    if not any(address in network for network in trusted):
        return peer
    for hop in reversed([h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]):
        try:
            hop_address = ipaddress.ip_address(hop)
        except ValueError:
            return hop
        if not any(hop_address in network for network in trusted):
            return hop
    return peer


def create_app(g: Gateway) -> FastAPI:
    app = FastAPI(title="Verathos gateway", docs_url=None, redoc_url=None, openapi_url=None)
    trusted = [ipaddress.ip_network(n, strict=False) for n in g.config.trusted_proxies]
    in_flight = 0

    def ready_reason() -> str | None:
        cat = g.catalog()
        if not g.routes_loaded() or cat is None:
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
        served = g.router.models()
        data = [] if cat is None else [{"id": m.model_id, "object": "model", "max_state_tokens": m.max_state_tokens}
                                       for m in cat.models if m.status == "active" and m.model_id in served]
        return {"object": "list", "data": data}

    @app.post("/v1/systemone")
    async def systemone(request: Request):
        nonlocal in_flight
        given = request.headers.get("x-request-id", "")
        request_id = given if _REQUEST_ID.fullmatch(given) else secrets.token_hex(12)
        started = g.clock()
        if in_flight >= g.config.max_in_flight:
            return error(503, "busy", "the gateway is at capacity", request_id, 1)
        key = bearer(request.headers.get("authorization"))
        try:
            principal = await g.accounts.authenticate(key) if key else None
        except Exception:
            log.exception("key lookup failed")
            return error(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if principal is None:
            wait = g.limiter.check(("address", client_address(request, trusted)), FAILED_KEY_RATE,
                                   FAILED_KEY_BURST, started)
            if wait:
                return error(429, "rate_limited", "too many requests with invalid keys", request_id, wait)
            return error(401, "unauthorized", "a valid API key is required (Authorization: Bearer ...)",
                         request_id)
        wait = g.limiter.check(principal.account_id, principal.rate_rps / g.config.instances, principal.burst,
                               started)
        if wait:
            return error(429, "rate_limited", "over the account's request rate", request_id, wait)
        if not g.meter.within_quota(principal):
            return error(429, "quota_exceeded", "the account's monthly quota is used up", request_id)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > g.config.max_body_bytes:
                return error(413, "too_large", f"the body exceeds {g.config.max_body_bytes} bytes", request_id)
        try:
            raw = json.loads(body)
            parsed = SystemOneRequest.model_validate(raw)
        except (ValueError, ValidationError) as exc:
            return error(400, "invalid_request", f"not a SystemOne request: {exc}"[:2000], request_id)
        cat = g.catalog()
        model = cat.get(parsed.model) if cat is not None else None
        if model is None or model.status != "active" or parsed.model not in g.router.models():
            return error(404, "model_not_found", f"{parsed.model} is not served", request_id)
        state = parsed.state if isinstance(parsed.state, str) else canonical_json(parsed.state).decode()
        if len(state) > STATE_CHARS_PER_TOKEN * model.max_state_tokens:
            return error(413, "too_large", f"the state exceeds what {model.model_id} accepts", request_id)

        in_flight += 1
        metrics.IN_FLIGHT.set(in_flight)
        try:
            outcome = await forward(parsed.model, raw, bytes(body), parsed, started)
        finally:
            in_flight -= 1
            metrics.IN_FLIGHT.set(in_flight)
        metrics.REQUEST_SECONDS.labels(parsed.model).observe(g.clock() - started)
        if isinstance(outcome, Exchange):
            usage = outcome.usage or {}
            g.meter.record(principal, parsed.model, answered=True, input_tokens=int(usage.get("input_tokens") or 0))
            g.sampler.offer(outcome)
            metrics.REQUESTS.labels(parsed.model, "answered").inc()
            return JSONResponse({"id": request_id, "model": parsed.model, "answers": outcome.answers,
                                 "usage": outcome.usage, "receipt": outcome.receipt},
                                headers={"X-Request-Id": request_id})
        g.meter.record(principal, parsed.model, answered=False)
        metrics.REQUESTS.labels(parsed.model, outcome).inc()
        if outcome == "busy":
            return error(503, "busy", f"no capacity free for {parsed.model} right now", request_id, 1)
        return error(503, "unavailable", f"no miner answered {parsed.model} in time", request_id, 1)

    async def forward(model_id: str, raw: dict[str, Any], body: bytes, parsed: SystemOneRequest,
                      started: float) -> Exchange | str:
        """Up to ``max_attempts`` attempts, each at another miner, within the request's deadline. -> the exchange,
        or "busy" (no endpoint could take it) or "failed"."""
        deadline = started + g.config.request_timeout_s
        tried: set[str] = set()
        result = "busy"
        for attempt in range(g.config.max_attempts):
            remaining = deadline - g.clock()
            if remaining <= 0.05:
                break
            endpoint = g.router.pick(model_id, tried)
            if endpoint is None:
                break
            if attempt:
                metrics.RETRIES.inc()
            g.router.started(endpoint)
            attempt_started = g.clock()
            try:
                exchange = await g.upstream.call(endpoint.route, raw, body, parsed,
                                                 min(g.config.attempt_timeout_s, remaining))
            except UpstreamFailure as failure:
                g.router.finished(endpoint, Outcome.BUSY if failure.busy else Outcome.FAILED)
                metrics.UPSTREAM_FAILURES.labels(failure.kind).inc()
                if failure.exchange is not None:
                    g.sampler.proof(failure.exchange)
                tried.add(endpoint.route.hotkey)
                result = "busy" if failure.busy and result == "busy" else "failed"
                continue
            g.router.finished(endpoint, Outcome.ANSWERED)
            metrics.UPSTREAM_SECONDS.labels(model_id).observe(g.clock() - attempt_started)
            return exchange
        return result

    return app
