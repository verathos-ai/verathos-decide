"""The gateway's admin API: inspection and debugging, for the operators only.

An app of its own (``admin_app``), served on a socket of its own (``admin_host:admin_port``: loopback or a private
network; ``ByPort`` hands it the requests that arrive there) and only with the admin token (``admin_token_file``). On
the public port none of its routes exist. Every call is logged.

    GET  /admin/v1/state       routes, catalog, and what is waiting to be written (usage) or sent (samples, counts)
    GET  /admin/v1/routes      every routable endpoint as the router sees it: capacity, cap, in flight, breaker
    POST /admin/v1/systemone   {"request": <SystemOne request>, "target": {"hotkey", "endpoint_id"} | null,
                                "count": false}
                               routed as users' requests are, or sent to the one ``target`` endpoint (one attempt,
                               no effect on its breaker); never billed; ``count``: whether it counts as user traffic
                               (demand, test windows, sampling). Returns the answers, the receipt, the miner's
                               response as received and every attempt.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from verathos.gateway.app import Checked, Gateway, Rejected, check_request, forward
from verathos.gateway.upstream import Exchange, UpstreamFailure

log = logging.getLogger("verathos.gateway.admin")
MIN_TOKEN_CHARS = 32


def read_token(path: Path) -> str:
    """The admin token from its file; refuses a short token, and a file others can read."""
    path = path.expanduser()
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others: chmod 600 it")
    token = path.read_text().strip()
    if len(token) < MIN_TOKEN_CHARS:
        raise SystemExit(f"the admin token in {path} is shorter than {MIN_TOKEN_CHARS} characters")
    return token


class Target(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hotkey: str
    endpoint_id: str


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: dict[str, Any]
    target: Target | None = None
    count: StrictBool = False


class ByPort:
    """One server, two sockets: requests that arrive on ``port`` go to ``app``, all others (and the lifespan) to
    ``default``."""

    def __init__(self, default: Any, port: int, app: Any) -> None:
        self.default, self.port, self.app = default, port, app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        server = scope.get("server") or (None, None)
        if scope["type"] != "lifespan" and server[1] == self.port:
            await self.app(scope, receive, send)
        else:
            await self.default(scope, receive, send)


def admin_app(g: Gateway, *, token: str) -> FastAPI:
    app = FastAPI(title="Verathos gateway admin", docs_url=None, redoc_url=None, openapi_url=None)
    expected = f"Bearer {token}".encode()

    @app.middleware("http")
    async def authorized(request: Request, call_next):
        client = request.client.host if request.client else "?"
        if not hmac.compare_digest(request.headers.get("authorization", "").encode(), expected):
            log.warning("admin call %s %s with a wrong token from %s", request.method, request.url.path, client)
            return JSONResponse({"error": {"type": "unauthorized", "message": "admin token required"}},
                                status_code=401)
        response = await call_next(request)
        log.info("admin call %s %s from %s -> %d", request.method, request.url.path, client, response.status_code)
        return response

    @app.get("/admin/v1/state")
    async def state():
        cat = g.catalog()
        return {"hotkey": g.hotkey, "catalog": cat.digest() if cat else None, "routes": g.routes_state(),
                "models": sorted(g.router.models()), "pending": g.pending_state(),
                "limits": {"instances": g.config.instances, "max_in_flight": g.config.max_in_flight,
                           "request_timeout_s": g.config.request_timeout_s,
                           "attempt_timeout_s": g.config.attempt_timeout_s, "max_attempts": g.config.max_attempts}}

    @app.get("/admin/v1/routes")
    async def routes():
        return {"routes": g.router.describe()}

    @app.post("/admin/v1/systemone")
    async def debug_request(request: Request):
        try:
            command = Command.model_validate_json(await request.body())
        except ValidationError as exc:
            return JSONResponse({"error": {"type": "invalid_request", "message": str(exc).splitlines()[0]}},
                                status_code=400)
        try:
            checked = check_request(g, json.dumps(command.request).encode())
        except Rejected as r:
            return JSONResponse({"error": {"type": r.kind, "message": r.message}}, status_code=r.status)
        started, attempts, target = g.clock(), [], command.target
        if target is not None:
            outcome = await _to_target(g, checked, target, attempts)
        else:
            outcome = await forward(g, checked, started, attempts, counted=command.count)
        log.info("admin request %s target=%s count=%s -> %s", checked.parsed.model,
                 f"{target.hotkey}/{target.endpoint_id}" if target else "router", command.count,
                 "answered" if isinstance(outcome, Exchange) else outcome)
        reply: dict[str, Any] = {"model": checked.parsed.model, "attempts": attempts, "counted": False,
                                 "ms": round(1000 * (g.clock() - started), 1)}
        if not isinstance(outcome, Exchange):
            return JSONResponse({**reply, "error": outcome}, status_code=502)
        if command.count:
            g.sampler.offer(outcome)
            g.served.record(outcome.route, outcome.input_tokens, outcome.seconds)
            reply["counted"] = True
        return {**reply, "endpoint": f"{outcome.route.hotkey}/{outcome.route.endpoint_id}", "url": outcome.route.url,
                "answers": outcome.answers, "input_tokens": outcome.input_tokens, "receipt": outcome.receipt,
                "response": outcome.response}

    return app


async def _to_target(g: Gateway, checked: Checked, target: Target, attempts: list[dict[str, Any]]):
    """One attempt at the named endpoint, outside the router's accounting (its breaker and load are untouched)."""
    endpoint = g.router.find(checked.parsed.model, target.hotkey, target.endpoint_id)
    if endpoint is None:
        attempts.append({"endpoint": f"{target.hotkey}/{target.endpoint_id}", "outcome": "failed",
                         "detail": "not a routable endpoint of this model"})
        return "failed"
    started = time.monotonic()
    note: dict[str, Any] = {"endpoint": endpoint.key}
    try:
        exchange = await g.upstream.call(endpoint.route, checked.raw, checked.body, checked.request_sha256,
                                         checked.parsed, g.config.request_timeout_s)
        note["outcome"] = "answered"
        return dataclasses.replace(exchange, seconds=time.monotonic() - started)
    except UpstreamFailure as failure:
        note.update(outcome="failed", kind=failure.kind, blame=failure.blame, detail=str(failure))
        return "failed"
    finally:
        note["ms"] = round(1000 * (time.monotonic() - started), 1)
        attempts.append(note)
