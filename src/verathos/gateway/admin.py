"""The gateway's admin API: inspection and debugging, for the operators only.

Served on a socket of its own (``admin_host:admin_port``: loopback or a private network) and only with the admin
token (``admin_token_file``); on the public port these routes do not exist. Every call is logged.

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

import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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


def add_admin_routes(app: FastAPI, g: Gateway, *, port: int, token: str) -> None:
    expected = f"Bearer {token}".encode()

    def denied(request: Request) -> JSONResponse | None:
        server = request.scope.get("server")
        if not server or server[1] != port:
            return JSONResponse({"error": {"type": "not_found", "message": "not found"}}, status_code=404)
        given = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(given, expected):
            log.warning("admin call to %s with a wrong token from %s", request.url.path,
                        request.client.host if request.client else "?")
            return JSONResponse({"error": {"type": "unauthorized", "message": "admin token required"}},
                                status_code=401)
        return None

    @app.get("/admin/v1/state")
    async def state(request: Request):
        if (refused := denied(request)) is not None:
            return refused
        cat = g.catalog()
        return {"hotkey": g.hotkey, "catalog": cat.digest() if cat else None, "routes": g.routes_state(),
                "models": sorted(g.router.models()), "pending": g.pending_state(),
                "limits": {"instances": g.config.instances, "max_in_flight": g.config.max_in_flight,
                           "request_timeout_s": g.config.request_timeout_s,
                           "attempt_timeout_s": g.config.attempt_timeout_s, "max_attempts": g.config.max_attempts}}

    @app.get("/admin/v1/routes")
    async def routes(request: Request):
        if (refused := denied(request)) is not None:
            return refused
        return {"routes": g.router.describe()}

    @app.post("/admin/v1/systemone")
    async def debug_request(request: Request):
        if (refused := denied(request)) is not None:
            return refused
        try:
            command = json.loads(await request.body())
            body = json.dumps(command["request"]).encode()
            target, count = command.get("target"), bool(command.get("count", False))
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": {"type": "invalid_request", "message": f"{type(exc).__name__}: {exc}"}},
                                status_code=400)
        try:
            checked = check_request(g, body)
        except Rejected as r:
            return JSONResponse({"error": {"type": r.kind, "message": r.message}}, status_code=r.status)
        started, attempts = g.clock(), []
        if target:
            outcome = await _to_target(g, checked, target, attempts)
        else:
            outcome = await forward(g, checked, started, attempts, counted=count)
        log.info("admin request %s target=%s count=%s -> %s", checked.parsed.model,
                 f"{target.get('hotkey')}/{target.get('endpoint_id')}" if target else "router", count,
                 "answered" if isinstance(outcome, Exchange) else outcome)
        reply: dict[str, Any] = {"model": checked.parsed.model, "attempts": attempts, "counted": False,
                                 "ms": round(1000 * (g.clock() - started), 1)}
        if not isinstance(outcome, Exchange):
            return JSONResponse({**reply, "error": outcome}, status_code=502)
        if count:
            g.sampler.offer(outcome)
            g.served.record(outcome.route, outcome.input_tokens, outcome.seconds)
            reply["counted"] = True
        return {**reply, "endpoint": f"{outcome.route.hotkey}/{outcome.route.endpoint_id}", "url": outcome.route.url,
                "answers": outcome.answers, "input_tokens": outcome.input_tokens, "receipt": outcome.receipt,
                "response": outcome.response}


async def _to_target(g: Gateway, checked: Checked, target: dict[str, Any], attempts: list[dict[str, Any]]):
    """One attempt at the named endpoint, outside the router's accounting (its breaker and load are untouched)."""
    endpoint = g.router.find(checked.parsed.model, str(target.get("hotkey")), str(target.get("endpoint_id")))
    if endpoint is None:
        attempts.append({"endpoint": f"{target.get('hotkey')}/{target.get('endpoint_id')}", "outcome": "failed",
                         "detail": "not a routable endpoint of this model"})
        return "failed"
    started = time.monotonic()
    note: dict[str, Any] = {"endpoint": endpoint.key}
    try:
        exchange = await g.upstream.call(endpoint.route, checked.raw, checked.body, checked.request_sha256,
                                         checked.parsed, g.config.request_timeout_s)
        note["outcome"] = "answered"
        return exchange
    except UpstreamFailure as failure:
        note.update(outcome="failed", kind=failure.kind, blame=failure.blame, detail=str(failure))
        return "failed"
    finally:
        note["ms"] = round(1000 * (time.monotonic() - started), 1)
        attempts.append(note)
