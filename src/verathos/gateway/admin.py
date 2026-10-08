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

Accounts, for the service that signs users up (amounts in micro-USD; ``name``: ``[A-Za-z0-9_.:@-]{1,128}``):

    POST   /admin/v1/accounts                       {"name", "rate_rps", "burst", "monthly_quota", "prepaid"} -> 201
    GET    /admin/v1/accounts/{name}                the account and its balance as written
    PATCH  /admin/v1/accounts/{name}                {"status", "rate_rps", "burst", "monthly_quota", "prepaid"}
    POST   /admin/v1/accounts/{name}/keys           {"name", "expires_at"} -> 201 with the key, shown only here
    GET    /admin/v1/accounts/{name}/keys
    DELETE /admin/v1/accounts/{name}/keys/{prefix}  revokes the key
    GET    /admin/v1/accounts/{name}/usage?since=   per day and model: requests, failed, input tokens, cost
    POST   /admin/v1/accounts/{name}/credits        {"amount", "kind", "reference", "note"} -> 201, or 200 with the
                                                    entry already recorded under ``reference``; 409 if it records
                                                    another one
    GET    /admin/v1/accounts/{name}/credits?limit= the ledger, newest first, and the balance
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError

from verathos.gateway.accounts import Conflict
from verathos.gateway.app import Checked, Gateway, Rejected, check_request, forward
from verathos.gateway.upstream import Exchange, UpstreamFailure

log = logging.getLogger("verathos.gateway.admin")
MIN_TOKEN_CHARS = 32
NAME = r"^[A-Za-z0-9_.:@-]{1,128}$"


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


class NewAccount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=NAME)]
    rate_rps: Annotated[float, Field(gt=0)]
    burst: Annotated[StrictInt, Field(ge=1)]
    monthly_quota: Annotated[StrictInt, Field(ge=1)] | None = None
    prepaid: StrictBool = False


class AccountChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "suspended"] | None = None
    rate_rps: Annotated[float, Field(gt=0)] | None = None
    burst: Annotated[StrictInt, Field(ge=1)] | None = None
    monthly_quota: Annotated[StrictInt, Field(ge=1)] | None = None  # null: unlimited
    prepaid: StrictBool | None = None


class NewKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=128)]
    expires_at: AwareDatetime | None = None


class NewCredit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: StrictInt
    kind: Literal["grant", "topup", "adjust"]
    reference: Annotated[str, Field(min_length=1, max_length=128)]
    note: Annotated[str, Field(max_length=256)] | None = None


def error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


async def parse(request: Request, model: type[BaseModel]) -> BaseModel | JSONResponse:
    try:
        return model.model_validate_json(await request.body())
    except ValidationError as exc:
        first = exc.errors()[0]
        return error(400, "invalid_request", f"{'.'.join(map(str, first['loc'])) or 'body'}: {first['msg']}")


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
        command = await parse(request, Command)
        if isinstance(command, JSONResponse):
            return command
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

    accounts = g.accounts

    async def lookup(name: str) -> dict[str, Any] | JSONResponse:
        account = await accounts.account(name)
        return error(404, "not_found", f"no account {name!r}") if account is None else account

    @app.post("/admin/v1/accounts")
    async def create_account(request: Request):
        new = await parse(request, NewAccount)
        if isinstance(new, JSONResponse):
            return new
        try:
            await accounts.create_account(new.name, rate_rps=new.rate_rps, burst=new.burst,
                                          monthly_quota=new.monthly_quota, prepaid=new.prepaid)
        except Conflict as exc:
            return error(409, "conflict", str(exc))
        return JSONResponse(jsonable_encoder(await accounts.account(new.name)), status_code=201)

    @app.get("/admin/v1/accounts/{name}")
    async def get_account(name: str):
        return await lookup(name)

    @app.patch("/admin/v1/accounts/{name}")
    async def change_account(name: str, request: Request):
        change = await parse(request, AccountChange)
        if isinstance(change, JSONResponse):
            return change
        values = change.model_dump(exclude_unset=True)
        if any(v is None for k, v in values.items() if k != "monthly_quota"):
            return error(400, "invalid_request", "only monthly_quota may be null")
        if values and not await accounts.set_account(name, **values):
            return error(404, "not_found", f"no account {name!r}")
        return await lookup(name)

    @app.post("/admin/v1/accounts/{name}/keys")
    async def create_key(name: str, request: Request):
        new = await parse(request, NewKey)
        if isinstance(new, JSONResponse):
            return new
        try:
            key = await accounts.create_key(name, new.name, new.expires_at)
        except LookupError as exc:
            return error(404, "not_found", str(exc))
        listed = next(k for k in await accounts.list_keys(name) if key.startswith(k["prefix"]))
        return JSONResponse(jsonable_encoder({**listed, "key": key}), status_code=201)

    @app.get("/admin/v1/accounts/{name}/keys")
    async def keys(name: str):
        account = await lookup(name)
        return account if isinstance(account, JSONResponse) else {"keys": await accounts.list_keys(name)}

    @app.delete("/admin/v1/accounts/{name}/keys/{prefix}")
    async def revoke_key(name: str, prefix: str):
        if not await accounts.revoke_key(name, prefix):
            return error(404, "not_found", f"no active key {prefix!r} of account {name!r}")
        return {"revoked": prefix}

    @app.get("/admin/v1/accounts/{name}/usage")
    async def usage(name: str, since: str | None = None):
        account = await lookup(name)
        if isinstance(account, JSONResponse):
            return account
        try:
            day = dt.date.fromisoformat(since) if since else dt.datetime.now(dt.UTC).date().replace(day=1)
        except ValueError:
            return error(400, "invalid_request", "since is an ISO date")
        return {"since": day, "usage": await accounts.usage(name, day)}

    @app.post("/admin/v1/accounts/{name}/credits")
    async def add_credit(name: str, request: Request):
        new = await parse(request, NewCredit)
        if isinstance(new, JSONResponse):
            return new
        try:
            entry, created = await accounts.add_credit(name, new.amount, kind=new.kind, reference=new.reference,
                                                       note=new.note)
        except LookupError as exc:
            return error(404, "not_found", str(exc))
        except ValueError as exc:
            return error(400, "invalid_request", str(exc))
        except Conflict as exc:
            return error(409, "conflict", str(exc))
        account = await accounts.account(name)
        return JSONResponse(jsonable_encoder({"credit": entry, "created": created, "balance": account["balance"]}),
                            status_code=201 if created else 200)

    @app.get("/admin/v1/accounts/{name}/credits")
    async def ledger(name: str, limit: int = 100):
        account = await lookup(name)
        if isinstance(account, JSONResponse):
            return account
        return {"balance": account["balance"], "credits": await accounts.credits(name, max(1, min(limit, 1000)))}

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
                                         checked.parsed, g.config.request_timeout_s, min_tokens=checked.min_tokens)
        note["outcome"] = "answered"
        return dataclasses.replace(exchange, seconds=time.monotonic() - started)
    except UpstreamFailure as failure:
        note.update(outcome="failed", kind=failure.kind, blame=failure.blame, detail=str(failure))
        return "failed"
    finally:
        note["ms"] = round(1000 * (time.monotonic() - started), 1)
        attempts.append(note)
