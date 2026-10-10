"""The gateway's admin API: inspection and debugging, for the operators only.

An app of its own (``admin_app``), served on a socket of its own (``admin_host:admin_port``: loopback or a private
network; ``ByPort`` hands it the requests that arrive there) and only with a token. On the public port none of its
routes exist. Every call is logged, and every credit with its caller.

Two tokens, two scopes. The admin token (``admin_token_file``) reaches everything. The account service's token
(``accounts_token_file``), for the web app that signs users up, reaches only the accounts named with
``accounts_prefix``, which it creates prepaid: their keys, usage, ledger and referral, changes other than ``prepaid``,
credit of kind grant or top-up within ``accounts_max_credit`` per entry and ``accounts_max_credit_per_hour`` (per
process; reaching a cap is logged as an error), refunds and negative adjustments; the network's usage totals and
series; and referral codes (registrations, look-ups, moves) and referrers' figures. Requests, state, routes and
revoking a code are the admin token's only.

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
    GET    /admin/v1/accounts/{name}/usage/series?days=&bucket=
                                                    the same per UTC day or week, every bucket of the range, all time
                                                    from the account's first usage (``Accounts.account_series``)
    POST   /admin/v1/accounts/{name}/credits        {"amount", "kind", "reference", "note", "refund_of"} -> 201, or
                                                    200 with the entry already recorded under ``reference``; 409 if it
                                                    records another one
    GET    /admin/v1/accounts/{name}/credits?limit= the ledger, newest first, and the balance
    GET    /admin/v1/usage/totals                   every account's answered requests and input tokens (all time, 24
                                                    hours, 30 days; per model), paid usage, top-ups and the accounts
                                                    that topped up (``Accounts.totals``)
    GET    /admin/v1/usage/series?days=&bucket=     the same per UTC day or week: requests and input tokens per model,
                                                    paid usage and top-ups (``Accounts.series``)

Referrals (``gateway.referrals``):

    POST   /admin/v1/referral-codes                 {"registration", "signature"} -> 201 with the code, 200 if its
                                                    hotkey has it already
    GET    /admin/v1/referral-codes/{code}          the code it reads as (any case, folded), with its hotkey
    DELETE /admin/v1/referral-codes/{code}          revokes it (the admin token only)
    POST   /admin/v1/referral-codes/move            {"move", "signature"} -> 200 with the codes the new hotkey holds
    PUT    /admin/v1/accounts/{name}/referral       {"code"} -> 201, or 200 if the account has that code already;
                                                    409 once it has another or a top-up
    GET    /admin/v1/accounts/{name}/referral       its referral or null, and whether it topped up
    GET    /admin/v1/referrers/{hotkey}?days=       per UTC day and code: accounts referred, paying, top-ups, refunds
                                                    and what counted (``Referrals.referrer``)
"""

from __future__ import annotations

import collections
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
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
)

from verathos.gateway.accounts import Conflict
from verathos.gateway.app import Checked, Gateway, Rejected, check_request, forward
from verathos.gateway.referrals import Refused
from verathos.gateway.upstream import Exchange, UpstreamFailure
from verathos.protocol.referral import SignedMove, SignedRegistration

log = logging.getLogger("verathos.gateway.admin")
MIN_TOKEN_CHARS = 32
NAME = r"^[A-Za-z0-9_.:@-]{1,128}$"
MAX_CREDIT = 10**15  # micro-USD in one entry, either sign (a billion USD): balances stay far within 64 bits
Rate = Annotated[float, Field(gt=0, allow_inf_nan=False)]


def read_token(path: Path, what: str = "the admin token") -> str:
    """A token (``what``) from its file; refuses a short token, and a file others can read."""
    path = path.expanduser()
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others: chmod 600 it")
    token = path.read_text().strip()
    if len(token) < MIN_TOKEN_CHARS:
        raise SystemExit(f"{what} in {path} is shorter than {MIN_TOKEN_CHARS} characters")
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
    rate_rps: Rate
    burst: Annotated[StrictInt, Field(ge=1)]
    monthly_quota: Annotated[StrictInt, Field(ge=1)] | None = None
    prepaid: StrictBool = False


class AccountChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["active", "suspended"] | None = None
    rate_rps: Rate | None = None
    burst: Annotated[StrictInt, Field(ge=1)] | None = None
    monthly_quota: Annotated[StrictInt, Field(ge=1)] | None = None  # null: unlimited
    prepaid: StrictBool | None = None


class NewKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(min_length=1, max_length=128)]
    expires_at: AwareDatetime | None = None


class NewCredit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: Annotated[StrictInt, Field(ge=-MAX_CREDIT, le=MAX_CREDIT)]
    kind: Literal["grant", "topup", "adjust", "refund"]
    reference: Annotated[str, Field(min_length=1, max_length=128)]
    note: Annotated[str, Field(max_length=256)] | None = None
    refund_of: Annotated[str, Field(min_length=1, max_length=128)] | None = None  # a refund: the top-up it returns

    @field_validator("amount")
    @classmethod
    def _not_zero(cls, value: int) -> int:
        if value == 0:
            raise ValueError("an entry is at least one micro-USD")
        return value


class NewReferral(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: Annotated[str, Field(min_length=1, max_length=64)]


def error(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


def series_days(days: str | None) -> int | None:
    """A usage series' ``days``: a whole number, or None for all time. Raises ValueError."""
    if days is not None and not (days.isascii() and days.isdigit() and len(days) <= 6):
        raise ValueError("days is a whole number of days")
    return None if days is None else int(days)


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


def admin_app(g: Gateway, *, token: str, accounts_token: str | None = None) -> FastAPI:
    """``accounts_token``: the account service's (see the module docstring)."""
    app = FastAPI(title="Verathos gateway admin", docs_url=None, redoc_url=None, openapi_url=None)
    scopes = [(f"Bearer {token}".encode(), "admin")]
    if accounts_token is not None:
        scopes.append((f"Bearer {accounts_token}".encode(), "accounts"))
    credited: collections.deque[tuple[float, int]] = collections.deque()  # the account service's credit, last hour

    def reaches(request: Request) -> bool:
        """Whether the caller's scope reaches this call: the account service only its own accounts' routes, the
        network's usage, and referral codes and referrers' figures (revoking a code aside)."""
        if request.state.scope == "admin" or request.url.path in ("/admin/v1/usage/totals", "/admin/v1/usage/series"):
            return True
        if request.url.path.startswith(("/admin/v1/referral-codes", "/admin/v1/referrers/")):
            return request.method != "DELETE"
        parts = request.url.path.split("/")  # "", "admin", "v1", "accounts", name, ...
        return parts[1:4] == ["admin", "v1", "accounts"] and (
            len(parts) == 4 or parts[4].startswith(g.config.accounts_prefix))

    @app.middleware("http")
    async def authorized(request: Request, call_next):
        client = request.client.host if request.client else "?"
        given = request.headers.get("authorization", "").encode()
        scope = next((name for expected, name in scopes if hmac.compare_digest(given, expected)), None)
        if scope is None:
            log.warning("admin call %s %s with a wrong token from %s", request.method, request.url.path, client)
            return JSONResponse({"error": {"type": "unauthorized", "message": "admin token required"}},
                                status_code=401)
        request.state.scope, request.state.caller = scope, f"{scope} token from {client}"
        if not reaches(request):
            log.warning("admin call %s %s by the %s: out of its scope", request.method, request.url.path,
                        request.state.caller)
            return error(403, "forbidden", "out of this token's scope")
        response = await call_next(request)
        log.info("admin call %s %s by the %s -> %d", request.method, request.url.path, request.state.caller,
                 response.status_code)
        return response

    def over_cap(request: Request, name: str, new: NewCredit) -> JSONResponse | tuple[float, int] | None:
        """The account service's credit limits (kinds, per entry, per hour): the refusal, or the share of the hour's
        cap the entry holds until it is known to be new; None for the admin token, and for entries that take credit
        back (refunds, negative adjustments)."""
        if request.state.scope == "admin" or new.kind == "refund" or (new.kind == "adjust" and new.amount < 0):
            return None
        if new.kind not in ("grant", "topup"):
            return error(403, "forbidden", "the account service records grants, top-ups, refunds and negative "
                                           "adjustments only")
        now = time.monotonic()
        while credited and credited[0][0] < now - 3600:
            credited.popleft()
        hour = sum(amount for _, amount in credited)
        if new.amount > g.config.accounts_max_credit or hour + new.amount > g.config.accounts_max_credit_per_hour:
            log.error("credit of %d micro-USD to %s refused, over the account service's cap (%d per entry, %d an "
                      "hour, %d recorded this hour): %s", new.amount, name, g.config.accounts_max_credit,
                      g.config.accounts_max_credit_per_hour, hour, request.state.caller)
            return error(403, "forbidden", "over the account service's credit cap")
        held = (now, new.amount)
        credited.append(held)
        return held

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
            checked = await check_request(g, json.dumps(command.request).encode())
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
                "answers": outcome.answers, "input_tokens": outcome.input_tokens, "image_tokens": checked.image_tokens,
                "receipt": outcome.receipt, "response": outcome.response}

    accounts = g.accounts

    async def lookup(name: str) -> dict[str, Any] | JSONResponse:
        account = await accounts.account(name)
        return error(404, "not_found", f"no account {name!r}") if account is None else account

    @app.post("/admin/v1/accounts")
    async def create_account(request: Request):
        new = await parse(request, NewAccount)
        if isinstance(new, JSONResponse):
            return new
        if request.state.scope != "admin" and not (new.name.startswith(g.config.accounts_prefix) and new.prepaid):
            return error(403, "forbidden", f"the account service creates prepaid accounts named "
                                           f"{g.config.accounts_prefix}... only")
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
        if request.state.scope != "admin" and "prepaid" in values:
            return error(403, "forbidden", "the account service does not change prepaid")
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

    @app.get("/admin/v1/accounts/{name}/usage/series")
    async def account_series(name: str, days: str | None = None, bucket: str = "day"):
        try:
            found = await accounts.account_series(name, series_days(days), bucket)
        except ValueError as exc:
            return error(400, "invalid_request", str(exc))
        return error(404, "not_found", f"no account {name!r}") if found is None else found

    @app.post("/admin/v1/accounts/{name}/credits")
    async def add_credit(name: str, request: Request):
        new = await parse(request, NewCredit)
        if isinstance(new, JSONResponse):
            return new
        recorded = request.state.scope != "admin" and await accounts.credit(new.reference) is not None
        held = None if recorded else over_cap(request, name, new)  # a repeat records nothing more
        if isinstance(held, JSONResponse):
            return held
        log.info("credit %r of %d micro-USD (%s%s) to %s by the %s", new.reference, new.amount, new.kind,
                 f" of {new.refund_of!r}" if new.refund_of else "", name, request.state.caller)
        created = False
        try:
            entry, created = await accounts.add_credit(name, new.amount, kind=new.kind, reference=new.reference,
                                                       note=new.note, refund_of=new.refund_of)
        except LookupError as exc:
            return error(404, "not_found", str(exc))
        except ValueError as exc:
            return error(400, "invalid_request", str(exc))
        except Conflict as exc:
            return error(409, "conflict", str(exc))
        finally:
            if held is not None and not created:  # a repeat or a failure records nothing: it takes none of the cap
                credited.remove(held)
        account = await accounts.account(name)
        return JSONResponse(jsonable_encoder({"credit": entry, "created": created, "balance": account["balance"]}),
                            status_code=201 if created else 200)

    @app.get("/admin/v1/usage/totals")
    async def totals():
        return await accounts.totals()

    @app.get("/admin/v1/usage/series")
    async def series(days: str | None = None, bucket: str = "day"):
        try:
            return await accounts.series(series_days(days), bucket)
        except ValueError as exc:
            return error(400, "invalid_request", str(exc))

    @app.get("/admin/v1/accounts/{name}/credits")
    async def ledger(name: str, limit: int = 100):
        account = await lookup(name)
        if isinstance(account, JSONResponse):
            return account
        return {"balance": account["balance"], "credits": await accounts.credits(name, max(1, min(limit, 1000)))}

    referrals = g.referrals

    @app.post("/admin/v1/referral-codes")
    async def register_code(request: Request):
        signed = await parse(request, SignedRegistration)
        if isinstance(signed, JSONResponse):
            return signed
        try:
            code, created = await referrals.register(signed)
        except Refused as r:
            return error(r.status, r.kind, r.message)
        if created:
            log.info("referral code %r registered for %s by the %s", code["code"], code["hotkey"],
                     request.state.caller)
        return JSONResponse(jsonable_encoder(code), status_code=201 if created else 200)

    @app.post("/admin/v1/referral-codes/move")
    async def move_codes(request: Request):
        signed = await parse(request, SignedMove)
        if isinstance(signed, JSONResponse):
            return signed
        try:
            answer, moved = await referrals.move(signed)
        except Refused as r:
            return error(r.status, r.kind, r.message)
        if moved:
            log.info("referral codes %s moved from %s to %s by the %s", ", ".join(moved), answer["from_hotkey"],
                     answer["to_hotkey"], request.state.caller)
        return answer

    @app.get("/admin/v1/referral-codes/{code}")
    async def get_code(code: str):
        found = await referrals.code(code)
        return error(404, "not_found", f"no referral code {code!r}") if found is None else found

    @app.delete("/admin/v1/referral-codes/{code}")
    async def revoke_code(code: str, request: Request):
        found = await referrals.revoke(code)
        if found is None:
            return error(404, "not_found", f"no referral code {code!r}")
        log.info("referral code %r of %s revoked by the %s", found["code"], found["hotkey"], request.state.caller)
        return found

    @app.put("/admin/v1/accounts/{name}/referral")
    async def refer(name: str, request: Request):
        new = await parse(request, NewReferral)
        if isinstance(new, JSONResponse):
            return new
        try:
            referral, created = await referrals.refer(name, new.code)
        except LookupError as exc:
            return error(404, "not_found", str(exc))
        except Refused as r:
            return error(r.status, r.kind, r.message)
        return JSONResponse(jsonable_encoder(referral), status_code=201 if created else 200)

    @app.get("/admin/v1/accounts/{name}/referral")
    async def get_referral(name: str):
        found = await referrals.referral(name)
        return error(404, "not_found", f"no account {name!r}") if found is None else found

    @app.get("/admin/v1/referrers/{hotkey}")
    async def referrer(hotkey: str, days: str | None = None):
        try:
            count = series_days(days)
            return await referrals.referrer(hotkey, 30 if count is None else count)
        except ValueError as exc:
            return error(400, "invalid_request", str(exc))

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
                                         checked.parsed, g.config.request_timeout_s, min_tokens=checked.min_tokens,
                                         max_tokens=checked.max_tokens)
        note["outcome"] = "answered"
        return dataclasses.replace(exchange, seconds=time.monotonic() - started)
    except UpstreamFailure as failure:
        note.update(outcome="failed", kind=failure.kind, blame=failure.blame, detail=str(failure))
        return "failed"
    finally:
        note["ms"] = round(1000 * (time.monotonic() - started), 1)
        attempts.append(note)
