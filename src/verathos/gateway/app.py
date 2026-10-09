"""The gateway's HTTP API.

    POST /v1/systemone   a SystemOne request with ``Authorization: Bearer <key>`` -> the answers, the usage, the
                         miner's signed receipt and the request id
    GET  /v1/models      the catalog's active models: state limit, the input each reads (and its image limits),
                         price, weights, whether available now, state and latency
    GET  /v1/status      per model: operational, degraded or unavailable; endpoints, capacity, latency, success
    GET  /health         the process is up
    GET  /ready          routes are loaded and the gateway is in the catalog (503 otherwise), and how old they are

Every response to ``POST /v1/systemone`` carries ``X-Request-Id`` (the client's, if it sent a valid one). Errors are
``{"error": {"type": ..., "message": ...}}``; 429 and 503 carry ``Retry-After`` (for a used-up quota: until the next
month).

    400 invalid_request     not JSON (NaN and infinities included), not a SystemOne request, images to a model that
                            reads text only or images it cannot read (a URL, not base64, not PNG, JPEG or WebP, a
                            content_type the bytes contradict, undecodable, an aspect ratio beyond 200:1), videos or
                            media_kwargs
    401 unauthorized        missing, unknown, revoked or expired key, or a suspended account
    402 insufficient_credit a prepaid account's balance is used up
    404 model_not_found     not an active model of the catalog
    408 timeout             the body took longer than ``body_timeout_s`` to arrive
    413 too_large           body, state, images, or the answers the questions would need, over the limits
    429 rate_limited        over the key's rate (or too many invalid keys from one address)
    429 quota_exceeded      over the account's monthly quota
    500 internal            a fault of the gateway
    503 busy                no capacity free for the model right now, the gateway is at capacity, or its image
                            decoders are all busy
    503 unavailable         no miner serves the model right now, the miners failed or refused the request within its
                            deadline, every miner of the model is failing, accounts cannot be checked, the gateway
                            has no catalog yet, or its image decoders cannot start

A request is forwarded as its model, state and questions (other fields are dropped) with its images made canonical
(``protocol.media``, in the decode workers of ``gateway.media``): its canonical JSON is the body the miners receive
and their receipts bind.
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
from verathos.gateway.media import CanonicalImage, Decoders, DecodersBusy, DecodingUnavailable
from verathos.gateway.router import Endpoint, Outcome, Router
from verathos.gateway.routes import stale_after_s
from verathos.gateway.sampler import Sampler
from verathos.gateway.served import ServedCounter
from verathos.gateway.stats import DEGRADED_SUCCESS, ModelStats
from verathos.gateway.upstream import Exchange, Upstream, UpstreamFailure
from verathos.gateway.usage import UsageMeter, month_start
from verathos.protocol.api import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    REFUSED_FIELDS,
    STATE_CHARS_PER_TOKEN,
    Choice,
    SystemOneRequest,
    prompt_chars,
    question_keys,
    text,
)
from verathos.protocol.canonical import canonical_json, sha256_hex
from verathos.protocol.gateway import Routes
from verathos.protocol.media import (
    BASE64_ALPHABET,
    ENGINE_IMAGES,
    ImageRefused,
    request_image_tokens,
    sent_image,
)
from verathos.throughput.workload import in_mix_form
from verathos.validator.canaries import canary_like

log = logging.getLogger(__name__)
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
    decoders: Decoders | None = None  # by default from the configuration

    def __post_init__(self) -> None:
        if self.decoders is None:
            self.decoders = Decoders(self.config.media_workers, self.config.max_media_in_flight,
                                     self.config.media_timeout_s)

    def body_limit(self) -> int:
        """The most a request's body may hold: a text model's ``max_body_bytes``, or an active image model's
        ``max_request_bytes`` if larger."""
        cat = self.catalog()
        image_models = [m.images.max_request_bytes for m in (cat.models if cat is not None else [])
                        if m.status == "active" and m.reads_images()]
        return max([self.config.max_body_bytes, *image_models])

    def routes_state(self) -> dict[str, Any] | None:
        """The routes in use; stale: older than ``routes.stale_after_s``."""
        table, cat = self.routes(), self.catalog()
        if table is None:
            return None
        age = time.time() - table.issued_ms / 1000
        return {"epoch": table.epoch, "issued_ms": table.issued_ms, "age_s": round(age, 1),
                "stale": cat is not None and age > stale_after_s(cat), "endpoints": len(table.routes),
                "catalog": table.catalog_digest}

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
    raw: dict[str, Any]  # the request as forwarded: its model, state and questions, and its images made canonical
    body: bytes  # its canonical JSON
    parsed: SystemOneRequest
    request_sha256: str
    price: float  # the model's, USD per million input tokens
    min_tokens: int  # billed at least, whatever the miner signs: one per 8 characters of the state, and the images'
    max_tokens: int  # billed at most: one per byte of the request without its images, and the images' tokens
    image_tokens: int = 0  # the images' input tokens (``protocol.media.request_image_tokens``)


@dataclass(frozen=True)
class Screened:
    """A request that passed the checks made before its images are decoded."""

    parsed: SystemOneRequest
    model: catalog_module.CatalogModel
    forwarded: dict[str, Any]  # what is forwarded of it besides its images: its model, state and questions
    text_body: bytes  # their canonical JSON
    images: list[tuple[str, str]]  # (media type, base64) of each image as sent

    def most_tokens(self) -> int:
        """The most input tokens it can be billed: one per byte of it without its images, and the most image tokens
        its model's limits allow."""
        return len(self.text_body) + (self.model.images.max_tokens() if self.images else 0)


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


def model_body_limit(g: Gateway, model_id: Any) -> int:
    """The most a request's body may hold for the model it names: its catalog's ``images.max_request_bytes`` for an
    active model that reads images, ``max_body_bytes`` for any other."""
    cat = g.catalog()
    model = cat.get(model_id) if cat is not None and isinstance(model_id, str) else None
    if model is not None and model.status == "active" and model.reads_images():
        return model.images.max_request_bytes
    return g.config.max_body_bytes


def screen(g: Gateway, body: bytes) -> Screened:
    """The checks made before any image is decoded: within the body limit of the model it names, beyond the text
    limit (``max_body_bytes``) only by base64 (so parsing a large body costs no more than a text one), strict JSON, a
    SystemOne request without videos or media_kwargs for an active model miners serve now, within the model's prompt
    limit (``prompt_chars``) and ``MAX_REQUEST_BYTES`` without its images, with answers that fit a response, whose
    state is not one of the subnet's test requests or canaries (a miner relaying them through the gateway would have
    its answers counted as users'), and with images only to a model that reads them, within its limits as far as they
    can be checked undecoded (``protocol.media.sent_image``). Raises Rejected."""
    text_limit = g.config.max_body_bytes
    if len(body) > text_limit and len(body.translate(None, BASE64_ALPHABET)) > text_limit:
        raise Rejected(413, "too_large", f"the body exceeds {text_limit} bytes besides its images' base64")
    try:
        raw = json.loads(body, parse_constant=_no_constants)
        if isinstance(raw, dict):
            if refused := [why for field, why in REFUSED_FIELDS.items() if field in raw]:
                raise Rejected(400, "invalid_request", refused[0])
            if len(body) > (limit := model_body_limit(g, raw.get("model"))):
                raise Rejected(413, "too_large", f"the body exceeds {limit} bytes")
        parsed = SystemOneRequest.model_validate(raw)
        prompt = prompt_chars(parsed)
    except ValidationError as exc:
        raise Rejected(400, "invalid_request", f"not a SystemOne request: {validation_message(exc)}") from None
    except (ValueError, TypeError, RecursionError) as exc:
        raise Rejected(400, "invalid_request", f"not a SystemOne request: {exc}"[:500]) from None
    if in_mix_form(parsed.state) or canary_like(parsed.state):  # relayed, so that the gateway's answers count
        raise Rejected(400, "invalid_request", "the state has the form of the subnet's test requests or canaries")
    cat = g.catalog()
    if cat is None:
        raise Rejected(503, "unavailable", "the gateway has no catalog yet", 5)
    model = cat.get(parsed.model)
    if model is None or model.status != "active":
        raise Rejected(404, "model_not_found", f"{parsed.model} is not served")
    if parsed.images and not model.reads_images():
        raise Rejected(400, "invalid_request", f"{model.model_id} reads text only", model=model.model_id)
    if parsed.model not in g.router.models():
        raise Rejected(503, "unavailable", f"no miner serves {parsed.model} right now", 30, parsed.model)
    if prompt > STATE_CHARS_PER_TOKEN * model.max_state_tokens:
        raise Rejected(413, "too_large", f"the state, instructions and criteria exceed what {model.model_id} accepts",
                       model=parsed.model)
    if answer_bytes(parsed) > MAX_RESPONSE_BYTES:
        raise Rejected(413, "too_large", "the answers to these questions would exceed a response's size limit",
                       model=parsed.model)
    forwarded = {"model": raw["model"], "state": raw["state"], "questions": raw["questions"]}
    try:
        text_body = canonical_json(forwarded)
    except (ValueError, TypeError, RecursionError) as exc:
        raise Rejected(400, "invalid_request", f"not a SystemOne request: {exc}"[:500]) from None
    if len(text_body) > MAX_REQUEST_BYTES:
        raise Rejected(413, "too_large", f"the request without its images exceeds {MAX_REQUEST_BYTES} bytes",
                       model=model.model_id)
    if parsed.images and (token := ENGINE_IMAGES[model.engine.name].image_token).encode() in text_body:
        raise Rejected(400, "invalid_request", f"a request with images may not hold {token} in its text",
                       model=model.model_id)
    return Screened(parsed, model, forwarded, text_body, sent_images(parsed.images or [], model))


def sent_images(images: list[Any], model: catalog_module.CatalogModel) -> list[tuple[str, str]]:
    """(media type, base64) of each image of a request, checked within the model's limits as far as they can be
    undecoded. Raises Rejected."""
    limits, sent, total = model.images, [], 0
    if len(images) > limits.max_count:
        raise Rejected(413, "too_large", f"{model.model_id} takes at most {limits.max_count} images per request",
                       model=model.model_id)
    for index, item in enumerate(images):
        try:
            content_type, data, size = sent_image(item, limits)
        except ImageRefused as refused:
            raise rejection(refused, f"images.{index}: {refused}", model.model_id) from None
        sent.append((content_type, data))
        total += size
    if total > limits.max_total_bytes:
        raise Rejected(413, "too_large", f"the images hold {total} bytes, more than {model.model_id} takes "
                                         f"({limits.max_total_bytes})", model=model.model_id)
    return sent


def rejection(refused: ImageRefused, message: str, model_id: str) -> Rejected:
    return Rejected(refused.status, "too_large" if refused.status == 413 else "invalid_request", message,
                    model=model_id)


def with_images(text_body: bytes, images: list[CanonicalImage]) -> bytes:
    """The canonical JSON of a request with its images, from that of the request without them: ``images`` sorts
    before its other fields, and neither base64 nor a media type needs escaping."""
    items = b",".join(b'{"base64":"' + i.base64.encode("ascii") + b'","content_type":"' + i.content_type.encode()
                      + b'"}' for i in images)
    return b'{"images":[' + items + b"]," + text_body[1:]


async def made_canonical(g: Gateway, screened: Screened) -> Checked:
    """A screened request with its images made canonical by the decoders: the request as forwarded, at most the
    model's ``max_request_bytes`` with its images, and the bounds of the input tokens it is billed. Raises
    Rejected."""
    model, model_id = screened.model, screened.model.model_id
    images: list[CanonicalImage] = []
    if screened.images:
        try:
            images = await g.decoders.canonical(screened.images, model.images)
        except DecodersBusy:
            raise Rejected(503, "busy", "the gateway's image decoders are all busy", 1, model_id) from None
        except DecodingUnavailable:
            raise Rejected(503, "unavailable", "the gateway cannot decode images right now", 5, model_id) from None
        except ImageRefused as refused:
            raise rejection(refused, str(refused), model_id) from None
    image_tokens = request_image_tokens([(i.width, i.height) for i in images], model.images.token_px)
    forwarded, body = screened.forwarded, screened.text_body
    if images:
        forwarded = {"images": [{"base64": i.base64, "content_type": i.content_type} for i in images], **forwarded}
        body = with_images(body, images)
        if len(body) > model.images.max_request_bytes:
            raise Rejected(413, "too_large", f"with its images made canonical the request exceeds "
                                             f"{model.images.max_request_bytes} bytes, {model_id}'s limit",
                           model=model_id)
    min_tokens = len(text(screened.parsed.state)) // STATE_CHARS_PER_BILLED_TOKEN + image_tokens
    return Checked(forwarded, body, screened.parsed, sha256_hex(body), model.price, min_tokens,
                   len(screened.text_body) + image_tokens, image_tokens)


async def check_request(g: Gateway, body: bytes) -> Checked:
    """A body the gateway can forward (``screen``), with its images made canonical (``made_canonical``). Raises
    Rejected."""
    return await made_canonical(g, screen(g, body))


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
    unsettled: list[Endpoint] = []  # breaker probes that ended counting neither way, yet
    try:
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
            probe = endpoint.breaker.probing  # this attempt is the breaker's one probe
            outcome: Outcome | None = None
            attempt_started = g.clock()
            note = {"endpoint": endpoint.key}
            try:
                exchange = await g.upstream.call(endpoint.route, checked.raw, checked.body, checked.request_sha256,
                                                 checked.parsed, min(g.config.attempt_timeout_s, remaining),
                                                 min_tokens=checked.min_tokens, max_tokens=checked.max_tokens)
                outcome = Outcome.ANSWERED
                note["outcome"] = "answered"
            except UpstreamFailure as failure:
                metrics.UPSTREAM_FAILURES.labels(failure.kind).inc()
                if times := _throttled(f"{endpoint.key} {failure.kind}"):
                    log.info("an attempt at %s failed: %s, blamed on %s (%d times since last logged)", endpoint.key,
                             failure, failure.blame, times)
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
                g.router.finished(endpoint, outcome, seconds, observe=counted, why=note.get("detail", ""))
                if outcome is None and probe:
                    unsettled.append(endpoint)
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
    finally:  # unless penalized, a probe that counted neither way stays so, also when the request is cancelled
        for other in unsettled:
            g.router.settled(other)


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
    minutes' requests answered (of those not refused for want of capacity), or more than half the endpoints failing;
    else operational."""
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
    in_flight = in_flight_bytes = 0
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
        if g.hotkey not in cat.gateway_hotkeys():
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
        routes = g.routes_state()
        return {"ready": True, "routes": {"age_s": routes["age_s"], "stale": routes["stale"]}}

    def models_document() -> dict[str, Any]:
        cat, states = g.catalog(), mark_states(g)
        return {"object": "list", "data": [] if cat is None else [{
            "id": m.model_id, "object": "model", "max_state_tokens": m.max_state_tokens,
            "input_modalities": m.modalities(), "images": m.images.document() if m.reads_images() else None,
            "price": {"usd_per_million_input_tokens": m.price, "usd_per_million_output_tokens": 0},
            "available": states.get(m.model_id, "unavailable") != "unavailable",
            "state": states.get(m.model_id, "unavailable"), "latency_ms": g.stats.latency(m.model_id),
            "license": m.license, "weights": {"repo": m.hf_repo, "revision": m.hf_revision},
            "base_model": {"repo": m.base_repo, "revision": m.base_revision} if m.base_repo else None,
        } for m in cat.models if m.status == "active"]}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return recent("models", models_document)

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
        nonlocal in_flight, in_flight_bytes
        given = request.headers.get("x-request-id", "")
        request_id = given if _REQUEST_ID.fullmatch(given) else secrets.token_hex(12)
        request.state.request_id, request.state.held_bytes = request_id, 0
        if in_flight >= g.config.max_in_flight:
            return reject(503, "busy", "the gateway is at capacity", request_id, 1)
        in_flight += 1
        metrics.IN_FLIGHT.set(in_flight)
        try:
            return await serve(request, request_id)
        finally:
            in_flight -= 1
            in_flight_bytes -= request.state.held_bytes
            metrics.IN_FLIGHT.set(in_flight)

    def hold(request: Request, size: int) -> bool:
        """Count ``size`` more bytes of the request's body as held until it ends; False if over the budget."""
        nonlocal in_flight_bytes
        if in_flight_bytes + size > g.config.max_in_flight_bytes:
            return False
        in_flight_bytes += size
        request.state.held_bytes += size
        return True

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
            now = dt.datetime.now(dt.UTC)
            next_month = dt.datetime.combine(month_start(month_start(now.date()) + dt.timedelta(days=32)), dt.time(),
                                             dt.UTC)
            return reject(429, "quota_exceeded", "the account's monthly quota is used up", request_id,
                          (next_month - now).total_seconds())
        try:
            funded = await bounded("balance lookup", g.meter.has_credit(principal))
        except Unchecked:
            return reject(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if not funded:
            return reject(402, "insufficient_credit", "the account's credit is used up", request_id)
        body, limit = bytearray(), g.body_limit()
        try:
            async with asyncio.timeout(g.config.body_timeout_s):
                async for chunk in request.stream():
                    body += chunk
                    if len(body) > limit:
                        return reject(413, "too_large", f"the body exceeds {limit} bytes", request_id)
                    if not hold(request, len(chunk)):
                        return reject(503, "busy", "the gateway is at capacity", request_id, 1)
        except TimeoutError:
            return reject(408, "timeout", f"the body took longer than {g.config.body_timeout_s:g} s", request_id)
        try:
            screened = screen(g, bytes(body))
        except Rejected as r:
            return reject(r.status, r.kind, r.message, request_id, r.retry_after, model=r.model)
        most = screened.most_tokens() * screened.model.price  # the most the answer may cost
        try:
            funded = await bounded("balance lookup", g.meter.reserve(principal, most))
        except Unchecked:
            return reject(503, "unavailable", "accounts cannot be checked right now", request_id, 5)
        if not funded:
            return reject(402, "insufficient_credit", "the account's credit is used up", request_id)
        try:
            try:
                checked = await made_canonical(g, screened)
            except Rejected as r:
                return reject(r.status, r.kind, r.message, request_id, r.retry_after, model=r.model)
            if checked.image_tokens and not hold(request, len(checked.body)):  # the request with its canonical images
                return reject(503, "busy", "the gateway is at capacity", request_id, 1, model=checked.parsed.model)
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
            usage = {"input_tokens": outcome.input_tokens}
            if checked.image_tokens:
                usage["image_tokens"] = checked.image_tokens
            response = JSONResponse({"id": request_id, "model": model, "answers": outcome.answers, "usage": usage,
                                     "receipt": outcome.receipt}, headers={"X-Request-Id": request_id})
            g.stats.record(model, "answered", g.clock() - started)
            paid = g.meter.paid(principal)  # before its cost is charged
            g.meter.record(principal, model, answered=True, input_tokens=outcome.input_tokens, price=checked.price)
            g.sampler.offer(outcome)
            g.served.record(outcome.route, outcome.input_tokens, outcome.seconds, paid=paid)
            metrics.REQUESTS.labels(model, "answered").inc()
            return response
        g.stats.record(model, "busy" if outcome == "busy" else "failed")
        g.meter.record(principal, model, answered=False)
        status, kind, message, retry = NO_ANSWER[outcome]
        return reject(status, kind, message.format(model=model), request_id, retry, model=model)

    return app
