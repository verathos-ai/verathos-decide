"""Miner HTTP service: one public port per machine for its endpoints, and on the primary (the axon) the manifest.

    GET  /v1/manifest                      the hotkey's signed manifest (public; the primary only)
    GET  /v1/endpoints                     this machine's endpoints as the manifest lists them, signed (public)
    POST /e/{endpoint_id}/v1/systemone     signed caller request, or auditor test traffic -> answers + signed receipt
    POST /e/{endpoint_id}/v1/identity      signed caller challenge -> hotkey signature proving endpoint control
    GET  /health

Every machine of a hotkey runs this service with the same callers: validators with a permit, gateways, the auditor's
test sessions. Test traffic carries the auditor's session for this endpoint and test instead of a per-request
signature (see ``throughput.policy``); its receipts name the auditor as caller and the request id as nonce.

Callers other than gateways (validators) need only their canaries: one request at once per endpoint, at most the size
of the model's largest standard request (``caller_request_bytes``), and while test traffic arrives their requests wait
behind it (``admission``). More would crowd out a test.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from bittensor_wallet import Keypair
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from verathos.miner.admission import BEHIND_TESTS, REGULAR, TEST, Admission
from verathos.miner.config import MinerConfig
from verathos.miner.engine import Engine, EngineError
from verathos.protocol import signing
from verathos.protocol.api import STATE_CHARS_PER_TOKEN, SystemOneRequest, known_fields, prompt_chars, well_formed
from verathos.protocol.canonical import canonical_json, digest
from verathos.protocol.manifest import ENDPOINTS_PATH, SignedHostEndpoints, SignedManifest
from verathos.protocol.receipts import Receipt, SignedReceipt
from verathos.protocol.requests import HDR_HOTKEY, NonceCache, RequestRejected, verify_request
from verathos.throughput.policy import HDR_SESSION, SessionGate
from verathos.throughput.workload import Workload
from verathos.validator import canaries

log = logging.getLogger(__name__)
IDENTITY_DOMAIN = "verathos/identity/v1"
MAX_REQUEST_BYTES = 1 << 20
BODY_TIMEOUT_S = 10.0
CALLER_CONCURRENCY = 1


@dataclass(frozen=True)
class TestCaller:
    hotkey: str  # the auditor, or this miner for its own measurement
    nonce: str  # the request id
    not_after_ms: int


class IdentityChallenge(BaseModel):
    challenge: str = Field(pattern=r"^[0-9a-f]{32,128}$")


def identity_payload(endpoint_id: str, challenge: str, caller: str) -> dict[str, str]:
    return {"endpoint_id": endpoint_id, "challenge": challenge, "caller": caller}


def caller_request_bytes(model_id: str, workload: Workload) -> int:
    """The largest request (canonical JSON) a caller other than a gateway may send an endpoint of the model: the
    largest of the model's request mix, or of the validators' canaries if that is larger."""
    return max(workload.max_request_bytes(model_id), canaries.max_request_bytes(model_id))


def create_app(
    *,
    config: MinerConfig,
    keypair: Keypair,
    manifest: Callable[[], SignedManifest | None],
    engines: dict[str, Engine],
    authorized: Callable[[str], bool],
    auditors: Callable[[], Collection[str]],
    host_endpoints: Callable[[], SignedHostEndpoints | None] = lambda: None,
    admissions: dict[str, Admission] | None = None,
    serving: Callable[[], Collection[str]] | None = None,
    max_state_tokens: Callable[[str], int | None] = lambda model_id: None,
    max_caller_request: Callable[[str], int | None] = lambda model_id: None,
    on_test_traffic: Callable[[], None] = lambda: None,
    testing: Callable[[], bool] = lambda: False,
    gateways: Callable[[], Collection[str]] | None = None,
    catalog_digest: Callable[[], str] | None = None,
) -> FastAPI:
    """``manifest`` gives the hotkey's manifest this machine serves (None on a worker), ``host_endpoints`` this
    machine's own endpoints document (None until there is one). ``auditors`` gives the hotkeys whose test sessions are
    accepted (besides the miner's own, for its measurement), and ``on_test_traffic`` is called for each of their
    requests; ``testing`` whether their traffic arrives now; ``admissions`` the engine admission per endpoint
    (default: ``max_concurrency`` slots); ``serving`` the endpoints answered (default: those the manifest lists);
    ``max_state_tokens`` a model's limit (``api.prompt_chars``); ``max_caller_request`` the largest request of a model
    callers other than gateways may send (``caller_request_bytes``); ``gateways`` the callers exempt from the
    per-caller limits (default: the configured ``gateway_hotkeys``); ``catalog_digest`` the catalog receipts name
    (default: the manifest's)."""
    app = FastAPI(title="Verathos miner")
    nonces = NonceCache()
    endpoints = {e.endpoint_id: e for e in config.endpoints}
    admissions = admissions or {e.endpoint_id: Admission(e.max_concurrency) for e in config.endpoints}
    hotkey = keypair.ss58_address
    sessions = SessionGate(config.netuid, hotkey)
    gateways = gateways or (lambda: config.gateway_hotkeys)
    catalog_digest = catalog_digest or (lambda: manifest().manifest.catalog_digest)
    in_flight: dict[tuple[str, str], int] = {}  # (caller, endpoint) -> requests, for callers other than gateways

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": message})

    async def read_body(request: Request) -> bytes | None:
        """The request body, or None if it exceeds MAX_REQUEST_BYTES or takes longer than BODY_TIMEOUT_S."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            return None
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_TIMEOUT_S):
                async for chunk in request.stream():
                    body += chunk
                    if len(body) > MAX_REQUEST_BYTES:
                        return None
        except TimeoutError:
            return None
        return bytes(body)

    async def authenticate(request: Request):
        """-> (caller, body, None) or (None, None, error response). The claimed caller must be permitted before its
        body is read or its signature checked."""
        if not authorized(request.headers.get(HDR_HOTKEY, "")):
            return None, None, error(403, "caller is not a permitted validator or gateway")
        body = await read_body(request)
        if body is None:
            return None, None, error(413, "request too large or too slow")
        try:
            caller = verify_request(dict(request.headers), request.method, request.url.path, body, hotkey, nonces)
        except RequestRejected as e:
            return None, None, error(401, str(e))
        return caller, body, None

    async def authenticate_test(request: Request, endpoint_id: str):
        """-> (the session's signer as caller, with the request id as nonce, body, None) or (None, None, error
        response). The session is checked before the body is read."""
        try:
            session, request_id = sessions.check(request.headers, auditors=auditors(), endpoint_id=endpoint_id,
                                                 now_ms=int(time.time() * 1000))
        except RequestRejected as e:
            return None, None, error(401, str(e))
        if session.auditor != hotkey:
            on_test_traffic()
        body = await read_body(request)
        if body is None:
            return None, None, error(413, "request too large or too slow")
        return TestCaller(session.auditor, request_id, session.not_after_ms), body, None

    def listed(endpoint_id: str) -> bool:
        """Whether the endpoint is served (a catalog change or a failed measurement can withdraw one)."""
        if serving is not None:
            return endpoint_id in serving()
        return any(e.endpoint_id == endpoint_id for e in manifest().manifest.endpoints)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"hotkey": hotkey, "endpoints": list(endpoints)}

    @app.get("/v1/manifest")
    async def get_manifest():
        signed = manifest()
        if signed is None:
            return error(404, f"this machine serves its endpoints for the hotkey's primary: {ENDPOINTS_PATH}")
        return signed.model_dump(mode="json")

    @app.get(ENDPOINTS_PATH)
    async def get_endpoints():
        signed = host_endpoints()
        if signed is None:
            return error(404, "no endpoint measured yet")
        return signed.model_dump(mode="json")

    @app.post("/e/{endpoint_id}/v1/identity")
    async def identity(endpoint_id: str, request: Request):
        if endpoint_id not in endpoints or not listed(endpoint_id):
            return error(404, "unknown endpoint")
        caller, body, rejected = await authenticate(request)
        if rejected:
            return rejected
        try:
            challenge = IdentityChallenge.model_validate_json(body).challenge
        except ValidationError:
            return error(400, "invalid challenge")
        payload = identity_payload(endpoint_id, challenge, caller.hotkey)
        return {"hotkey": hotkey, **payload, "signature": signing.sign(keypair, IDENTITY_DOMAIN, payload)}

    @app.post("/e/{endpoint_id}/v1/systemone")
    async def systemone(endpoint_id: str, request: Request):
        endpoint = endpoints.get(endpoint_id)
        if endpoint is None or not listed(endpoint_id):
            return error(404, "unknown endpoint")
        received_ms = int(time.time() * 1000)
        test = HDR_SESSION in request.headers
        caller, body, rejected = await (authenticate_test(request, endpoint_id) if test else authenticate(request))
        if rejected:
            return rejected
        try:
            raw = json.loads(body)
            parsed = SystemOneRequest.model_validate(raw)
        except (ValueError, ValidationError) as e:
            return error(422, f"invalid request: {e}")
        if parsed.model != endpoint.model_id:
            return error(422, f"endpoint serves {endpoint.model_id}, not {parsed.model}")
        limit = max_state_tokens(endpoint.model_id)
        if limit is not None and prompt_chars(parsed) > STATE_CHARS_PER_TOKEN * limit:
            return error(413, "state, instructions and criteria longer than the model accepts")
        capped = (caller.hotkey, endpoint_id) if not test and caller.hotkey not in gateways() else None
        if capped is not None:
            largest = max_caller_request(endpoint.model_id)
            if largest is not None and len(canonical_json(raw)) > largest:
                return error(413, "request larger than the model's standard requests")
            if in_flight.get(capped, 0) >= CALLER_CONCURRENCY:
                return error(429, "too many concurrent requests from this caller")
            in_flight[capped] = in_flight.get(capped, 0) + 1
        cls = TEST if test else BEHIND_TESTS if capped is not None and testing() else REGULAR
        try:
            return await answer(endpoint, endpoint_id, request, raw, parsed, caller, cls, received_ms)
        finally:
            if capped is not None:
                in_flight[capped] -= 1
                if not in_flight[capped]:
                    del in_flight[capped]

    async def answer(endpoint, endpoint_id: str, request: Request, raw: dict[str, Any], parsed: SystemOneRequest,
                     caller, cls: int, received_ms: int):
        admission = admissions[endpoint_id]
        # A test request is useful only until its session ends.
        wait_s = (caller.not_after_ms - time.time() * 1000) / 1000 if cls == TEST else None
        if not await admission.acquire(cls, wait_s=wait_s):
            return error(503, "endpoint at capacity")
        try:
            if await request.is_disconnected():
                return error(499, "caller gone")
            response = await engines[endpoint_id].answer(raw)
        except EngineError as e:
            log.warning("engine %s failed: %s", endpoint_id, e)
            return error(502, "engine failure")
        finally:
            admission.release()
        answers = known_fields(parsed, response.get("answers"))  # an engine's extra fields are not signed
        if not well_formed(parsed, answers):
            log.warning("engine %s returned malformed answers; not signed", endpoint_id)
            return error(502, "engine returned malformed answers")
        receipt = Receipt(
            netuid=config.netuid, mechid=config.mechid, miner_hotkey=hotkey, endpoint_id=endpoint_id,
            model_id=endpoint.model_id, catalog_digest=catalog_digest(), caller_hotkey=caller.hotkey,
            request_nonce=caller.nonce, request_sha256=digest(raw), answers_sha256=digest(answers),
            received_ms=received_ms, responded_ms=int(time.time() * 1000),
            input_tokens=int((response.get("usage") or {}).get("input_tokens", 0)),
        )
        return {"model": endpoint.model_id, "answers": answers, "usage": response.get("usage"),
                "receipt": SignedReceipt.create(receipt, keypair).model_dump(mode="json")}

    return app

