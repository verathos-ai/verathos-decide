"""Miner HTTP service: one public port (the axon) for the manifest and every endpoint.

    GET  /v1/manifest                      signed manifest (public)
    POST /e/{endpoint_id}/v1/systemone     signed caller request, or auditor test traffic -> answers + signed receipt
    POST /e/{endpoint_id}/v1/identity      signed caller challenge -> hotkey signature proving endpoint control
    GET  /health

Test traffic carries the auditor's session for this endpoint and test instead of a per-request signature (see
``throughput.policy``); its receipts name the auditor as caller and the request id as nonce.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from bittensor_wallet import Keypair
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from verathos.miner.admission import Admission
from verathos.miner.config import MinerConfig
from verathos.miner.engine import Engine, EngineError
from verathos.protocol import signing
from verathos.protocol.api import SystemOneRequest
from verathos.protocol.canonical import digest
from verathos.protocol.manifest import SignedManifest
from verathos.protocol.receipts import Receipt, SignedReceipt
from verathos.protocol.requests import HDR_HOTKEY, NonceCache, RequestRejected, VerifiedCaller, verify_request
from verathos.throughput.policy import HDR_SESSION, SessionGate

log = logging.getLogger(__name__)
IDENTITY_DOMAIN = "verathos/identity/v1"
MAX_REQUEST_BYTES = 1 << 20


class IdentityChallenge(BaseModel):
    challenge: str = Field(pattern=r"^[0-9a-f]{32,128}$")


def identity_payload(endpoint_id: str, challenge: str, caller: str) -> dict[str, str]:
    return {"endpoint_id": endpoint_id, "challenge": challenge, "caller": caller}


def create_app(
    *,
    config: MinerConfig,
    keypair: Keypair,
    manifest: Callable[[], SignedManifest],
    engines: dict[str, Engine],
    authorized: Callable[[str], bool],
    auditor: Callable[[], str | None],
    admissions: dict[str, Admission] | None = None,
) -> FastAPI:
    """``auditor`` gives the hotkey whose test sessions are accepted; ``admissions`` the engine admission per
    endpoint (default: ``max_concurrency`` slots)."""
    app = FastAPI(title="Verathos miner")
    nonces = NonceCache()
    endpoints = {e.endpoint_id: e for e in config.endpoints}
    admissions = admissions or {e.endpoint_id: Admission(e.max_concurrency) for e in config.endpoints}
    hotkey = keypair.ss58_address
    sessions = SessionGate(config.netuid, hotkey)

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": message})

    async def read_body(request: Request) -> bytes | None:
        """The request body, or None if it exceeds MAX_REQUEST_BYTES."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_REQUEST_BYTES:
            return None
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_REQUEST_BYTES:
                return None
        return bytes(body)

    async def authenticate(request: Request):
        """-> (caller, body, None) or (None, None, error response). The claimed caller must be permitted before its
        body is read or its signature checked."""
        if not authorized(request.headers.get(HDR_HOTKEY, "")):
            return None, None, error(403, "caller is not a permitted validator or gateway")
        body = await read_body(request)
        if body is None:
            return None, None, error(413, "request too large")
        try:
            caller = verify_request(dict(request.headers), request.method, request.url.path, body, hotkey, nonces)
        except RequestRejected as e:
            return None, None, error(401, str(e))
        return caller, body, None

    async def authenticate_test(request: Request, endpoint_id: str):
        """-> (auditor as caller with the request id as nonce, body, None) or (None, None, error response). The
        session is checked before the body is read."""
        try:
            session, request_id = sessions.check(request.headers, auditor=auditor(), endpoint_id=endpoint_id,
                                                 now_ms=int(time.time() * 1000))
        except RequestRejected as e:
            return None, None, error(401, str(e))
        body = await read_body(request)
        if body is None:
            return None, None, error(413, "request too large")
        return VerifiedCaller(session.auditor, request_id), body, None

    def listed(endpoint_id: str) -> bool:
        """Whether the current manifest lists the endpoint (a catalog change can withdraw one)."""
        return any(e.endpoint_id == endpoint_id for e in manifest().manifest.endpoints)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"hotkey": hotkey, "endpoints": list(endpoints)}

    @app.get("/v1/manifest")
    async def get_manifest() -> dict[str, Any]:
        return manifest().model_dump(mode="json")

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
        admission = admissions[endpoint_id]
        if not await admission.acquire(test=test):
            return error(503, "endpoint at capacity")
        try:
            response = await engines[endpoint_id].answer(raw)
        except EngineError as e:
            log.warning("engine %s failed: %s", endpoint_id, e)
            return error(502, "engine failure")
        finally:
            admission.release()
        answers = response.get("answers")
        if not isinstance(answers, dict):
            return error(502, "engine returned no answers")
        receipt = Receipt(
            netuid=config.netuid, mechid=config.mechid, miner_hotkey=hotkey, endpoint_id=endpoint_id,
            model_id=endpoint.model_id, catalog_digest=manifest().manifest.catalog_digest, caller_hotkey=caller.hotkey,
            request_nonce=caller.nonce, request_sha256=digest(raw), answers_sha256=digest(answers),
            received_ms=received_ms, responded_ms=int(time.time() * 1000),
            input_tokens=int((response.get("usage") or {}).get("input_tokens", 0)),
        )
        return {"model": endpoint.model_id, "answers": answers, "usage": response.get("usage"),
                "receipt": SignedReceipt.create(receipt, keypair).model_dump(mode="json")}

    return app
