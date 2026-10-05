"""Miner HTTP service: one public port (the axon) for the manifest and every endpoint.

    GET  /v1/manifest                      signed manifest (public)
    POST /e/{endpoint_id}/v1/systemone     signed caller request -> answers + signed receipt
    POST /e/{endpoint_id}/v1/identity      signed caller challenge -> hotkey signature proving endpoint control
    GET  /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from bittensor_wallet import Keypair
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from decide.miner.config import MinerConfig
from decide.miner.engine import Engine, EngineError
from decide.protocol import signing
from decide.protocol.api import SystemOneRequest
from decide.protocol.canonical import digest
from decide.protocol.manifest import SignedManifest
from decide.protocol.receipts import Receipt, SignedReceipt
from decide.protocol.requests import NonceCache, RequestRejected, verify_request

log = logging.getLogger(__name__)
IDENTITY_DOMAIN = "verathos-decide/identity/v1"


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
    catalog_digest: str,
) -> FastAPI:
    app = FastAPI(title="verathos-decide miner")
    nonces = NonceCache()
    endpoints = {e.endpoint_id: e for e in config.endpoints}
    slots = {e.endpoint_id: asyncio.Semaphore(e.max_concurrency) for e in config.endpoints}
    hotkey = keypair.ss58_address

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": message})

    async def caller_of(request: Request, body: bytes):
        try:
            caller = verify_request(dict(request.headers), request.method, request.url.path, body, hotkey, nonces)
        except RequestRejected as e:
            return None, error(401, str(e))
        if not authorized(caller.hotkey):
            return None, error(403, "caller is not a permitted validator or gateway")
        return caller, None

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"hotkey": hotkey, "endpoints": list(endpoints)}

    @app.get("/v1/manifest")
    async def get_manifest() -> dict[str, Any]:
        return manifest().model_dump(mode="json")

    @app.post("/e/{endpoint_id}/v1/identity")
    async def identity(endpoint_id: str, request: Request):
        if endpoint_id not in endpoints:
            return error(404, "unknown endpoint")
        body = await request.body()
        caller, rejected = await caller_of(request, body)
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
        if endpoint is None:
            return error(404, "unknown endpoint")
        received_ms = int(time.time() * 1000)
        body = await request.body()
        caller, rejected = await caller_of(request, body)
        if rejected:
            return rejected
        try:
            raw = json.loads(body)
            parsed = SystemOneRequest.model_validate(raw)
        except (ValueError, ValidationError) as e:
            return error(422, f"invalid request: {e}")
        if parsed.model != endpoint.model_id:
            return error(422, f"endpoint serves {endpoint.model_id}, not {parsed.model}")
        slot = slots[endpoint_id]
        if slot.locked():
            return error(503, "endpoint at capacity")
        async with slot:
            try:
                response = await engines[endpoint_id].answer(raw)
            except EngineError as e:
                log.warning("engine %s failed: %s", endpoint_id, e)
                return error(502, "engine failure")
        answers = response.get("answers")
        if not isinstance(answers, dict):
            return error(502, "engine returned no answers")
        receipt = Receipt(
            netuid=config.netuid, mechid=config.mechid, miner_hotkey=hotkey, endpoint_id=endpoint_id,
            model_id=endpoint.model_id, catalog_digest=catalog_digest, caller_hotkey=caller.hotkey,
            request_nonce=caller.nonce, request_sha256=digest(raw), answers_sha256=digest(answers),
            received_ms=received_ms, responded_ms=int(time.time() * 1000),
            input_tokens=int((response.get("usage") or {}).get("input_tokens", 0)),
        )
        return {"model": endpoint.model_id, "answers": answers, "usage": response.get("usage"),
                "receipt": SignedReceipt.create(receipt, keypair).model_dump(mode="json")}

    return app
