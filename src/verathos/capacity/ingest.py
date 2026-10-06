"""The auditor's routes: artifact ingest for miners and signed verdicts for validators.

Arrival time is taken when a request arrives, before its body is read; parsing and signature checks run off the
event loop, so neither a slow chain read in the auditor nor a large upload delays other miners' receipts. A proof
upload names its slot in headers, signed by the slot's hotkey, and is refused before its body is read unless that
slot is waiting for a proof and has no other upload in progress.
"""

from __future__ import annotations

import json

from bittensor_wallet import Keypair
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from verathos.capacity import core
from verathos.capacity.auditor import Auditor, Response
from verathos.capacity.verdicts import VERDICTS_PATH, signed_verdicts

MAX_RECEIPT_BYTES = 64 * 1024


def create_ingest_app(auditor: Auditor, keypair: Keypair) -> FastAPI:
    app = FastAPI(title="verathos capacity audit", docs_url=None, redoc_url=None, openapi_url=None)
    add_capacity_routes(app, auditor, keypair)
    return app


def _json(result: Response) -> JSONResponse:
    return JSONResponse(result.body, status_code=result.status)


def add_capacity_routes(app: FastAPI, auditor: Auditor, keypair: Keypair) -> None:
    async def read(request: Request, limit: int) -> bytes | None:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            return None
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                return None
        return bytes(body)

    def process(kind: str, body: bytes, received_at: float) -> Response:
        try:
            artifact = json.loads(body)
        except ValueError:
            return Response(400, {"error": "invalid json"})
        return auditor.receive(kind, artifact, received_at)

    async def ingest(request: Request, kind: str, limit: int, received_at: float) -> JSONResponse:
        body = await read(request, limit)
        if body is None:
            return JSONResponse({"error": "payload too large", "max_bytes": limit}, status_code=413)
        return _json(await run_in_threadpool(process, kind, body, received_at))

    @app.get("/capacity/v1/health")
    async def health() -> dict:
        return {"status": "ok", "protocol_version": core.PROTOCOL_VERSION, "mode": auditor.policy.mode}

    @app.get(VERDICTS_PATH)
    async def verdicts(epoch: int = Query(ge=0)) -> JSONResponse:
        current = await run_in_threadpool(auditor.current_epoch)
        if epoch > current:
            return JSONResponse({"error": f"epoch {epoch} has not started (current {current})"}, status_code=400)
        gated = await run_in_threadpool(auditor.gated, epoch)
        return JSONResponse(signed_verdicts(keypair, auditor.netuid, epoch, gated))

    @app.post(core.RECEIPT_PATH)
    async def receipt(request: Request) -> JSONResponse:
        received_at = auditor.clock()
        return await ingest(request, "receipt", MAX_RECEIPT_BYTES, received_at)

    @app.post(core.PROOF_PATH)
    async def proof(request: Request) -> JSONResponse:
        received_at = auditor.clock()
        audit_id, slot_id = request.headers.get(core.HDR_AUDIT_ID), request.headers.get(core.HDR_SLOT_ID)
        refused = await run_in_threadpool(auditor.begin_upload, audit_id, slot_id,
                                          request.headers.get(core.HDR_UPLOAD_SIGNATURE))
        if refused is not None:
            return _json(refused)
        try:
            return await ingest(request, "proof", auditor.policy.max_proof_payload_bytes, received_at)
        finally:
            auditor.end_upload(audit_id, slot_id)
