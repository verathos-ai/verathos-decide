"""HTTP ingest for capacity-audit artifacts, served by the auditor on its axon."""

from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from verathos.capacity import core
from verathos.capacity.auditor import Auditor

RECEIPT_PATH = "/capacity/v1/receipt"
PROOF_PATH = "/capacity/v1/proof"
MAX_RECEIPT_BYTES = 64 * 1024


def create_ingest_app(auditor: Auditor) -> FastAPI:
    app = FastAPI(title="verathos capacity audit", docs_url=None, redoc_url=None, openapi_url=None)

    async def ingest(request: Request, kind: str, limit: int) -> JSONResponse:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return JSONResponse({"error": "payload too large", "max_bytes": limit}, status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                return JSONResponse({"error": "payload too large", "max_bytes": limit}, status_code=413)
        received_at = auditor.clock()
        try:
            artifact = json.loads(body)
        except ValueError:
            return JSONResponse({"error": "invalid json"}, status_code=400)
        result = auditor.receive(kind, artifact, received_at)
        return JSONResponse(result.body, status_code=result.status)

    @app.get("/capacity/v1/health")
    async def health() -> dict:
        return {"status": "ok", "protocol_version": core.PROTOCOL_VERSION, "mode": auditor.policy.mode}

    @app.post(RECEIPT_PATH)
    async def receipt(request: Request) -> JSONResponse:
        return await ingest(request, "receipt", MAX_RECEIPT_BYTES)

    @app.post(PROOF_PATH)
    async def proof(request: Request) -> JSONResponse:
        return await ingest(request, "proof", auditor.policy.max_proof_payload_bytes)

    return app
