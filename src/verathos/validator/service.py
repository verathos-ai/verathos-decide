"""The validator's HTTP service on its axon.

The subnet owner's validator serves catalogs by digest (``GET /v1/catalog/{digest}``) so miners and validators can
fetch a newly committed catalog; as the throughput auditor it serves its signed verdicts
(``GET /v1/throughput/{epoch}``), and to the subnet's gateways the routes and the intake for sampled exchanges
(``protocol.gateway``). It serves each hotkey's status (``GET /v1/status/{hotkey}``, see ``validator.status``) and all
of them at once (``GET /v1/network``). Other validators run no service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Collection

import bittensor as bt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from verathos.catalog import CATALOG_PATH, CatalogStore
from verathos.protocol.gateway import (
    MAX_SAMPLES_BYTES,
    MAX_SAMPLES_PER_POST,
    ROUTES_PATH,
    SAMPLES_PATH,
    SERVED_PATH,
    ServedReport,
)
from verathos.protocol.requests import HDR_HOTKEY, NonceCache, RequestRejected, verify_request
from verathos.protocol.status import NETWORK_PATH, STATUS_PATH, HotkeyStatus
from verathos.subnet.chain import serve_axon
from verathos.throughput.auditor import Auditor
from verathos.throughput.feed import VERDICTS_PATH
from verathos.validator.status import StatusBoard

log = logging.getLogger(__name__)
INTAKE_BODY_S = 10.0
INTAKE_CONCURRENCY = 4


class _DropInvalidRequests(logging.Filter):
    """The public port receives scanner traffic; uvicorn's warning for every malformed request is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid HTTP request" not in record.getMessage()


def add_status_route(app: FastAPI, board: StatusBoard, auditor: Auditor | None) -> None:
    @app.get(STATUS_PATH)
    def get_status(hotkey: str):
        status = board.get(hotkey) or HotkeyStatus(hotkey=hotkey)
        if auditor is not None:
            epoch = auditor.follower.head // auditor.policy.epoch_blocks
            status = status.model_copy(update={"tests": auditor.outcomes(hotkey, epoch)})
        return status.model_dump()


def add_network_route(app: FastAPI, board: StatusBoard) -> None:
    @app.get(NETWORK_PATH)
    def get_network():
        return JSONResponse(board.network().model_dump(), headers={"Cache-Control": "public, max-age=30"})


def add_verdicts_route(app: FastAPI, auditor: Auditor) -> None:
    @app.get(VERDICTS_PATH)
    def get_verdicts(epoch: int):
        signed = auditor.verdicts(epoch)
        if signed is None:
            return JSONResponse({"error": "no verdicts for this epoch (not started, or its tests are not settled)"},
                                status_code=404)
        return signed.model_dump(mode="json")


def add_gateway_routes(app: FastAPI, auditor: Auditor, *, probation: Callable[[], Collection[str]],
                       gateways: Callable[[], Collection[str]]) -> None:
    """The routes for the subnet's gateways, and the intake of their sampled exchanges and served counts (signed by
    a catalog gateway, with a declared length within the limit, read within ``INTAKE_BODY_S``, at most
    ``INTAKE_CONCURRENCY`` at once)."""
    nonces, slots = NonceCache(), asyncio.Semaphore(INTAKE_CONCURRENCY)

    @app.get(ROUTES_PATH)
    def get_routes():
        signed = auditor.routes(probation)
        if signed is None:
            return JSONResponse({"error": "no routes yet: no verdicts"}, status_code=503)
        return signed.model_dump(mode="json")

    async def intake(request: Request):
        """-> (caller hotkey, body) or (None, error response)."""
        if request.headers.get(HDR_HOTKEY, "") not in gateways():
            return None, JSONResponse({"error": "caller is not a gateway of the catalog"}, status_code=403)
        declared = request.headers.get("content-length", "")
        if not declared.isdigit() or int(declared) > MAX_SAMPLES_BYTES:
            return None, JSONResponse({"error": f"a content-length up to {MAX_SAMPLES_BYTES} is required"},
                                      status_code=413)
        body = bytearray()
        try:
            async with asyncio.timeout(INTAKE_BODY_S):
                async for chunk in request.stream():
                    body += chunk
                    if len(body) > int(declared):
                        return None, JSONResponse({"error": "body longer than declared"}, status_code=413)
        except TimeoutError:
            return None, JSONResponse({"error": "body too slow"}, status_code=408)
        try:
            caller = verify_request(dict(request.headers), "POST", request.url.path, bytes(body), auditor.hotkey,
                                    nonces)
        except RequestRejected as e:
            return None, JSONResponse({"error": str(e)}, status_code=401)
        return caller.hotkey, bytes(body)

    @app.post(SAMPLES_PATH)
    async def post_samples(request: Request):
        if slots.locked():
            return JSONResponse({"error": "busy"}, status_code=503)
        async with slots:
            gateway, body = await intake(request)
            if gateway is None:
                return body
            try:
                samples = json.loads(body)["samples"]
                if not isinstance(samples, list) or len(samples) > MAX_SAMPLES_PER_POST:
                    raise ValueError(f"samples must be a list of at most {MAX_SAMPLES_PER_POST}")
            except (ValueError, KeyError, TypeError) as e:
                return JSONResponse({"error": f"invalid samples: {e}"}, status_code=422)
            return {"kept": await asyncio.to_thread(auditor.accept_samples, gateway, samples)}

    @app.post(SERVED_PATH)
    async def post_served(request: Request):
        if slots.locked():
            return JSONResponse({"error": "busy"}, status_code=503)
        async with slots:
            gateway, body = await intake(request)
            if gateway is None:
                return body
            try:
                report = ServedReport.model_validate_json(body)
            except ValidationError as e:
                return JSONResponse({"error": f"invalid report: {e.error_count()} errors"}, status_code=422)
            return {"taken": await asyncio.to_thread(auditor.accept_served, report.counts, report.report_id)}


def start_service(*, st: bt.Subtensor, wallet: bt.Wallet, netuid: int, public_host: str, port: int,
                  store: CatalogStore | None, auditor: Auditor | None, board: StatusBoard,
                  probation: Callable[[], Collection[str]] = frozenset,
                  gateways: Callable[[], Collection[str]] = frozenset) -> None:
    app = FastAPI(title="verathos validator", docs_url=None, redoc_url=None, openapi_url=None)
    add_status_route(app, board, auditor)
    add_network_route(app, board)

    if store is not None:
        @app.get(CATALOG_PATH)
        async def get_catalog(digest: str):
            path = store.path(digest)
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or not path.exists():
                return JSONResponse({"error": "unknown catalog"}, status_code=404)
            return FileResponse(path, media_type="application/json")
    if auditor is not None:
        add_verdicts_route(app, auditor)
        add_gateway_routes(app, auditor, probation=probation, gateways=gateways)
    logging.getLogger("uvicorn.error").addFilter(_DropInvalidRequests())
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="validator-service", daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"the validator service could not start on port {port}")
        time.sleep(0.1)
    serve_axon(st, wallet, netuid, public_host, port)
    log.info("validator service on port %d (%s)", port, ", ".join(
        role for role, on in (("catalog", store is not None), ("throughput verdicts", auditor is not None)) if on))
