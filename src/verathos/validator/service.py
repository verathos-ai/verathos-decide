"""The validator's HTTP service on its axon.

The subnet owner's validator serves catalogs by digest (``GET /v1/catalog/{digest}``) so miners and validators can
fetch a newly committed catalog; the auditor serves the capacity-audit ingest and verdicts. Both serve each
hotkey's status (``GET /v1/status/{hotkey}``, see ``validator.status``). Other validators run no service.
"""

from __future__ import annotations

import logging
import re
import threading

import bittensor as bt
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from verathos.capacity.auditor import Auditor
from verathos.capacity.ingest import add_capacity_routes
from verathos.catalog import CATALOG_PATH, CatalogStore
from verathos.protocol.status import STATUS_PATH, AuditOutcome, HotkeyStatus
from verathos.subnet.chain import serve_axon
from verathos.validator.status import StatusBoard

log = logging.getLogger(__name__)


class _DropInvalidRequests(logging.Filter):
    """The public port receives scanner traffic; uvicorn's warning for every malformed request is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid HTTP request" not in record.getMessage()


def add_status_route(app: FastAPI, board: StatusBoard, auditor: Auditor | None) -> None:
    @app.get(STATUS_PATH)
    def get_status(hotkey: str):
        status = board.get(hotkey) or HotkeyStatus(hotkey=hotkey)
        audits = [AuditOutcome(**a) for a in auditor.recent_outcomes(hotkey)] if auditor is not None else []
        return status.model_copy(update={"audits": audits}).model_dump()


def start_service(*, st: bt.Subtensor, wallet: bt.Wallet, netuid: int, public_host: str, port: int,
                  store: CatalogStore | None, auditor: Auditor | None, board: StatusBoard) -> None:
    app = FastAPI(title="verathos validator", docs_url=None, redoc_url=None, openapi_url=None)
    add_status_route(app, board, auditor)

    if store is not None:
        @app.get(CATALOG_PATH)
        async def get_catalog(digest: str):
            path = store.path(digest)
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or not path.exists():
                return JSONResponse({"error": "unknown catalog"}, status_code=404)
            return FileResponse(path, media_type="application/json")
    if auditor is not None:
        add_capacity_routes(app, auditor, wallet.hotkey)
    logging.getLogger("uvicorn.error").addFilter(_DropInvalidRequests())
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", access_log=False))
    threading.Thread(target=server.run, name="validator-service", daemon=True).start()
    serve_axon(st, wallet, netuid, public_host, port)
    log.info("validator service on port %d (%s)", port, ", ".join(
        role for role, on in (("catalog", store is not None), ("capacity audit", auditor is not None)) if on))
