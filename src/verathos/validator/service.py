"""The validator's service on its axon, over TLS only.

The subnet owner's validator serves catalogs by digest (``GET /v1/catalog/{digest}``) so miners and validators can
fetch a newly committed catalog; as the throughput auditor it serves its signed verdicts
(``GET /v1/throughput/{epoch}``), and to the subnet's gateways the routes and the intake for sampled exchanges
(``protocol.gateway``). It serves each hotkey's status (``GET /v1/status/{hotkey}``, see ``validator.status``) and all
of them at once with the weights it set (``GET /v1/network``, which light validators follow), both signed with its
hotkey (``protocol.status``). A full validator with a ``public_host`` serves these two as well; other validators run no
service. The service presents the validator's certificate (``protocol.tls``, kept in its state directory): the
auditor's pin is in the catalog, another validator's in its chain commitment (``protocol.callers``).

A hotkey's signed status is built at most once per ``STATUS_CACHE_S`` (sooner once the board publishes an epoch or the
auditor records, confirms or voids a test), and may be cached as long; the network's is built once per epoch and
signed again at most as often. A client address (an IPv4 address, or an IPv6 /64) gets a burst of ``STATUS_BURST``
status requests, then one per ``STATUS_EVERY_S``; beyond, HTTP 429 with ``Retry-After``. Miners poll their status
every 2 minutes per machine, so one address holds about 24 machines' polls. Behind a reverse proxy on the same host,
the address is the client's from ``X-Forwarded-For``, which uvicorn takes from loopback peers only.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Collection

import bittensor as bt
import uvicorn
from bittensor_wallet import Keypair
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
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
from verathos.protocol.status import (
    NETWORK_DOMAIN,
    NETWORK_PATH,
    STATUS_DOMAIN,
    STATUS_PATH,
    HotkeyStatus,
    sign_document,
)
from verathos.protocol.tls import Identity, server_options
from verathos.subnet.chain import serve_axon
from verathos.throughput.auditor import Auditor
from verathos.throughput.feed import VERDICTS_PATH
from verathos.validator.status import StatusBoard

log = logging.getLogger(__name__)
INTAKE_BODY_S = 30.0  # a post of samples, at most MAX_SAMPLES_BYTES, arrives within it
INTAKE_CONCURRENCY = 4
STATUS_CACHE_S = 30.0
STATUS_CACHE_ENTRIES = 1024  # hotkeys whose signed status is kept at once (a subnet has 256 UIDs)
STATUS_BURST, STATUS_EVERY_S = 10, 5.0
MAX_ADDRESSES = 4096  # client addresses the status limit tracks at once
MAX_HOTKEY_CHARS = 64  # an SS58 address has 48


class _DropInvalidRequests(logging.Filter):
    """The public port receives scanner traffic; uvicorn's warning for every malformed request is noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid HTTP request" not in record.getMessage()


def address_group(host: str) -> str:
    """The unit requests are limited by: an IPv4 address (also one mapped into IPv6), or an IPv6 /64."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return str(ipaddress.ip_network(f"{ip}/64", strict=False)) if ip.version == 6 else str(ip)


class AddressLimit:
    """Per key, a burst of ``burst`` requests, then one per ``every_s`` (the generic cell rate algorithm). At most
    ``max_keys`` keys are tracked: beyond, the one allowed a request longest ago is forgotten and starts afresh."""

    def __init__(self, burst: int = STATUS_BURST, every_s: float = STATUS_EVERY_S,
                 max_keys: int = MAX_ADDRESSES) -> None:
        self.burst, self.every_s, self.max_keys = burst, every_s, max_keys
        self._tat: OrderedDict[str, float] = OrderedDict()  # key -> theoretical arrival time of its next request
        self._lock = threading.Lock()

    def check(self, key: str, now: float) -> float:
        """0.0 if a request of ``key`` is allowed now (and counts it); otherwise the seconds until it would be."""
        with self._lock:
            tat = max(self._tat.get(key, now), now)
            wait = tat - now - (self.burst - 1) * self.every_s
            if wait > 0:
                return wait
            self._tat[key] = tat + self.every_s
            self._tat.move_to_end(key)
            while len(self._tat) > self.max_keys:
                self._tat.popitem(last=False)
            return 0.0


def add_status_route(app: FastAPI, board: StatusBoard, auditor: Auditor | None, keypair: Keypair, netuid: int, *,
                     clock: Callable[[], float] = time.monotonic) -> None:
    """``GET /v1/status/{hotkey}``: the board's status of the hotkey and its results in the auditor's last tests,
    signed; cached and limited per client address (see the module)."""
    limit, lock = AddressLimit(), threading.Lock()
    cache: OrderedDict[str, tuple[float, tuple[int, int], bytes]] = OrderedDict()  # -> (built, revisions, body)

    @app.get(STATUS_PATH)
    def get_status(hotkey: str, request: Request):
        now = clock()
        wait = limit.check(address_group(request.client.host if request.client else ""), now)
        if wait:
            return JSONResponse({"error": "too many status requests from this address"}, status_code=429,
                                headers={"Retry-After": str(max(1, math.ceil(wait)))})
        if len(hotkey) > MAX_HOTKEY_CHARS:
            return JSONResponse({"error": "not a hotkey"}, status_code=404)
        revisions = (board.revision, auditor.revision if auditor is not None else 0)
        with lock:
            cached = cache.get(hotkey)
        if cached is None or cached[1] != revisions or now - cached[0] >= STATUS_CACHE_S:
            status = board.get(hotkey) or HotkeyStatus(hotkey=hotkey)
            if auditor is not None:
                status = status.model_copy(update={"tests": auditor.outcomes(hotkey)})
            document = sign_document(status.model_dump(mode="json"), keypair, netuid=netuid, domain=STATUS_DOMAIN,
                                     now_ms=int(time.time() * 1000))
            cached = (now, revisions, json.dumps(document, separators=(",", ":")).encode())
            with lock:
                cache[hotkey] = cached
                cache.move_to_end(hotkey)
                while len(cache) > STATUS_CACHE_ENTRIES:
                    cache.popitem(last=False)
        return Response(cached[2], media_type="application/json",
                        headers={"Cache-Control": f"public, max-age={STATUS_CACHE_S:.0f}"})


def add_network_route(app: FastAPI, board: StatusBoard, keypair: Keypair, netuid: int, *,
                      clock: Callable[[], float] = time.monotonic) -> None:
    """``GET /v1/network``: every hotkey's status of the board's epoch and the weights set, built once per epoch and
    signed again at most every ``STATUS_CACHE_S``, so its ``issued_ms`` shows a light validator it is served now."""
    lock = threading.Lock()
    signed: list[tuple[float, int, bytes]] = []  # [(signed at, board revision, signed body)]

    @app.get(NETWORK_PATH)
    def get_network():
        now = clock()
        with lock:
            if not signed or signed[0][1] != board.revision or now - signed[0][0] >= STATUS_CACHE_S:
                revision, body = board.versioned_network()
                document = sign_document(json.loads(body), keypair, netuid=netuid, domain=NETWORK_DOMAIN,
                                         now_ms=int(time.time() * 1000))
                signed[:] = [(now, revision, json.dumps(document, separators=(",", ":")).encode())]
            body = signed[0][2]
        return Response(body, media_type="application/json",
                        headers={"Cache-Control": f"public, max-age={STATUS_CACHE_S:.0f}"})


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


def start_service(*, st: bt.Subtensor, wallet: bt.Wallet, netuid: int, public_host: str, port: int, identity: Identity,
                  store: CatalogStore | None, auditor: Auditor | None, board: StatusBoard,
                  probation: Callable[[], Collection[str]] = frozenset,
                  gateways: Callable[[], Collection[str]] = frozenset) -> None:
    app = FastAPI(title="verathos validator", docs_url=None, redoc_url=None, openapi_url=None)
    add_status_route(app, board, auditor, wallet.hotkey, netuid)
    add_network_route(app, board, wallet.hotkey, netuid)

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
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", access_log=False,
                                           **server_options(identity)))
    thread = threading.Thread(target=server.run, name="validator-service", daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"the validator service could not start on port {port}")
        time.sleep(0.1)
    serve_axon(st, wallet, netuid, public_host, port)
    log.info("validator service on port %d, TLS certificate pin %s (%s)", port, identity.pin, ", ".join(
        role for role, on in (("status and weights", True), ("catalog", store is not None),
                              ("throughput verdicts", auditor is not None)) if on))
