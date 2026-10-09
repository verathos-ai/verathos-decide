"""The front of a reference machine: its engines, which serve on this machine's loopback, served to its callers over
TLS.

    POST /reference/{model}/v1/systemone   the model's float32 reference
    GET  /reference/{model}/v1/models
    POST /screen/{model}/v1/systemone      the model's bfloat16 screen
    GET  /screen/{model}/v1/models

Only the callers' addresses reach it: the listening socket drops every other address at its first packet, before TCP
answers it (``miner.sockfilter``, which needs no privilege, so it works in a container), and the app refuses one that
got through all the same (403). Every request carries the callers' bearer key (``Authorization: Bearer <key>``), or
gets 401. Both are checked before anything of the request is read. The front presents its own certificate, kept in
``state_dir``; callers accept it by its pin. A body is at most what the engines take; the engine's answer goes back as
it is, and the key never reaches the engine.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Callable, Mapping

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from verathos.engine.server import MAX_BODY_BYTES
from verathos.gateway.logs import Throttle
from verathos.miner import sockfilter
from verathos.miner.app import SourceGate
from verathos.protocol import tls
from verathos.protocol.http import raise_open_file_limit
from verathos.reference.config import ReferenceConfig, caller_address

log = logging.getLogger(__name__)
BODY_S = 60.0  # a request's body arrives within it
CONNECT_S = 5.0
ANSWER_S = 900.0  # an engine answers within it; the caller's own timeout is usually shorter
LOOPBACK = frozenset({"127.0.0.1", "::1"})  # this machine, as the socket filter passes it


def allows(callers: frozenset[str]) -> Callable[[str | None], bool]:
    """Whether a connection from an address is served: a caller's, or this machine's loopback."""
    def check(address: str | None) -> bool:
        try:
            normal = caller_address(address or "")
        except ValueError:
            return False
        return normal in callers or normal in LOOPBACK

    return check


class KeyGate:
    """Refuses (401) a request without the callers' key before anything of it is read; each address's refusals are
    logged at most once a minute."""

    def __init__(self, app, key: str) -> None:
        self.app, self.expected, self.said = app, f"Bearer {key}".encode(), Throttle()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            given = next((v for k, v in scope["headers"] if k == b"authorization"), b"")
            if not hmac.compare_digest(given, self.expected):
                client = (scope.get("client") or ("?",))[0]
                if times := self.said(client):
                    log.warning("refused %d request(s) from %s: the key is missing or wrong", times, client)
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"error": "the key is missing or wrong"}'})
                return
        await self.app(scope, receive, send)


def create_app(engines: Mapping[tuple[str, str], str], key: str, allowed: Callable[[str | None], bool],
               upstream: httpx.AsyncClient | None = None) -> FastAPI:
    """The front: ``engines`` maps (kind, model) to the engine's base URL on this machine; ``allowed`` decides which
    source addresses are served (``allows``); ``upstream`` reaches the engines (default: a plain client, loopback
    only)."""
    app = FastAPI(title="verathos reference", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(KeyGate, key=key)
    app.add_middleware(SourceGate, allows=allowed)  # added last: runs first
    client = upstream or httpx.AsyncClient(timeout=httpx.Timeout(ANSWER_S, connect=CONNECT_S), trust_env=False,
                                           limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def forward(base: str, path: str, body: bytes | None = None) -> Response:
        try:
            if body is None:
                r = await client.get(base + path)
            else:
                r = await client.post(base + path, content=body, headers={"content-type": "application/json"})
        except httpx.TimeoutException:
            return error(504, "the engine did not answer in time")
        except httpx.HTTPError:
            return error(502, "the engine does not answer (starting, or down)")
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def read_body(request: Request) -> bytes | None:
        """The body, or None if it exceeds ``MAX_BODY_BYTES`` or takes longer than ``BODY_S``."""
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return None
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_S):
                async for chunk in request.stream():
                    body += chunk
                    if len(body) > MAX_BODY_BYTES:
                        return None
        except TimeoutError:
            return None
        return bytes(body)

    @app.get("/{kind}/{model}/v1/models")
    async def models(kind: str, model: str):
        base = engines.get((kind, model))
        return error(404, f"this machine serves no {kind} of {model}") if base is None else await forward(
            base, "/v1/models")

    @app.post("/{kind}/{model}/v1/systemone")
    async def systemone(kind: str, model: str, request: Request):
        base = engines.get((kind, model))
        if base is None:
            return error(404, f"this machine serves no {kind} of {model}")
        body = await read_body(request)
        if body is None:
            return error(413, f"a request is at most {MAX_BODY_BYTES} bytes, sent within {BODY_S:.0f} s")
        return await forward(base, "/v1/systemone", body)

    return app


def bind(config: ReferenceConfig, identity: tls.Identity, key: str) -> tuple[uvicorn.Server, object]:
    """The front's server and its listening socket, filtered to the callers' addresses: ``server.run(sockets=[it])``
    serves. A socket the kernel does not filter is logged; the app refuses other addresses on it all the same."""
    engines = {(kind, m): config.engines[m][kind].rstrip("/") for kind, m in config.served()}
    app = create_app(engines, key, allows(frozenset(config.allow)))
    # Connections beyond the descriptors left are refused, so idle connections cannot starve the rest.
    limit = max(256, raise_open_file_limit() - 256)
    server = uvicorn.Server(uvicorn.Config(app, host=config.listen_host, port=config.port, log_level="warning",
                                           access_log=False, limit_concurrency=limit, **tls.server_options(identity)))
    try:
        listening = server.config.bind_socket()
    except SystemExit:
        raise SystemExit(f"the front could not listen on port {config.port}") from None
    try:
        sockfilter.attach(listening, config.allow)
    except (ValueError, OSError) as exc:
        log.warning("the listening socket is not filtered (%s): other addresses get a TLS handshake and then 403", exc)
    return server, listening
