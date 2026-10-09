"""TLS for in-process services: each host presents a certificate of its own, as a TLS server would.

Tests reach miner and validator apps over in-memory transports (``httpx.ASGITransport``), which carry no TLS.
``Presenting`` stands in for one: requests must be https, and must accept the host's certificate (its pin, or any
certificate); the response carries the certificate in its ``network_stream`` extension, as a TLS connection's does,
so ``tls.peer_pin`` reads it. Certificates are real (``tls.identity``'s kind), one per name. ``serving`` runs an app
over real TLS on a loopback port.
"""

from __future__ import annotations

import contextlib
import functools
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from verathos.protocol import tls


@functools.cache
def certificate(name: str) -> bytes:
    """A self-signed certificate (DER) for ``name``, the same within a test run."""
    return tls.certificate_for(ec.generate_private_key(ec.SECP256R1())).public_bytes(serialization.Encoding.DER)


def pin(name: str) -> str:
    return tls.pin_of(certificate(name))


class _Peer:
    def __init__(self, der: bytes) -> None:
        self._der = der

    def getpeercert(self, binary_form: bool = False) -> bytes:
        return self._der


class _Stream:
    def __init__(self, der: bytes) -> None:
        self._peer = _Peer(der)

    def get_extra_info(self, info: str) -> object:
        return self._peer if info == "ssl_object" else None


def der(identity: tls.Identity) -> bytes:
    """The certificate of a service's identity (``tls.identity``), DER."""
    return x509.load_pem_x509_certificate(identity.cert_path.read_bytes()).public_bytes(serialization.Encoding.DER)


class Presenting(httpx.AsyncBaseTransport):
    """Serves each host's requests from ``apps`` (host -> ASGI app, or a transport) and presents the host's
    certificate (``certificates[host]``, DER, or else ``certificate(host)``), refusing what a pinned client
    refuses."""

    def __init__(self, apps: dict[str, object], certificates: dict[str, bytes] | None = None) -> None:
        self._t = {host: app if isinstance(app, httpx.AsyncBaseTransport) else httpx.ASGITransport(app=app)
                   for host, app in apps.items()}
        self._certificates = certificates or {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        accepted = tls.mode_of(request)  # https, and a pin or ANY, as the real client requires
        host = request.url.host
        if host not in self._t:
            raise httpx.ConnectError(f"no route to {host}", request=request)
        der = self._certificates.get(host) or certificate(host)
        if accepted not in (tls.ANY, tls.pin_of(der)):
            raise httpx.ConnectError(f"certificate pin {tls.pin_of(der)} is not the pinned {accepted}",
                                     request=request)
        response = await self._t[host].handle_async_request(request)
        response.extensions = {**response.extensions, "network_stream": _Stream(der)}
        return response


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serving(app: object, identity: tls.Identity, port: int | None = None) -> Iterator[str]:
    """``app`` served over TLS with ``identity`` on a loopback port (``port``, else a free one), as a miner or
    validator serves; -> its base URL."""
    port = port or free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           **tls.server_options(identity)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "the server did not start"
        time.sleep(0.02)
    try:
        yield f"https://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)
