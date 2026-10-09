"""Decision engines behind a miner endpoint.

An engine takes a SystemOne request body and returns a SystemOne response body. Production engines run as
separate processes (one per GPU or Mac) and are reached over plain HTTP on the machine's loopback, or over https
(``engine_tls``); the Kev server (``python -m kev.serve``) is one such engine.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from verathos.protocol import tls
from verathos.protocol.api import SystemOneResponse


class EngineError(Exception):
    pass


class Engine(Protocol):
    async def answer(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def info(self) -> Any: ...


def engine_tls(url: str, pin: str | None) -> str | None:
    """What a connection to an engine at ``url`` accepts: None for plain http on this machine's loopback, else (https)
    its ``pin``, or a public CA's certificate for its hostname (``tls.CA``). Raises ValueError for plain http off
    loopback, an https engine at an IP address without a pin, or a pin for an http engine."""
    if url.startswith("http://") and tls.loopback_url(url):
        if pin:
            raise ValueError(f"{url}: a TLS pin is for an https engine")
        return None
    tls.https_url(url, what="engine url (plain http only on this machine's loopback)")
    if pin is None and tls.authenticated(url) == tls.ANY:
        raise ValueError(f"{url}: an https engine at an IP address needs its TLS pin (no public CA certifies an "
                         "address)")
    return tls.check_pin(pin) if pin is not None else tls.CA


class HttpEngine:
    """An engine process speaking ``POST /v1/systemone``: on this machine over plain http, or over https accepting
    ``tls_pin`` (or, without one, a public CA's certificate) (``engine_tls``)."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 120.0,
                 max_connections: int = 256, tls_pin: str | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._url = self._base + "/v1/systemone"
        self._headers = {"Authorization": f"Bearer {api_key or 'local'}"}
        accepts = engine_tls(self._base, tls_pin)
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        if accepts is None:
            self._client = httpx.AsyncClient(timeout=timeout_s, limits=limits, trust_env=False)
            self._extensions: dict[str, str] = {}
        else:
            self._client = tls.client(timeout=timeout_s, limits=limits)
            self._extensions = tls.use(accepts)

    async def info(self) -> Any:
        """The engine's ``GET /v1/models`` document, or None if it serves none."""
        try:
            r = await self._client.get(self._base + "/v1/models", headers=self._headers, extensions=self._extensions)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def reachable(self) -> bool:
        """Whether the engine answers HTTP at all (any reply: it is up)."""
        try:
            await self._client.get(self._base + "/v1/models", headers=self._headers, extensions=self._extensions)
        except httpx.HTTPError:
            return False
        return True

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(self._url, json=request, headers=self._headers, extensions=self._extensions)
        except httpx.HTTPError as e:
            raise EngineError(f"engine unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"engine returned {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            SystemOneResponse.model_validate(data)
        except (ValueError, ValidationError) as e:
            raise EngineError(f"engine returned an invalid response: {e}") from e
        return data

    async def close(self) -> None:
        await self._client.aclose()
