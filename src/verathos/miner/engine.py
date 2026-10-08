"""Decision engines behind a miner endpoint.

An engine takes a SystemOne request body and returns a SystemOne response body. Production engines run as
separate processes (one per GPU or Mac) and are reached over local HTTP; the Kev server (``python -m kev.serve``)
is one such engine.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from verathos.protocol.api import SystemOneResponse


class EngineError(Exception):
    pass


class Engine(Protocol):
    async def answer(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def info(self) -> Any: ...


class HttpEngine:
    """A local engine process speaking ``POST /v1/systemone``."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 120.0,
                 max_connections: int = 256) -> None:
        self._base = base_url.rstrip("/")
        self._url = self._base + "/v1/systemone"
        self._headers = {"Authorization": f"Bearer {api_key or 'local'}"}
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
        self._client = httpx.AsyncClient(timeout=timeout_s, limits=limits)

    async def info(self) -> Any:
        """The engine's ``GET /v1/models`` document, or None if it serves none."""
        try:
            r = await self._client.get(self._base + "/v1/models", headers=self._headers)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(self._url, json=request, headers=self._headers)
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
