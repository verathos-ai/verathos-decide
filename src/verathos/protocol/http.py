"""Size- and time-bounded JSON over HTTP, for responses from other neurons."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx


class ResponseTooLarge(ValueError):
    pass


def _parse(chunks: list[bytes]) -> Any:
    try:
        return json.loads(b"".join(chunks))
    except RecursionError:
        raise ValueError("JSON nested too deeply") from None


async def request_json(client: httpx.AsyncClient, method: str, url: str, *, limit: int, timeout: float,
                       **kwargs: Any) -> Any:
    """The JSON body of a 2xx response, read up to ``limit`` bytes, all within ``timeout`` seconds in total.

    Raises httpx.HTTPError, asyncio.TimeoutError, ResponseTooLarge or ValueError (invalid JSON).
    """

    async def run() -> Any:
        async with client.stream(method, url, timeout=timeout, **kwargs) as response:
            response.raise_for_status()
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > limit:
                    raise ResponseTooLarge(f"response exceeds {limit} bytes")
                chunks.append(chunk)
        return _parse(chunks)

    return await asyncio.wait_for(run(), timeout)


def get_json(url: str, *, limit: int, timeout: float) -> Any:
    """Synchronous GET of a JSON body of at most ``limit`` bytes within ``timeout`` seconds in total. Raises
    httpx.HTTPError, ResponseTooLarge or ValueError."""
    deadline = time.monotonic() + timeout
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                raise httpx.ReadTimeout("response took too long")
            size += len(chunk)
            if size > limit:
                raise ResponseTooLarge(f"response exceeds {limit} bytes")
            chunks.append(chunk)
    return _parse(chunks)
