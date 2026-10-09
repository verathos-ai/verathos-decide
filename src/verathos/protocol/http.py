"""Size- and time-bounded JSON over https, for responses from other neurons, and the open-file limit connections need.

Every request names what TLS certificate it accepts (``tls``: a pin, ``tls.ANY`` or ``tls.CA``, see
``protocol.tls``); the clients are ``tls.client`` and ``tls.sync_client``.
"""

from __future__ import annotations

import asyncio
import json
import resource
import time
from collections.abc import Callable
from typing import Any

import httpx

from verathos.protocol import tls as tls_module


class ResponseTooLarge(ValueError):
    pass


def _parse(chunks: list[bytes]) -> Any:
    try:
        return json.loads(b"".join(chunks))
    except RecursionError:
        raise ValueError("JSON nested too deeply") from None


async def request_document(client: httpx.AsyncClient, method: str, url: str, *, limit: int, timeout: float, tls: str,
                           **kwargs: Any) -> tuple[Any, str | None]:
    """The JSON body of a 2xx response, read up to ``limit`` bytes, all within ``timeout`` seconds in total, and the
    pin of the certificate the server presented (None without TLS, as in tests).

    Raises httpx.HTTPError, asyncio.TimeoutError, ResponseTooLarge or ValueError (invalid JSON).
    """

    async def run() -> tuple[Any, str | None]:
        async with client.stream(method, url, timeout=timeout, extensions=tls_module.use(tls), **kwargs) as response:
            response.raise_for_status()
            seen = tls_module.peer_pin(response)
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > limit:
                    raise ResponseTooLarge(f"response exceeds {limit} bytes")
                chunks.append(chunk)
        return _parse(chunks), seen

    return await asyncio.wait_for(run(), timeout)


async def request_json(client: httpx.AsyncClient, method: str, url: str, *, limit: int, timeout: float, tls: str,
                       **kwargs: Any) -> Any:
    """``request_document``'s JSON body alone."""
    return (await request_document(client, method, url, limit=limit, timeout=timeout, tls=tls, **kwargs))[0]


async def post_idempotent(client: httpx.AsyncClient, url: str, body: bytes, headers: Callable[[], dict[str, str]], *,
                          timeout: float, tls: str) -> httpx.Response:
    """POST ``body`` and raise for an error status; once more if the server closed the connection without an answer (a
    kept-alive connection it closed as it was reused), so only for a post the server takes once however often it
    arrives. ``headers`` gives each try's headers (a signed request's nonce is good once)."""
    for retry in (False, True):
        try:
            r = await client.post(url, content=body, headers=headers(), timeout=timeout,
                                  extensions=tls_module.use(tls))
        except httpx.RemoteProtocolError:
            if retry:
                raise
            continue
        r.raise_for_status()
        return r
    raise AssertionError("unreachable")


def get_json(url: str, *, limit: int, timeout: float, tls: str) -> Any:
    """Synchronous GET of a JSON body of at most ``limit`` bytes within ``timeout`` seconds in total, over https
    accepting ``tls`` (redirects followed only to public CAs' certificates). Raises httpx.HTTPError, ResponseTooLarge
    or ValueError."""
    deadline = time.monotonic() + timeout
    with tls_module.sync_client(timeout=timeout, follow_redirects=tls == tls_module.CA) as client, \
            client.stream("GET", url, extensions=tls_module.use(tls)) as response:
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


def raise_open_file_limit() -> int:
    """Raise the soft open-file limit to the hard one (every connection holds a descriptor); returns it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
