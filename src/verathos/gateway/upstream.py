"""Signed requests to miners, and the checks on their responses.

The request is signed with the gateway's hotkey for the endpoint's miner (``protocol.requests``). A response counts
only as an uncompressed JSON object within ``MAX_RESPONSE_BYTES`` whose receipt is the miner's signature over this
request and these answers, issued to this gateway for this request's nonce, with well-formed answers. Signed
malformed answers are proof on their own.

A failed attempt says whose fault it is: ``miner`` failures (no connection, an invalid or unsigned response, a refusal
of this gateway) count against the miner at once; ``unclear`` ones (a timeout, an error status) could be caused by
the request itself, and count only if another miner then answers the same request; a ``busy`` miner and a request
the miner ``rejected`` (4xx) count against nobody.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from bittensor_wallet import Keypair

from verathos.protocol.api import MAX_RESPONSE_BYTES, SystemOneRequest, well_formed
from verathos.protocol.gateway import Route
from verathos.protocol.receipts import parse_response, signed_exchange
from verathos.protocol.requests import HDR_NONCE, sign_request


@dataclass(frozen=True)
class Exchange:
    route: Route
    request: dict[str, Any]
    response: str  # the response body as received
    answers: dict[str, Any]
    receipt: dict[str, Any]  # as validated: only what the miner signed
    input_tokens: int  # as the miner signed it
    seconds: float = 0.0  # the attempt's duration, set by the caller


class UpstreamFailure(Exception):
    """An attempt at a miner without a usable answer; ``blame`` is miner, unclear, busy or rejected (see the module
    docstring); ``exchange`` holds signed malformed answers, proof against the miner."""

    def __init__(self, kind: str, blame: str, detail: str = "", *, exchange: Exchange | None = None) -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind, self.blame, self.exchange = kind, blame, exchange


class Upstream:
    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *, netuid: int, mechid: int) -> None:
        self.keypair, self.client, self.netuid, self.mechid = keypair, client, netuid, mechid
        self.hotkey = keypair.ss58_address

    async def call(self, route: Route, request: dict[str, Any], body: bytes, request_sha256: str,
                   parsed: SystemOneRequest, timeout_s: float) -> Exchange:
        """One attempt at ``route``. Raises UpstreamFailure."""
        url = route.url + "/v1/systemone"
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity",
                   **sign_request(self.keypair, "POST", urlsplit(url).path, body, route.hotkey)}
        try:
            async with asyncio.timeout(timeout_s):
                async with self.client.stream("POST", url, content=body, headers=headers) as r:
                    if r.status_code in (429, 503):
                        raise UpstreamFailure("busy", "busy", str(r.status_code))
                    if r.status_code in (401, 403):
                        raise UpstreamFailure("refused", "miner", str(r.status_code))
                    if 400 <= r.status_code < 500:
                        raise UpstreamFailure("rejected", "rejected", str(r.status_code))
                    if r.status_code != 200:
                        raise UpstreamFailure("status", "unclear", str(r.status_code))
                    if r.headers.get("content-encoding", "identity").lower() != "identity":
                        raise UpstreamFailure("encoding", "miner", r.headers["content-encoding"])
                    raw = bytearray()
                    async for chunk in r.aiter_raw():
                        raw += chunk
                        if len(raw) > MAX_RESPONSE_BYTES:
                            raise UpstreamFailure("too_large", "miner")
        except TimeoutError:
            raise UpstreamFailure("timeout", "unclear") from None
        except httpx.HTTPError as exc:
            raise UpstreamFailure("connect", "miner", type(exc).__name__) from None
        text = raw.decode("utf-8", errors="replace")
        response = parse_response(text)
        signed = None if response is None else signed_exchange(
            response, request, request_id=headers[HDR_NONCE], hotkey=route.hotkey, endpoint_id=route.endpoint_id,
            model_id=route.model_id, callers={self.hotkey}, netuid=self.netuid, mechid=self.mechid,
            request_sha256=request_sha256)
        if signed is None:
            raise UpstreamFailure("receipt", "miner", "no receipt binding this request and these answers")
        answers, receipt = signed
        # at most one token per byte of the request; the auditor checks sampled counts against its reference
        tokens = min(max(receipt.receipt.input_tokens, 0), len(body))
        exchange = Exchange(route, request, text, answers, receipt.model_dump(mode="json"), tokens)
        if not well_formed(parsed, answers):
            raise UpstreamFailure("malformed", "miner", "signed malformed answers", exchange=exchange)
        return exchange
