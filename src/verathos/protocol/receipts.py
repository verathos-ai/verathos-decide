"""Signed receipts: the miner's commitment to an answer.

Every answer carries a receipt signed by the miner hotkey. It binds the exact request and the exact answers, so a
later reference recomputation that disagrees is attributable and cannot be denied.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.protocol import signing
from verathos.protocol.api import MAX_RESPONSE_BYTES
from verathos.protocol.canonical import digest

RECEIPT_DOMAIN = "verathos/receipt/v1"
MAX_JSON_DEPTH = 32  # a response nested deeper is not a valid answer


class Receipt(BaseModel):
    version: Literal[1] = 1
    netuid: int
    mechid: int
    miner_hotkey: str
    endpoint_id: str
    model_id: str
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    caller_hotkey: str
    request_nonce: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    received_ms: int
    responded_ms: int
    input_tokens: int = 0


class SignedReceipt(BaseModel):
    receipt: Receipt
    signature: str

    @classmethod
    def create(cls, receipt: Receipt, keypair: Keypair) -> SignedReceipt:
        if keypair.ss58_address != receipt.miner_hotkey:
            raise ValueError("keypair does not match receipt.miner_hotkey")
        return cls(receipt=receipt, signature=signing.sign(keypair, RECEIPT_DOMAIN, receipt.model_dump(mode="json")))

    def signature_valid(self) -> bool:
        return signing.verify(self.receipt.miner_hotkey, RECEIPT_DOMAIN, self.receipt.model_dump(mode="json"),
                              self.signature)

    def binds(self, request: dict[str, Any], answers: dict[str, Any], *, request_sha256: str | None = None) -> bool:
        """Whether this receipt commits to exactly this request and these answers (False for non-JSON values);
        ``request_sha256`` is the request's digest if already known."""
        try:
            return (self.receipt.request_sha256 == (request_sha256 or digest(request))
                    and self.receipt.answers_sha256 == digest(answers))
        except (ValueError, TypeError):
            return False


def parse_response(text: str) -> dict[str, Any] | None:
    """A response body as a JSON object, or None if it is not one, or too large or too deeply nested to be an
    answer."""
    if len(text) > MAX_RESPONSE_BYTES:
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) and not too_deep(value, MAX_JSON_DEPTH) else None


def too_deep(value: Any, limit: int) -> bool:
    """Whether containers nest deeper than ``limit`` (iterative: a hostile document can nest deeper than the
    interpreter's stack)."""
    stack = [(value, 1)]
    while stack:
        node, depth = stack.pop()
        children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else None
        if children is None:
            continue
        if depth > limit:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False


def signed_answers(response: dict[str, Any], request: dict[str, Any], *, request_id: str, hotkey: str,
                   endpoint_id: str, model_id: str, callers: Collection[str], netuid: int, mechid: int,
                   request_sha256: str | None = None) -> Any:
    """The answers of a response if its receipt is the miner's signature over exactly this request and these answers,
    issued to one of ``callers`` for the request with nonce ``request_id``; None otherwise. ``request_sha256`` is the
    request's digest if already known."""
    try:
        receipt = SignedReceipt.model_validate(response["receipt"])
        answers = response["answers"]
    except (KeyError, TypeError, ValueError):
        return None
    r = receipt.receipt
    bound = (receipt.signature_valid() and receipt.binds(request, answers, request_sha256=request_sha256)
             and r.miner_hotkey == hotkey
             and r.endpoint_id == endpoint_id and r.model_id == model_id and r.caller_hotkey in callers
             and r.request_nonce == request_id and r.netuid == netuid and r.mechid == mechid)
    return answers if bound else None
