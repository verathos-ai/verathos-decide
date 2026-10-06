"""Signed receipts: the miner's commitment to an answer.

Every answer carries a receipt signed by the miner hotkey. It binds the exact request and the exact answers, so a
later reference recomputation that disagrees is attributable and cannot be denied.
"""

from __future__ import annotations

from typing import Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.protocol import signing
from verathos.protocol.canonical import digest

RECEIPT_DOMAIN = "verathos/receipt/v1"


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
        return signing.verify(self.receipt.miner_hotkey, RECEIPT_DOMAIN, self.receipt.model_dump(mode="json"), self.signature)

    def binds(self, request: dict[str, Any], answers: dict[str, Any]) -> bool:
        """Whether this receipt commits to exactly this request and these answers."""
        return self.receipt.request_sha256 == digest(request) and self.receipt.answers_sha256 == digest(answers)
