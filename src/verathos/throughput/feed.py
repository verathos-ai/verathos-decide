"""The auditor's signed verdicts, one document per scoring epoch: ``GET /v1/throughput/{epoch}``.

For scoring epoch ``E`` the verdicts list the credited capacity of every endpoint in the latest confirmed test of
epochs ``E - credit_ttl_epochs .. E - 1``, its median latency there, whether it has qualified, and the evidence of a
wrong answer for every hotkey the auditor proved one against in those tests. Each piece of evidence is the request
and the miner's response as received (answers and signed receipt), so any validator can check it against its own
reference.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

import httpx
from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.miner.engine import Engine, EngineError
from verathos.protocol import signing
from verathos.protocol.api import MAX_RESPONSE_BYTES, Distribution, distributions
from verathos.protocol.http import request_json
from verathos.protocol.receipts import SignedReceipt
from verathos.validator.compare import RequestComparison, Tolerance, compare_request

VERDICTS_PATH = "/v1/throughput/{epoch}"
VERDICTS_DOMAIN = "verathos/throughput-verdicts/v1"
MAX_JSON_DEPTH = 32  # nor one nested deeper
MAX_VERDICTS_BYTES = 32 << 20  # at most one piece of evidence per hotkey
MALFORMED = "malformed answers"


class Credit(BaseModel):
    hotkey: str
    endpoint_id: str
    model_id: str
    credited_rps: float = Field(ge=0)
    latency_ms: float = Field(ge=0)  # median latency of its counted test requests
    test_epoch: int
    qualified: bool
    drift: bool = False  # its test answers drifted beyond the model's statistical tolerance: no credit


GATEWAY_EVIDENCE = "gateway"  # the test_id of evidence from a sampled gateway answer


class Evidence(BaseModel):
    hotkey: str
    endpoint_id: str
    model_id: str
    test_id: str  # the test, or GATEWAY_EVIDENCE for an answer the gateway sampled
    reason: str
    request_id: str  # the receipt's nonce: the test request id, or the gateway's request nonce
    request: dict[str, Any]
    response: str  # the response body as received: the answers and the signed receipt


class Verdicts(BaseModel):
    netuid: int
    auditor: str
    epoch: int
    credits: list[Credit]
    invalid: list[Evidence]


class SignedVerdicts(BaseModel):
    verdicts: Verdicts
    signature: str

    @classmethod
    def create(cls, verdicts: Verdicts, keypair: Keypair) -> SignedVerdicts:
        return cls(verdicts=verdicts,
                   signature=signing.sign(keypair, VERDICTS_DOMAIN, verdicts.model_dump(mode="json")))


class VerdictsUnavailable(Exception):
    pass


class ReferenceUnavailable(Exception):
    """The reference engine could not answer, so an answer could not be checked."""


async def fetch(client: httpx.AsyncClient, auditor_url: str | None, *, auditor_hotkey: str, netuid: int,
                epoch: int) -> Verdicts:
    """The auditor's verdicts for ``epoch``, signature checked. Raises VerdictsUnavailable (also while the epoch's
    tests are still being settled)."""
    if not auditor_url:
        raise VerdictsUnavailable("auditor axon unknown")
    try:
        document = await request_json(client, "GET", auditor_url.rstrip("/") + VERDICTS_PATH.format(epoch=epoch),
                                      limit=MAX_VERDICTS_BYTES, timeout=30.0)
        signed = SignedVerdicts.model_validate(document)
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        raise VerdictsUnavailable(f"{type(exc).__name__}: {exc}") from None
    v = signed.verdicts
    if not signing.verify(auditor_hotkey, VERDICTS_DOMAIN, v.model_dump(mode="json"), signed.signature):
        raise VerdictsUnavailable("signature does not verify against the auditor hotkey")
    if (v.auditor, v.netuid, v.epoch) != (auditor_hotkey, netuid, epoch):
        raise VerdictsUnavailable("verdicts are for another auditor, netuid or epoch")
    return v


def parse_response(text: str) -> dict[str, Any] | None:
    """A response body as a JSON object, or None if it is not one, or too large or too deeply nested to be an
    answer."""
    if len(text) > MAX_RESPONSE_BYTES:
        return None
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) and _depth(value) <= MAX_JSON_DEPTH else None


def _depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max(map(_depth, value.values()), default=0)
    if isinstance(value, list):
        return 1 + max(map(_depth, value), default=0)
    return 0


def signed_answers(response: dict[str, Any], request: dict[str, Any], *, request_id: str, hotkey: str,
                   endpoint_id: str, model_id: str, callers: Collection[str], netuid: int, mechid: int) -> Any:
    """The answers of a response if its receipt is the miner's signature over exactly this request and these answers,
    issued to one of ``callers`` for the request with nonce ``request_id``; None otherwise."""
    try:
        receipt = SignedReceipt.model_validate(response["receipt"])
        answers = response["answers"]
    except (KeyError, TypeError, ValueError):
        return None
    r = receipt.receipt
    bound = (receipt.signature_valid() and receipt.binds(request, answers) and r.miner_hotkey == hotkey
             and r.endpoint_id == endpoint_id and r.model_id == model_id and r.caller_hotkey in callers
             and r.request_nonce == request_id and r.netuid == netuid and r.mechid == mechid)
    return answers if bound else None


def readable(answers: Any) -> dict[str, Distribution] | None:
    """The answers' distributions, or None if they cannot be read."""
    try:
        return distributions(answers)
    except Exception:  # whatever a miner signed, unreadable answers are wrong answers
        return None


async def recompute(request: dict[str, Any], got: dict[str, Distribution], reference: Engine,
                    tolerance: Tolerance) -> RequestComparison:
    """Compare readable answers with the reference's own answers to the request. Raises ReferenceUnavailable."""
    try:
        expected = distributions((await reference.answer(request))["answers"])
    except (EngineError, KeyError, ValueError) as exc:
        raise ReferenceUnavailable(str(exc)) from None
    return compare_request(expected, got, tolerance)


def fault_of(comparison: RequestComparison, tolerance: Tolerance) -> str | None:
    """Why a compared answer fails the hard check, or None."""
    if not comparison.hard_fail(tolerance):
        return None
    return comparison.structural_error or (f"answers differ from the reference: max difference "
                                           f"{comparison.max_abs:.3f}, {comparison.confident_flips} confident "
                                           "answers changed")


async def answer_fault(request: dict[str, Any], answers: Any, reference: Engine | None,
                       tolerance: Tolerance) -> str | None:
    """Why ``answers`` are wrong for ``request``: unreadable, or (with a reference engine) outside the model's hard
    tolerance. None if they pass, or if there is no reference engine. Raises ReferenceUnavailable."""
    got = readable(answers)
    if got is None:
        return MALFORMED
    if reference is None:
        return None
    return fault_of(await recompute(request, got, reference, tolerance), tolerance)
