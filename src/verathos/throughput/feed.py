"""The auditor's signed verdicts, one document per scoring epoch: ``GET /v1/throughput/{epoch}``.

For scoring epoch ``E`` the verdicts list the credited capacity of every endpoint in the latest confirmed test of
epochs ``E - credit_ttl_epochs .. E - 1``, whether it has qualified, and the evidence of a wrong answer for every
hotkey the auditor proved one against in those tests. Each piece of evidence is the request, the miner's answers and
its signed receipt, so any validator can check it against its own reference.
"""

from __future__ import annotations

from typing import Any

import httpx
from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.miner.engine import Engine, EngineError
from verathos.protocol import signing
from verathos.protocol.api import distributions
from verathos.protocol.http import request_json
from verathos.protocol.receipts import SignedReceipt
from verathos.validator.compare import Tolerance, compare_request

VERDICTS_PATH = "/v1/throughput/{epoch}"
VERDICTS_DOMAIN = "verathos/throughput-verdicts/v1"
MAX_EVIDENCE_BYTES = 64 << 10  # a response larger than this is not a valid answer
MAX_VERDICTS_BYTES = 32 << 20  # at most one piece of evidence per hotkey
MALFORMED = "malformed answers"


class Credit(BaseModel):
    hotkey: str
    endpoint_id: str
    model_id: str
    credited_rps: float = Field(ge=0)
    test_epoch: int
    qualified: bool


class Evidence(BaseModel):
    hotkey: str
    endpoint_id: str
    model_id: str
    test_id: str
    reason: str
    request_id: str
    request: dict[str, Any]
    response: dict[str, Any]  # the answers and the signed receipt, as the miner returned them


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


def signed_answers(response: Any, request: dict[str, Any], *, request_id: str, hotkey: str, endpoint_id: str,
                   model_id: str, auditor: str, netuid: int, mechid: int) -> Any:
    """The answers of a test response if its receipt is the miner's signature over exactly this request and these
    answers, issued for this test request; None otherwise."""
    try:
        receipt = SignedReceipt.model_validate(response["receipt"])
        answers = response["answers"]
    except (KeyError, TypeError, ValueError):
        return None
    r = receipt.receipt
    bound = (receipt.signature_valid() and receipt.binds(request, answers) and r.miner_hotkey == hotkey
             and r.endpoint_id == endpoint_id and r.model_id == model_id and r.caller_hotkey == auditor
             and r.request_nonce == request_id and r.netuid == netuid and r.mechid == mechid)
    return answers if bound else None


async def answer_fault(request: dict[str, Any], answers: Any, reference: Engine | None,
                       tolerance: Tolerance) -> str | None:
    """Why ``answers`` are wrong for ``request``: unreadable, or (with a reference engine) outside the model's hard
    tolerance. None if they pass, or if there is no reference engine. Raises ReferenceUnavailable."""
    try:
        got = distributions(answers)
    except Exception:  # whatever the miner signed, unreadable answers are wrong answers
        return MALFORMED
    if reference is None:
        return None
    try:
        expected = distributions((await reference.answer(request))["answers"])
    except (EngineError, KeyError, ValueError) as exc:
        raise ReferenceUnavailable(str(exc)) from None
    c = compare_request(expected, got, tolerance)
    if not c.hard_fail(tolerance):
        return None
    return c.structural_error or (f"answers differ from the reference: max difference {c.max_abs:.3f}, "
                                  f"{c.confident_flips} confident answers changed")
