"""The auditor's signed verdicts, one document per scoring epoch: ``GET /v1/throughput/{epoch}``.

For scoring epoch ``E`` the verdicts list the credited capacity of every endpoint in the latest confirmed test of
epochs ``E - credit_ttl_epochs .. E - 1``, its median latency there, whether it has qualified, and the evidence of a
wrong answer for every hotkey the auditor proved one against in those epochs. Evidence from a test is the request and
the miner's response as received (answers and signed receipt), so any validator can check it against its own
reference. Evidence from a sampled user request carries only the miner's signed receipt: user requests are not
published, so validators check that the miner answered the gateway and take the auditor's word for the wrongness,
as they take it for capacity.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import httpx
from bittensor_wallet import Keypair
from pydantic import BaseModel, ConfigDict, Field

from verathos.miner.engine import Engine, EngineError
from verathos.protocol import signing
from verathos.protocol.api import Distribution, distributions
from verathos.protocol.http import request_json
from verathos.validator.compare import RequestComparison, Tolerance, compare_request

VERDICTS_PATH = "/v1/throughput/{epoch}"
VERDICTS_DOMAIN = "verathos/throughput-verdicts/v1"
MAX_VERDICTS_BYTES = 32 << 20  # at most one piece of evidence per hotkey
MALFORMED = "malformed answers"


class Credit(BaseModel):
    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    hotkey: str
    endpoint_id: str
    model_id: str
    credited_rps: float = Field(ge=0)
    latency_ms: float = Field(ge=0)  # median latency of its counted test requests
    test_epoch: int
    qualified: bool
    drift: bool = False  # its test answers drifted beyond the model's statistical tolerance: no credit
    # What the gateways' users got from it in the epoch before the verdicts' (see protocol.gateway.ServedCount):
    user_offers: int = Field(0, ge=0)  # times the gateways considered it for a user request
    user_refusals: int = Field(0, ge=0)  # of those it was not ready (busy, failing, slow), plus requests it refused
    user_answers: int = Field(0, ge=0)  # answers with a measured duration
    user_latency_ms: float = Field(0.0, ge=0)  # their median


GATEWAY_EVIDENCE = "gateway"  # evidence from a sampled gateway answer: test_id "gateway-<request nonce>"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    hotkey: str
    endpoint_id: str
    model_id: str
    test_id: str  # the test, or "gateway-<request nonce>" for an answer the gateway sampled
    reason: str
    request_id: str  # the receipt's nonce: the test request id, or the gateway's request nonce
    request: dict[str, Any]  # the test request; empty for a user request
    response: str  # the response as received (answers and signed receipt); for a user request, {"receipt": ...}

    @property
    def from_gateway(self) -> bool:
        return self.test_id.startswith(GATEWAY_EVIDENCE + "-")


class Verdicts(BaseModel):
    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    netuid: int
    auditor: str
    epoch: int
    credits: list[Credit]
    invalid: list[Evidence]
    demand: dict[str, float] = {}  # model -> the users' input tokens/s through the gateways (EMA over epochs)


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


def readable(answers: Any) -> dict[str, Distribution] | None:
    """The answers' distributions, or None if they cannot be read."""
    try:
        return distributions(answers)
    except Exception:  # whatever a miner signed, unreadable answers are wrong answers
        return None


TOKENS_SLACK = 0.02  # input tokens a miner may sign beyond the reference's count: this share of it ...
TOKENS_SLACK_ABS = 2  # ... plus this many


async def recompute(request: dict[str, Any], got: dict[str, Distribution], reference: Engine,
                    tolerance: Tolerance) -> RequestComparison:
    """Compare readable answers with the reference's own answers to the request. Raises ReferenceUnavailable."""
    try:
        response = await reference.answer(request)
        expected = distributions(response["answers"])
        tokens = (response.get("usage") or {}).get("input_tokens")
        tokens = int(tokens) if tokens is not None else None
    except (EngineError, KeyError, ValueError, TypeError) as exc:
        raise ReferenceUnavailable(str(exc)) from None
    return dataclasses.replace(compare_request(expected, got, tolerance), reference_input_tokens=tokens)


def claimed_tokens(response: dict[str, Any] | None) -> int | None:
    """The input tokens a parsed response's receipt signs, if any."""
    try:
        return int(response["receipt"]["receipt"]["input_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


def tokens_fault(claimed: int | None, comparison: RequestComparison | None) -> str | None:
    """Why the signed input tokens are wrong: more than the reference counts (they set the users' demand per
    model, and with it the models' emission shares). None if they are not, or cannot be told."""
    if claimed is None or comparison is None or comparison.reference_input_tokens is None:
        return None
    if claimed > comparison.reference_input_tokens * (1 + TOKENS_SLACK) + TOKENS_SLACK_ABS:
        return f"input tokens misreported: signed {claimed}, the reference counts {comparison.reference_input_tokens}"
    return None


def fault_of(comparison: RequestComparison, tolerance: Tolerance) -> str | None:
    """Why a compared answer fails the hard check, or None."""
    if not comparison.hard_fail(tolerance):
        return None
    return comparison.structural_error or (f"answers differ from the reference: max difference "
                                           f"{comparison.max_abs:.3f}, {comparison.confident_flips} confident "
                                           "answers changed")


async def answer_fault(request: dict[str, Any], answers: Any, reference: Engine | None,
                       tolerance: Tolerance, claimed: int | None = None) -> str | None:
    """Why ``answers`` are wrong for ``request``: unreadable, or (with a reference engine) outside the model's hard
    tolerance, or signed with more input tokens (``claimed``) than the reference counts. None if they pass, or if
    there is no reference engine. Raises ReferenceUnavailable."""
    got = readable(answers)
    if got is None:
        return MALFORMED
    if reference is None:
        return None
    comparison = await recompute(request, got, reference, tolerance)
    return fault_of(comparison, tolerance) or tokens_fault(claimed, comparison)
