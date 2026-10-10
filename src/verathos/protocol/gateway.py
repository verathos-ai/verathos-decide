"""What the subnet's gateway exchanges with the auditor (the subnet owner's validator).

    GET  /v1/routes            the endpoints the gateway may send user traffic to, signed by the auditor
    POST /v1/gateway/samples   sampled gateway exchanges for the auditor to check, signed by a catalog gateway
    POST /v1/gateway/served    what each endpoint answered the gateway, per second, signed by a catalog gateway: its
                               answers, input tokens, how often it was considered for a user request and how often
                               it was not ready or refused, and how long its answers took
    POST /v1/gateway/referrals the referral codes' signed registrations and moves, and the referred accounts' top-ups
                               and refunds as they count, signed by a catalog gateway (``ReferralReport``)

All of it over TLS, accepting the auditor's certificate by the pin the catalog names. A route is an endpoint listed in
an accepted manifest, with the pin of its machine's TLS certificate, that has qualified and is credited in the
auditor's current verdicts, whose hotkey has no proven wrong answer there and is not on probation at the owner's
validator. A sample is a request the gateway sent and the response it received, picked after the answer arrived; the
miner's receipt binds the answers to the request and names the gateway as caller, so a wrong answer is proof against
the miner. Miners serve the gateway before the auditor's test traffic, so the auditor counts what an endpoint answered
the gateway inside a test's measured window as delivered too. What users get from an endpoint outside the tests
(refusals and latency) is scored too: tests are known shortly ahead and their requests are recognisable, so they alone
cannot show it.
"""

from __future__ import annotations

import math
from typing import Any

from bittensor_wallet import Keypair
from pydantic import BaseModel, ConfigDict, Field, model_validator

from verathos.protocol import signing
from verathos.protocol.api import MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES
from verathos.protocol.media import MAX_IMAGE_REQUEST_BYTES, MAX_IMAGE_TOKENS
from verathos.protocol.referral import READABLE_LENGTH, SignedMove, SignedRegistration
from verathos.protocol.tls import PIN_PATTERN

ROUTES_PATH = "/v1/routes"
ROUTES_DOMAIN = "verathos/routes/v1"
SAMPLES_PATH = "/v1/gateway/samples"
SERVED_PATH = "/v1/gateway/served"
REFERRALS_PATH = "/v1/gateway/referrals"
MAX_REFERRAL_ENTRIES = 5_000  # per post ...
MAX_REFERRAL_SIGNED = 1_000  # ... and registrations or moves
MAX_REFERRAL_MICROS = 10**12  # micro-USD one entry counts, either sign
MAX_SAMPLES_PER_POST = 64
# A post of samples, at most: one sample of the largest request an image model may take, with its response, fits.
MAX_SAMPLES_BYTES = MAX_IMAGE_REQUEST_BYTES + (1 << 20)
MAX_SERVED_COUNTS = 20_000  # entries per post ...
MAX_SERVED_BYTES = 8 << 20  # ... and bytes
MAX_INPUT_TOKENS = MAX_REQUEST_BYTES + MAX_IMAGE_TOKENS  # one answer is billed at most: a token per byte of its
# request without images, and its images' tokens


class Route(BaseModel):
    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    hotkey: str
    endpoint_id: str
    model_id: str
    url: str  # the endpoint's base URL: requests go to url + "/v1/systemone"
    tls: str = Field(pattern=PIN_PATTERN)  # the pin of its machine's certificate, as its manifest names it
    credited_rps: float = Field(gt=0)  # proven capacity in requests/s of the model's standard mix
    latency_ms: float = Field(ge=0)  # median latency in its throughput test


class Routes(BaseModel):
    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    netuid: int
    auditor: str
    epoch: int
    issued_ms: int
    catalog_digest: str
    routes: list[Route]


class SignedRoutes(BaseModel):
    routes: Routes
    signature: str

    @classmethod
    def create(cls, routes: Routes, keypair: Keypair) -> SignedRoutes:
        return cls(routes=routes, signature=signing.sign(keypair, ROUTES_DOMAIN, routes.model_dump(mode="json")))

    def verify(self, auditor: str, netuid: int) -> bool:
        r = self.routes
        return (r.auditor, r.netuid) == (auditor, netuid) and signing.verify(
            auditor, ROUTES_DOMAIN, r.model_dump(mode="json"), self.signature)


class GatewaySample(BaseModel):
    """One sampled exchange; a post is ``{"samples": [...]}`` with up to ``MAX_SAMPLES_PER_POST`` of them within
    ``MAX_SAMPLES_BYTES``, each judged on its own.

    ``test_id`` marks the exchange the gateway drew, per endpoint and test, from those it sent the endpoint during the
    test's traffic (``throughput.policy.ScheduledTest.test_id``), posted once the test's sessions end
    (``AuditPolicy.session_s`` after the start block). The auditor counts the endpoint's answers to users in that test
    as delivered only once this exchange is checked."""

    hotkey: str
    endpoint_id: str
    model_id: str
    request: dict[str, Any]
    response: str = Field(max_length=MAX_RESPONSE_BYTES)  # the response body as received: answers and receipt
    test_id: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")


# Latency bucket i holds answers that took [BASE x RATIO^i, BASE x RATIO^(i + 1)) ms; the last one all slower ones.
LATENCY_BASE_MS = 10.0
LATENCY_RATIO = 1.1
LATENCY_BUCKETS = 120


def latency_bucket(seconds: float) -> int:
    ms = max(seconds * 1000, LATENCY_BASE_MS)
    return min(int(math.log(ms / LATENCY_BASE_MS) / math.log(LATENCY_RATIO)), LATENCY_BUCKETS - 1)


def latency_median_ms(buckets: dict[int, int]) -> float | None:
    """The median of a latency histogram, at the geometric middle of the bucket holding it; None if empty."""
    total = sum(buckets.values())
    if total == 0:
        return None
    seen = 0
    for i in sorted(buckets):
        seen += buckets[i]
        if seen >= total / 2:
            return LATENCY_BASE_MS * LATENCY_RATIO ** (i + 0.5)
    raise AssertionError("unreachable")


class ServedCount(BaseModel):
    """One endpoint's second at one gateway process."""

    hotkey: str
    endpoint_id: str
    second: int  # unix time, whole seconds
    attempts: int = Field(0, ge=0, le=1_000_000)  # user requests sent to it in [second, second + 1)
    answered: int = Field(0, ge=0, le=1_000_000)  # of those, the ones it answered
    input_tokens: int = Field(0, ge=0)  # of those requests, as billed (the miner's signed count within bounds)
    # of those, the input tokens of requests their account paid for (bought credit spent on them, its free credit
    # spent first); None: this gateway does not tell paid from unpaid. Only paid usage moves the models' emission
    # shares.
    paid_input_tokens: int | None = Field(None, ge=0)
    offered: int = Field(0, ge=0, le=10_000_000)  # times the router considered it for a user request (drew it)
    refused: int = Field(0, ge=0, le=10_000_000)  # of those, it was not ready (busy, open breaker, slow), plus
    # requests it answered busy, refused or failed
    latency: dict[int, int] = Field(default_factory=dict)  # its answers' durations: bucket -> answers

    @model_validator(mode="after")
    def _consistent(self) -> ServedCount:
        if not (self.attempts or self.answered or self.offered or self.refused):
            raise ValueError("an empty count")
        if any(not 0 <= b < LATENCY_BUCKETS or n < 1 for b, n in self.latency.items()):
            raise ValueError("latency buckets out of range")
        if sum(self.latency.values()) > self.answered:
            raise ValueError("more latencies than answers")
        if self.input_tokens > self.answered * MAX_INPUT_TOKENS:
            raise ValueError("more input tokens than the requests can hold")
        if (self.paid_input_tokens or 0) > self.input_tokens:
            raise ValueError("more paid input tokens than input tokens")
        return self


class ServedReport(BaseModel):
    report_id: str = Field("", max_length=64)  # the auditor takes a report once, even when it is posted again
    counts: list[ServedCount] = Field(min_length=1, max_length=MAX_SERVED_COUNTS)


class ReferralEntry(BaseModel):
    """A referred account's top-up or refund as it counts for its code: a top-up within the account's term and cap,
    a refund (negative) at most what its top-up counted."""

    id: str = Field(pattern=r"^[0-9a-f]{64}$")  # keyed with the gateway's secret: the auditor takes it once
    code: str = Field(min_length=1, max_length=READABLE_LENGTH[1])
    amount_micros: int = Field(ge=-MAX_REFERRAL_MICROS, le=MAX_REFERRAL_MICROS)
    credited_ms: int = Field(ge=0)  # the start of the UTC hour the gateway credited it in

    @model_validator(mode="after")
    def _not_zero(self) -> ReferralEntry:
        if self.amount_micros == 0:
            raise ValueError("an entry counts at least one micro-USD")
        return self


class ReferralReport(BaseModel):
    """What the gateway has not reported yet: codes' registrations and moves (before the entries that name them) and
    entries. Every item is taken once, however often it is posted."""

    codes: list[SignedRegistration] = Field(default_factory=list, max_length=MAX_REFERRAL_SIGNED)
    moves: list[SignedMove] = Field(default_factory=list, max_length=MAX_REFERRAL_SIGNED)
    entries: list[ReferralEntry] = Field(default_factory=list, max_length=MAX_REFERRAL_ENTRIES)
