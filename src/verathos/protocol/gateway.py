"""What the subnet's gateway exchanges with the auditor (the subnet owner's validator).

    GET  /v1/routes            the endpoints the gateway may send user traffic to, signed by the auditor
    POST /v1/gateway/samples   sampled gateway exchanges for the auditor to check, signed by a catalog gateway
    POST /v1/gateway/served    what each endpoint answered the gateway, per second, signed by a catalog gateway: its
                               answers, input tokens, how often it was considered for a user request and how often
                               it was not ready or refused, and how long its answers took

A route is an endpoint listed in an accepted manifest that has qualified and is credited in the auditor's current
verdicts, whose hotkey has no proven wrong answer there and is not on probation at the owner's validator. A sample is
a request the gateway sent and the response it received, picked after the answer arrived; the miner's receipt binds
the answers to the request and names the gateway as caller, so a wrong answer is proof against the miner. Miners
serve the gateway before the auditor's test traffic, so the auditor counts what an endpoint answered the gateway
inside a test's measured window as delivered too. What users get from an endpoint outside the tests (refusals and
latency) is scored too: tests are known shortly ahead and their requests are recognisable, so they alone cannot show
it.
"""

from __future__ import annotations

import math
from typing import Any

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, model_validator

from verathos.protocol import signing
from verathos.protocol.api import MAX_RESPONSE_BYTES

ROUTES_PATH = "/v1/routes"
ROUTES_DOMAIN = "verathos/routes/v1"
SAMPLES_PATH = "/v1/gateway/samples"
SERVED_PATH = "/v1/gateway/served"
MAX_SAMPLES_PER_POST = 64
MAX_SAMPLES_BYTES = 8 << 20
MAX_SERVED_COUNTS = 20_000  # entries per post ...
MAX_SERVED_BYTES = 8 << 20  # ... and bytes
MAX_REQUEST_BYTES = 1 << 20  # a gateway request's body, at most


class Route(BaseModel):
    hotkey: str
    endpoint_id: str
    model_id: str
    url: str  # the endpoint's base URL: requests go to url + "/v1/systemone"
    credited_rps: float = Field(gt=0)  # proven capacity in requests/s of the model's standard mix
    latency_ms: float = Field(ge=0)  # median latency in its throughput test


class Routes(BaseModel):
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
    ``MAX_SAMPLES_BYTES``, each judged on its own."""

    hotkey: str
    endpoint_id: str
    model_id: str
    request: dict[str, Any]
    response: str = Field(max_length=MAX_RESPONSE_BYTES)  # the response body as received: answers and receipt


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
    answered: int = Field(0, ge=0, le=1_000_000)  # user requests sent in [second, second + 1) that it answered
    input_tokens: int = Field(0, ge=0)  # of those requests, as the miner signed them (at most one per request byte)
    offered: int = Field(0, ge=0, le=10_000_000)  # times the router considered it for a user request (drew it)
    refused: int = Field(0, ge=0, le=10_000_000)  # of those, it was not ready (busy, open breaker, slow), plus
    # requests it answered busy, refused or failed
    latency: dict[int, int] = Field(default_factory=dict)  # its answers' durations: bucket -> answers

    @model_validator(mode="after")
    def _consistent(self) -> ServedCount:
        if not (self.answered or self.offered or self.refused):
            raise ValueError("an empty count")
        if any(not 0 <= b < LATENCY_BUCKETS or n < 1 for b, n in self.latency.items()):
            raise ValueError("latency buckets out of range")
        if sum(self.latency.values()) > self.answered:
            raise ValueError("more latencies than answers")
        if self.input_tokens > self.answered * MAX_REQUEST_BYTES:
            raise ValueError("more input tokens than the requests' bytes")
        return self


class ServedReport(BaseModel):
    report_id: str = Field("", max_length=64)  # the auditor takes a report once, even when it is posted again
    counts: list[ServedCount] = Field(min_length=1, max_length=MAX_SERVED_COUNTS)
