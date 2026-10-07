"""What the subnet's gateway exchanges with the auditor (the subnet owner's validator).

    GET  /v1/routes            the endpoints the gateway may send user traffic to, signed by the auditor
    POST /v1/gateway/samples   sampled gateway exchanges for the auditor to check, signed by a catalog gateway
    POST /v1/gateway/served    what each endpoint answered the gateway, per second, signed by a catalog gateway

A route is an endpoint listed in an accepted manifest that has qualified and is credited in the auditor's current
verdicts, whose hotkey has no proven wrong answer there and is not on probation at the owner's validator. A sample is
a request the gateway sent and the response it received, picked after the answer arrived; the miner's receipt binds
the answers to the request and names the gateway as caller, so a wrong answer is proof against the miner. Miners
serve the gateway before the auditor's test traffic, so the auditor counts what an endpoint answered the gateway
inside a test's measured window as delivered too.
"""

from __future__ import annotations

from typing import Any

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.protocol import signing
from verathos.protocol.api import MAX_RESPONSE_BYTES

ROUTES_PATH = "/v1/routes"
ROUTES_DOMAIN = "verathos/routes/v1"
SAMPLES_PATH = "/v1/gateway/samples"
SERVED_PATH = "/v1/gateway/served"
MAX_SAMPLES_PER_POST = 64
MAX_SAMPLES_BYTES = 8 << 20
MAX_SERVED_COUNTS = 50_000


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


class ServedCount(BaseModel):
    hotkey: str
    endpoint_id: str
    second: int  # unix time, whole seconds: answers completed in [second, second + 1)
    answered: int = Field(ge=1)
    input_tokens: int = Field(0, ge=0)  # of those requests, as the miner signed them


class ServedReport(BaseModel):
    counts: list[ServedCount] = Field(min_length=1, max_length=MAX_SERVED_COUNTS)
