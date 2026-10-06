"""One epoch of canary audits.

For every model with a reference engine, the validator computes reference answers for the epoch's canary pool once,
then sends each endpoint its assigned subset as signed requests, checks every receipt, and compares the answers with
the reference. Nothing a miner returns can abort the epoch: every reply is size- and time-bounded and every parse
failure is recorded against that endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from bittensor_wallet import Keypair

from verathos.miner.engine import Engine, EngineError
from verathos.protocol.api import distributions
from verathos.protocol.http import request_json
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import HDR_NONCE, sign_request
from verathos.validator.canaries import assignment, canary_pool
from verathos.validator.compare import EndpointTally, RequestComparison, Tolerance, compare_request
from verathos.validator.discovery import DiscoveredEndpoint, verify_identity

log = logging.getLogger(__name__)
DRAIN_POLL_S = 2.0
MAX_DRAIN_WAIT_S = 600.0
REQUEST_TIMEOUT_S = 60.0
MAX_REPLY_BYTES = 1 << 20


@dataclass
class EndpointResult:
    endpoint: DiscoveredEndpoint
    tolerance: Tolerance
    identity_ok: bool = False
    sent: int = 0
    answered: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tally: EndpointTally = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tally is None:
            self.tally = EndpointTally(self.tolerance)


async def reference_answers(pool: list[dict[str, Any]], engine: Engine) -> list[dict[str, dict[str, float]] | None]:
    """Reference distributions for each canary; None where the reference engine failed."""
    out: list[dict[str, dict[str, float]] | None] = []
    for request in pool:
        try:
            out.append(distributions((await engine.answer(request))["answers"]))
        except (EngineError, KeyError, ValueError) as e:
            log.warning("reference failed for a %s canary: %s", request["model"], e)
            out.append(None)
    return out


def _invalid(reason: str) -> RequestComparison:
    return RequestComparison(1.0, 0.0, 0, 0, reason)


async def _canary(result: EndpointResult, request: dict[str, Any], reference: dict[str, dict[str, float]],
                  keypair: Keypair, client: httpx.AsyncClient, netuid: int, mechid: int) -> None:
    """Send one canary and record its outcome. A request the endpoint declines (no 2xx reply) costs availability;
    a reply whose receipt does not bind it, or whose answers are malformed, fails the request outright."""
    ep = result.endpoint
    path = ep.path_prefix + "/v1/systemone"
    body = json.dumps(request).encode()
    headers = {"Content-Type": "application/json", **sign_request(keypair, "POST", path, body, ep.hotkey)}
    result.sent += 1
    t0 = time.monotonic()
    try:
        data = await request_json(client, "POST", ep.base_url + path, content=body, headers=headers,
                                  limit=MAX_REPLY_BYTES, timeout=REQUEST_TIMEOUT_S)
    except (httpx.HTTPError, TimeoutError, ValueError) as e:
        result.errors.append(f"request failed: {type(e).__name__}")
        return
    elapsed_ms = (time.monotonic() - t0) * 1000
    result.answered += 1
    result.latencies_ms.append(elapsed_ms)
    try:
        receipt = SignedReceipt.model_validate(data["receipt"])
        answers = data["answers"]
        got = distributions(answers)
        r_ = receipt.receipt
        bound = (receipt.signature_valid() and receipt.binds(request, answers) and r_.miner_hotkey == ep.hotkey
                 and r_.endpoint_id == ep.endpoint.endpoint_id and r_.model_id == ep.endpoint.model_id
                 and r_.caller_hotkey == keypair.ss58_address and r_.request_nonce == headers[HDR_NONCE]
                 and r_.netuid == netuid and r_.mechid == mechid)
        comparison = compare_request(reference, got, result.tolerance) if bound else None
    except Exception as e:  # anything a reply can make the parser or comparison raise fails the request
        result.errors.append(f"malformed reply: {type(e).__name__}")
        result.tally.add(_invalid("malformed reply"))
        return
    if comparison is None:
        result.errors.append("invalid receipt")
        comparison = _invalid("invalid receipt")
    result.tally.add(comparison)


async def _audit_endpoint(result: EndpointResult, canaries: list[tuple[dict[str, Any], dict[str, dict[str, float]]]],
                          keypair: Keypair, client: httpx.AsyncClient, spread_s: float, rng: random.Random,
                          drained: Callable[[str, str], bool], netuid: int, mechid: int) -> None:
    ep = result.endpoint
    result.identity_ok = await verify_identity(ep, keypair, client)
    for request, reference in canaries:
        if spread_s:
            await asyncio.sleep(rng.uniform(0, spread_s))
        waited = 0.0
        while drained(ep.hotkey, ep.endpoint.endpoint_id) and waited < MAX_DRAIN_WAIT_S:
            await asyncio.sleep(DRAIN_POLL_S)  # the endpoint is inside a capacity-audit hold
            waited += DRAIN_POLL_S
        await _canary(result, request, reference, keypair, client, netuid, mechid)


async def audit_epoch(
    endpoints: list[DiscoveredEndpoint],
    *,
    epoch: int,
    secret: bytes,
    keypair: Keypair,
    references: dict[str, Engine],
    tolerances: dict[str, Tolerance],
    client: httpx.AsyncClient,
    netuid: int,
    mechid: int,
    pool_size: int = 16,
    per_endpoint: int = 10,
    spread_s: float = 0.0,
    drained: Callable[[str, str], bool] = lambda hotkey, endpoint_id: False,
) -> dict[str, EndpointResult]:
    """Audit every endpoint whose model has a reference engine. Returns results keyed by endpoint key; endpoints
    without a usable canary this epoch (no reference engine, or every reference failed) have no result."""
    pools: dict[str, list[tuple[dict[str, Any], dict[str, dict[str, float]] | None]]] = {}
    for model_id in sorted({e.endpoint.model_id for e in endpoints}):
        if model_id not in references:
            log.info("no reference engine for %s; its endpoints are not audited this epoch", model_id)
            continue
        pool = canary_pool(secret, epoch, model_id, pool_size)
        pools[model_id] = list(zip(pool, await reference_answers(pool, references[model_id])))

    results: dict[str, EndpointResult] = {}
    tasks = []
    rng = random.Random(int.from_bytes(secret[:8], "big") ^ epoch)
    for ep in endpoints:
        pool = pools.get(ep.endpoint.model_id)
        if pool is None:
            continue
        picked = [pool[i] for i in assignment(secret, epoch, ep.key, len(pool), per_endpoint) if pool[i][1] is not None]
        if not picked:
            continue
        res = EndpointResult(ep, tolerances[ep.endpoint.model_id])
        results[ep.key] = res
        tasks.append(_audit_endpoint(res, picked, keypair, client, spread_s, random.Random(rng.random()), drained,
                                     netuid, mechid))
    for res, outcome in zip(results.values(), await asyncio.gather(*tasks, return_exceptions=True)):
        if isinstance(outcome, BaseException):  # an aborted audit verifies nothing, so it fails
            log.error("audit of %s failed: %r", res.endpoint.key, outcome)
            res.errors.append(f"audit failed: {type(outcome).__name__}")
            res.tally.add(_invalid("audit aborted"))
    return results
