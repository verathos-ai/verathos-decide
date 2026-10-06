"""One epoch of canary audits.

For every model with a reference engine, the validator computes reference answers for the epoch's canary pool once,
then sends each endpoint its assigned subset as signed requests, checks every receipt, and compares the answers with
the reference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from bittensor_wallet import Keypair

from decide.miner.engine import Engine, EngineError
from decide.protocol.api import distributions
from decide.protocol.receipts import SignedReceipt
from decide.protocol.requests import sign_request
from decide.validator.canaries import assignment, canary_pool
from decide.validator.discovery import DiscoveredEndpoint, verify_identity
from decide.verify.compare import EndpointTally, Tolerance, compare_request

log = logging.getLogger(__name__)


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


async def _audit_endpoint(result: EndpointResult, canaries: list[tuple[dict[str, Any], dict[str, dict[str, float]]]],
                          keypair: Keypair, client: httpx.AsyncClient, spread_s: float, rng: random.Random) -> None:
    ep = result.endpoint
    result.identity_ok = await verify_identity(ep, keypair, client)
    path = ep.path_prefix + "/v1/systemone"
    for request, reference in canaries:
        if spread_s:
            await asyncio.sleep(rng.uniform(0, spread_s))
        body = json.dumps(request).encode()
        headers = {"Content-Type": "application/json", **sign_request(keypair, "POST", path, body, ep.hotkey)}
        result.sent += 1
        t0 = time.monotonic()
        try:
            r = await client.post(ep.base_url + path, content=body, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            receipt = SignedReceipt.model_validate(data["receipt"])
            answers = data["answers"]
        except (httpx.HTTPError, KeyError, ValueError) as e:
            result.errors.append(f"request failed: {type(e).__name__}")
            continue
        r_ = receipt.receipt
        if not (receipt.signature_valid() and receipt.binds(request, answers) and r_.miner_hotkey == ep.hotkey
                and r_.endpoint_id == ep.endpoint.endpoint_id and r_.model_id == ep.endpoint.model_id
                and r_.caller_hotkey == keypair.ss58_address):
            result.errors.append("invalid receipt")
            continue
        result.answered += 1
        result.latencies_ms.append((time.monotonic() - t0) * 1000)
        try:
            got = distributions(answers)
        except ValueError:
            got = {}
        result.tally.add(compare_request(reference, got, result.tolerance))


async def audit_epoch(
    endpoints: list[DiscoveredEndpoint],
    *,
    epoch: int,
    secret: bytes,
    keypair: Keypair,
    references: dict[str, Engine],
    tolerances: dict[str, Tolerance],
    client: httpx.AsyncClient,
    pool_size: int = 10,
    per_endpoint: int = 6,
    spread_s: float = 0.0,
) -> dict[str, EndpointResult]:
    """Audit every endpoint whose model has a reference engine. Returns results keyed by endpoint key."""
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
        res = EndpointResult(ep, tolerances[ep.endpoint.model_id])
        results[ep.key] = res
        tasks.append(_audit_endpoint(res, picked, keypair, client, spread_s, random.Random(rng.random())))
    await asyncio.gather(*tasks)
    return results
