"""One epoch of canary audits.

For every model with a reference engine, the validator computes reference answers for the epoch's canary pool once
(the models side by side, each within ``REFERENCE_SHARE`` of the epoch's audit time: what is not done by then is left
out of the pool), then sends each endpoint its assigned subset of the canaries computed, spread over the time left,
as signed requests, checks every receipt, and compares the answers with the reference. The reference work does not
grow with the number of endpoints, and a slow reference shrinks the pool instead of delaying the epoch. Nothing a
miner returns can abort the epoch: every reply is size- and time-bounded and every parse failure is recorded against
that endpoint. An endpoint gets canaries only once it proved control at its URL in the epoch
(``discovery.Identities``), while the references are computed; at its hotkey's own hosts a failed challenge is tried
again before each canary.
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
from verathos.throughput.policy import AuditPolicy
from verathos.validator.canaries import assignment, canary_pool
from verathos.validator.compare import EndpointTally, RequestComparison, Tolerance, compare_request
from verathos.validator.discovery import DiscoveredEndpoint, Identities

log = logging.getLogger(__name__)
HOLD_POLL_S = 2.0
MAX_HOLD_WAIT_S = 120.0
MAX_REPLY_BYTES = 1 << 20
REFERENCE_SHARE = 0.3  # of an epoch's audit time, the most the canaries' reference answers may take


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


async def reference_answers(pool: list[dict[str, Any]], engine: Engine,
                            deadline: float | None = None) -> list[dict[str, dict[str, float]] | None]:
    """Reference distributions for each canary, in order until ``deadline`` (``time.monotonic``); None where the
    reference engine failed or the time was up."""
    out: list[dict[str, dict[str, float]] | None] = []
    for request in pool:
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            out.append(None)
            continue
        try:
            out.append(distributions((await asyncio.wait_for(engine.answer(request), left))["answers"]))
        except TimeoutError:
            out.append(None)
        except (EngineError, KeyError, ValueError) as e:
            log.warning("reference failed for a %s canary: %s", request["model"], e)
            out.append(None)
    missing = sum(r is None for r in out)
    if missing:
        log.warning("%s: %d of %d canaries without a reference answer this epoch", pool[0]["model"] if pool else "?",
                    missing, len(pool))
    return out


def _invalid(reason: str) -> RequestComparison:
    return RequestComparison(1.0, 0.0, 0, 0, reason)


async def _canary(result: EndpointResult, request: dict[str, Any], reference: dict[str, dict[str, float]],
                  keypair: Keypair, client: httpx.AsyncClient, netuid: int, mechid: int, timeout_s: float) -> None:
    """Send one canary and record its outcome. A request the endpoint declines (no 2xx reply within ``timeout_s``)
    costs availability; a reply whose receipt does not bind it, or whose answers are malformed, fails the request
    outright."""
    ep = result.endpoint
    body = json.dumps(request).encode()
    headers = {"Content-Type": "application/json",
               **sign_request(keypair, "POST", ep.path_prefix + "/v1/systemone", body, ep.hotkey)}
    result.sent += 1
    t0 = time.monotonic()
    try:
        data = await request_json(client, "POST", ep.url + "/v1/systemone", content=body, headers=headers,
                                  limit=MAX_REPLY_BYTES, timeout=timeout_s)
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
                          holding: Callable[[], bool], netuid: int, mechid: int, identities: Identities, epoch: int,
                          timeout_s: float) -> None:
    """Send the endpoint its canaries, each only while it has proven control at its URL this epoch."""
    ep = result.endpoint
    for request, reference in canaries:
        if not result.identity_ok and not identities.own(ep, epoch):
            break  # not the hotkey's host: no traffic
        if spread_s:
            await asyncio.sleep(rng.uniform(0, spread_s))
        waited = 0.0
        while holding() and waited < MAX_HOLD_WAIT_S:
            await asyncio.sleep(HOLD_POLL_S)  # a throughput test is running
            waited += HOLD_POLL_S
        if not result.identity_ok:
            await identities.verify([ep], epoch, keypair, client)
            result.identity_ok = identities.verified(ep.key, ep.url, epoch)
            if not result.identity_ok:
                continue
        await _canary(result, request, reference, keypair, client, netuid, mechid, timeout_s)
    if not result.identity_ok:
        result.errors.append("identity not proven at its URL")


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
    budget_s: float | None = None,
    holding: Callable[[], bool] = lambda: False,
    request_timeout_s: float = AuditPolicy.request_timeout_s,
    identities: Identities | None = None,
) -> dict[str, EndpointResult]:
    """Audit every endpoint whose model has a reference engine, within ``budget_s`` (None: references unbounded,
    canaries at once), each canary within ``request_timeout_s`` as test requests are. Returns results keyed by
    endpoint key; endpoints without a usable canary this epoch (no reference engine, or no reference answer) have no
    result. ``identities``: the validator's record of who proved control where (default: a new one)."""
    start = time.monotonic()
    identities = identities or Identities()
    deadline = None if budget_s is None else start + REFERENCE_SHARE * budget_s
    models = [m for m in sorted({e.endpoint.model_id for e in endpoints}) if m in references]
    for model_id in sorted({e.endpoint.model_id for e in endpoints} - set(models)):
        log.info("no reference engine for %s; its endpoints are not audited this epoch", model_id)
    canaries = {m: canary_pool(secret, epoch, m, pool_size) for m in models}
    audited = [e for e in endpoints if e.endpoint.model_id in models]
    *answers, _ = await asyncio.gather(*(reference_answers(canaries[m], references[m], deadline) for m in models),
                                       identities.verify(audited, epoch, keypair, client))
    pools = {m: [(c, a) for c, a in zip(canaries[m], got) if a is not None] for m, got in zip(models, answers)}
    spread_s = 0.0 if budget_s is None else max(start + budget_s - time.monotonic(), 0.0) / max(per_endpoint, 1)

    results: dict[str, EndpointResult] = {}
    tasks = []
    rng = random.Random(int.from_bytes(secret[:8], "big") ^ epoch)
    for ep in endpoints:
        pool = pools.get(ep.endpoint.model_id)
        if pool is None:
            continue
        picked = [pool[i] for i in assignment(secret, epoch, ep.key, len(pool), per_endpoint)]
        if not picked:
            continue
        res = EndpointResult(ep, tolerances[ep.endpoint.model_id],
                             identity_ok=identities.verified(ep.key, ep.url, epoch))
        results[ep.key] = res
        tasks.append(_audit_endpoint(res, picked, keypair, client, spread_s, random.Random(rng.random()), holding,
                                     netuid, mechid, identities, epoch, request_timeout_s))
    for res, outcome in zip(results.values(), await asyncio.gather(*tasks, return_exceptions=True)):
        if isinstance(outcome, BaseException):  # an aborted audit verifies nothing, so it fails
            log.error("audit of %s failed: %r", res.endpoint.key, outcome)
            res.errors.append(f"audit failed: {type(outcome).__name__}")
            res.tally.add(_invalid("audit aborted"))
    return results
