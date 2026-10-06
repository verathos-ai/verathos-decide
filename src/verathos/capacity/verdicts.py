"""Capacity verdicts the auditor publishes for the other validators.

The auditor serves the endpoints it gates for an epoch, signed by its hotkey, at ``GET /capacity/v1/verdicts``.
Every other validator fetches them from the auditor's axon once per epoch, checks the signature against the auditor
hotkey, and gates the same endpoints. Without a valid feed a validator gates nothing for that epoch: an unreachable
auditor never costs miners their score.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from bittensor_wallet import Keypair

from verathos.protocol import signing

log = logging.getLogger(__name__)

VERDICTS_PATH = "/capacity/v1/verdicts"
VERDICTS_DOMAIN = "verathos/capacity-verdicts/v1"


def signed_verdicts(keypair: Keypair, netuid: int, epoch: int, gated: dict[tuple[str, str], str]) -> dict[str, Any]:
    verdicts = {"netuid": netuid, "epoch": epoch,
                "gated": [{"hotkey": hk, "endpoint_id": ep, "rule": rule} for (hk, ep), rule in sorted(gated.items())]}
    return {"verdicts": verdicts, "signature": signing.sign(keypair, VERDICTS_DOMAIN, verdicts)}


def accepted_verdicts(document: Any, *, auditor_hotkey: str, netuid: int, epoch: int) -> dict[tuple[str, str], str]:
    """The gated endpoints of a verdict document, or ValueError if it is not the auditor's for this epoch."""
    if not isinstance(document, dict) or not isinstance(document.get("verdicts"), dict):
        raise ValueError("malformed verdict document")
    verdicts, signature = document["verdicts"], document.get("signature")
    if not isinstance(signature, str) or not signing.verify(auditor_hotkey, VERDICTS_DOMAIN, verdicts, signature):
        raise ValueError("verdicts are not signed by the auditor")
    if verdicts.get("netuid") != netuid or verdicts.get("epoch") != epoch:
        raise ValueError("verdicts are for another netuid or epoch")
    return {(v["hotkey"], v["endpoint_id"]): v["rule"] for v in verdicts.get("gated", [])}


async def fetch_verdicts(client: httpx.AsyncClient, auditor_url: str | None, *, auditor_hotkey: str, netuid: int,
                         epoch: int) -> dict[tuple[str, str], str]:
    """The auditor's gated endpoints for ``epoch``; empty when the auditor is unreachable or the feed is invalid."""
    if not auditor_url:
        log.warning("capacity auditor %s has no axon; no capacity verdicts this epoch", auditor_hotkey[:12])
        return {}
    try:
        r = await client.get(auditor_url.rstrip("/") + VERDICTS_PATH, params={"epoch": epoch}, timeout=15)
        r.raise_for_status()
        return accepted_verdicts(r.json(), auditor_hotkey=auditor_hotkey, netuid=netuid, epoch=epoch)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        log.warning("capacity verdicts unavailable this epoch: %s", exc)
        return {}
