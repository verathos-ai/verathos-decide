"""Endpoint status as validators serve it (``GET /v1/status/{hotkey}``, ``GET /v1/network`` for every hotkey)
and miners read it.

The validator that serves them, the subnet owner's (the throughput auditor), signs both documents with its hotkey:
to the status's own fields it adds ``netuid``, ``auditor`` (its hotkey) and ``issued_ms`` (when it signed), and signs
the canonical JSON of all of them (``protocol.signing``) into ``signature``. A reader that knows only the status's own
fields ignores the four. A miner acts on the test results in its status only once the document verifies
(``unverified``).
"""

from __future__ import annotations

from typing import Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel

from verathos.protocol import signing

STATUS_PATH = "/v1/status/{hotkey}"
NETWORK_PATH = "/v1/network"
STATUS_DOMAIN = "verathos/status/v1"
NETWORK_DOMAIN = "verathos/network-status/v1"
STATUS_TESTS = 10  # a hotkey's status lists its results in the auditor's last this many tests (one per epoch at most)
MAX_STATUS_BYTES = 1 << 20  # a status: up to 64 endpoints, and their results in STATUS_TESTS tests
STATUS_MAX_AGE_S = 300.0  # a miner acts on a status signed at most this long before or after its own clock

State = Literal["scored", "qualifying", "candidate", "failed", "invalid", "probation", "unanswered", "not_audited",
                "drifted"]


class EndpointStatus(BaseModel):
    endpoint_id: str
    model_id: str
    state: State
    declared_rps: float = 0.0
    credited_rps: float = 0.0  # capacity proven by the throughput audit, as scored this epoch
    drift: bool = False  # the auditor's recomputes of its answers drifted beyond the model's tolerance: no credit
    user_availability: float | None = None  # share of the times users found its hotkey's endpoints of the model ready
    latency_factor: float | None = None  # what its latency did to its score (1: neutral), if scored
    sent: int = 0
    answered: int = 0
    verdict: str = ""  # this epoch's answer check, over the statistical window
    mean_abs: float = 0.0
    errors: list[str] = []
    score: float = 0.0
    ema: float = 0.0
    probation_epochs: int = 0  # epochs of zero score still to come


class ThroughputOutcome(BaseModel):
    """One endpoint's result in a throughput test, as the auditor recorded it."""

    endpoint_id: str
    epoch: int
    start_block: int
    state: str  # unconfirmed | confirmed | void
    outcome: str  # passed | partial | invalid | void
    reason: str = ""
    declared_rps: float = 0.0
    tested_rps: float = 0.0
    delivered_rps: float = 0.0
    credited_rps: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


class HotkeyStatus(BaseModel):
    hotkey: str
    uid: int | None = None
    weight: float = 0.0  # the UID's share of the weights the validator set after the epoch
    epoch: int | None = None  # None: the validator has not scored the hotkey yet (test outcomes only)
    epoch_blocks: int | None = None
    endpoints: list[EndpointStatus] = []
    tests: list[ThroughputOutcome] = []
    model_shares: dict[str, float] = {}  # each active model's share of the epoch's emission


class NetworkStatus(BaseModel):
    """Every hotkey's status in the validator's last scored epoch."""

    epoch: int | None = None
    epoch_blocks: int | None = None
    model_shares: dict[str, float] = {}
    hotkeys: list[HotkeyStatus] = []


def sign_document(document: dict[str, Any], keypair: Keypair, *, netuid: int, domain: str,
                  now_ms: int) -> dict[str, Any]:
    """``document`` (a status in JSON form) with ``netuid``, ``auditor``, ``issued_ms`` and the ``signature`` of them
    all under ``domain``."""
    body = {**document, "netuid": netuid, "auditor": keypair.ss58_address, "issued_ms": now_ms}
    return {**body, "signature": signing.sign(keypair, domain, body)}


def unverified(document: Any, *, domain: str, auditor: str | None, netuid: int, now_ms: int) -> str | None:
    """Why ``document`` (as received) is not a document ``auditor`` signed under ``domain`` for ``netuid`` within
    ``STATUS_MAX_AGE_S`` of ``now_ms``; None if it is."""
    if not isinstance(document, dict) or "signature" not in document:
        return "it is not signed (the auditor runs an older release)"
    if not auditor:
        return "the subnet owner hotkey is not known yet"
    if document.get("auditor") != auditor or document.get("netuid") != netuid:
        return f"it is not signed by the subnet owner hotkey {auditor} for netuid {netuid}"
    if not signing.verify(auditor, domain, {k: v for k, v in document.items() if k != "signature"},
                          document["signature"]):
        return "its signature does not verify against the subnet owner hotkey"
    issued = document.get("issued_ms")
    if not isinstance(issued, int) or isinstance(issued, bool) or abs(now_ms - issued) > STATUS_MAX_AGE_S * 1000:
        return f"it was not signed within {STATUS_MAX_AGE_S:.0f} s of this machine's clock"
    return None
