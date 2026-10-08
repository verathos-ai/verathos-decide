"""Endpoint status as validators serve it (``GET /v1/status/{hotkey}``, ``GET /v1/network`` for every hotkey)
and miners read it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

STATUS_PATH = "/v1/status/{hotkey}"
NETWORK_PATH = "/v1/network"

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
