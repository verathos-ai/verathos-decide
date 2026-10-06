"""Endpoint status as validators serve it (``GET /v1/status/{hotkey}``) and miners read it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

STATUS_PATH = "/v1/status/{hotkey}"

State = Literal["scored", "qualifying", "candidate", "failed", "gated", "probation", "unauditable", "unanswered",
                "not_audited"]


class EndpointStatus(BaseModel):
    endpoint_id: str
    model_id: str
    state: State
    scored_from_epoch: int | None = None  # set while qualifying
    sent: int = 0
    answered: int = 0
    verdict: str = ""  # this epoch's answer check, over the statistical window
    mean_abs: float = 0.0
    errors: list[str] = []
    score: float = 0.0
    ema: float = 0.0
    probation_epochs: int = 0  # epochs of zero score still to come
    capacity_rule: str = ""


class AuditOutcome(BaseModel):
    endpoint_id: str
    selection_block: int
    window: str  # pending | confirmed | void
    outcome: str
    reason: str = ""
    final_s: float | None = None  # seconds from the auditor's B_start observation to the final receipt


class HotkeyStatus(BaseModel):
    hotkey: str
    epoch: int | None = None  # None: the validator has not scored the hotkey yet (audit outcomes only)
    epoch_blocks: int | None = None
    endpoints: list[EndpointStatus] = []
    audits: list[AuditOutcome] = []
