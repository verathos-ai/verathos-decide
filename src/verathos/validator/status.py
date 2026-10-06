"""Where each endpoint stands, as a validator saw it in its last epoch.

Validators that run a service (the subnet owner and the capacity auditor) serve it read-only at
``GET /v1/status/{hotkey}``, together with the hotkey's recent capacity-audit outcomes when the validator is the
auditor. Miners poll it and log it. It is that validator's own view; every validator scores the same way from the
same rules.
"""

from __future__ import annotations

import threading
from collections.abc import Collection

from verathos.protocol.status import EndpointStatus, HotkeyStatus
from verathos.validator.audit import EndpointResult
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState, windowed_verdict


def build(*, epoch: int, epoch_blocks: int, endpoints: list[DiscoveredEndpoint], results: dict[str, EndpointResult],
          scores: dict[str, float], failed: Collection[str], gated: dict[str, str], unauditable: Collection[str],
          qualifying: Collection[str], scored_models: Collection[str], qualification_epochs: int,
          state: ScoreState) -> dict[str, HotkeyStatus]:
    """Status per hotkey after ``update`` has folded the epoch into ``state``."""
    out: dict[str, HotkeyStatus] = {}
    for ep in endpoints:
        key, r = ep.key, results.get(ep.key)
        status = EndpointStatus(endpoint_id=ep.endpoint.endpoint_id, model_id=ep.endpoint.model_id,
                                state="scored", score=scores.get(key, 0.0), ema=state.ema.get(key, 0.0),
                                probation_epochs=state.probation.get(key, 0), capacity_rule=gated.get(key, ""))
        if r is not None:
            status.sent, status.answered, status.mean_abs = r.sent, r.answered, r.tally.mean_abs
            status.verdict, status.errors = windowed_verdict(state, key, r).value, sorted(set(r.errors))
        if key in gated:
            status.state = "gated"
        elif key in failed:
            status.state = "failed"
        elif status.probation_epochs:
            status.state = "probation"
        elif key in unauditable:
            status.state = "unauditable"
        elif r is None:
            status.state = "not_audited"
        elif key in qualifying:
            status.state = "qualifying"
            status.scored_from_epoch = state.first_seen[key] + qualification_epochs
        elif ep.endpoint.model_id not in scored_models:
            status.state = "candidate"
        elif not r.identity_ok or not r.answered:
            status.state = "unanswered"
        out.setdefault(ep.hotkey, HotkeyStatus(hotkey=ep.hotkey, epoch=epoch, epoch_blocks=epoch_blocks,
                                               endpoints=[])).endpoints.append(status)
    return out


class StatusBoard:
    """The latest epoch's status, replaced as a whole after each epoch."""

    def __init__(self) -> None:
        self._by_hotkey: dict[str, HotkeyStatus] = {}
        self._lock = threading.Lock()

    def publish(self, by_hotkey: dict[str, HotkeyStatus]) -> None:
        with self._lock:
            self._by_hotkey = by_hotkey

    def get(self, hotkey: str) -> HotkeyStatus | None:
        with self._lock:
            return self._by_hotkey.get(hotkey)
