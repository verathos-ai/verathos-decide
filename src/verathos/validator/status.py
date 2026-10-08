"""Where each endpoint stands, as a validator saw it in its last epoch.

Validators that run a service (the subnet owner and the throughput auditor) serve it read-only at
``GET /v1/status/{hotkey}``, together with the hotkey's recent throughput-test results when the validator is the
auditor, and every hotkey's at ``GET /v1/network`` (for network dashboards). Miners poll it and log it. It is that
validator's own view; every validator scores the same way from the same rules.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping

from verathos.protocol.status import EndpointStatus, HotkeyStatus, NetworkStatus
from verathos.validator.audit import EndpointResult
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState, windowed_verdict


def build(*, epoch: int, epoch_blocks: int, endpoints: list[DiscoveredEndpoint], results: dict[str, EndpointResult],
          scores: dict[str, float], failed: Collection[str], invalid_hotkeys: Collection[str],
          credited: dict[str, float], scored_models: Collection[str], state: ScoreState,
          model_shares: dict[str, float] | None = None, weights: dict[int, float] | None = None,
          drifted: Collection[str] = (), user_availability: Mapping[str, float] | None = None,
          latency_factors: Mapping[str, float] | None = None) -> dict[str, HotkeyStatus]:
    """Status per hotkey after ``update`` has folded the epoch into ``state``; ``weights``: per UID, as set;
    ``drifted``, ``user_availability`` and ``latency_factors`` per endpoint key, as scored."""
    out: dict[str, HotkeyStatus] = {}
    for ep in endpoints:
        key, r = ep.key, results.get(ep.key)
        status = EndpointStatus(endpoint_id=ep.endpoint.endpoint_id, model_id=ep.endpoint.model_id,
                                state="scored", declared_rps=ep.endpoint.capacity_rps,
                                credited_rps=credited.get(key, 0.0), score=scores.get(key, 0.0),
                                ema=state.ema.get(key, 0.0), probation_epochs=state.probation.get(key, 0),
                                drift=key in drifted, user_availability=(user_availability or {}).get(key),
                                latency_factor=(latency_factors or {}).get(key))
        if r is not None:
            status.sent, status.answered, status.mean_abs = r.sent, r.answered, r.tally.mean_abs
            status.verdict, status.errors = windowed_verdict(state, key, r).value, sorted(set(r.errors))
        if ep.hotkey in invalid_hotkeys:
            status.state = "invalid"
        elif key in failed:
            status.state = "failed"
        elif status.probation_epochs:
            status.state = "probation"
        elif r is None:
            status.state = "not_audited"
        elif key in drifted:
            status.state = "drifted"
        elif key not in credited:
            status.state = "qualifying"
        elif ep.endpoint.model_id not in scored_models:
            status.state = "candidate"
        elif not r.identity_ok or not r.answered:
            status.state = "unanswered"
        out.setdefault(ep.hotkey, HotkeyStatus(
            hotkey=ep.hotkey, uid=ep.uid, weight=round((weights or {}).get(ep.uid, 0.0), 6), epoch=epoch,
            epoch_blocks=epoch_blocks, endpoints=[], model_shares=model_shares or {})).endpoints.append(status)
    return out


class StatusBoard:
    """The latest epoch's status, replaced as a whole after each epoch. The network's document is built once per
    epoch: it is public, and every request for it would otherwise cost the validator's process a full rebuild."""

    def __init__(self) -> None:
        self._by_hotkey: dict[str, HotkeyStatus] = {}
        self._network = _network({}).model_dump_json().encode()
        self._lock = threading.Lock()

    def publish(self, by_hotkey: dict[str, HotkeyStatus]) -> None:
        network = _network(by_hotkey).model_dump_json().encode()
        with self._lock:
            self._by_hotkey, self._network = by_hotkey, network

    def get(self, hotkey: str) -> HotkeyStatus | None:
        with self._lock:
            return self._by_hotkey.get(hotkey)

    def network(self) -> bytes:
        """Every hotkey's status (``NetworkStatus``) as JSON."""
        with self._lock:
            return self._network


def _network(by_hotkey: dict[str, HotkeyStatus]) -> NetworkStatus:
    hotkeys = sorted(by_hotkey.values(), key=lambda h: (h.uid is None, h.uid, h.hotkey))
    first = hotkeys[0] if hotkeys else None
    return NetworkStatus(epoch=first.epoch if first else None, epoch_blocks=first.epoch_blocks if first else None,
                         model_shares=first.model_shares if first else {}, hotkeys=hotkeys)
