"""Where each endpoint stands, as a validator saw it in its last epoch.

Validators that run a service (the subnet owner, which is the throughput auditor, and full validators with a
``public_host``) serve it read-only and signed at ``GET /v1/status/{hotkey}``, together with the hotkey's results in
the auditor's last ``STATUS_TESTS`` throughput tests (``protocol.status``), and every hotkey's at ``GET /v1/network``
with the weights the validator set (for network dashboards and light validators; see ``validator.service``). Miners
poll it and log it. It is that validator's own view; every validator scores the same way from the same rules.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Collection, Mapping
from pathlib import Path

from verathos.protocol.status import EndpointStatus, HotkeyStatus, NetworkStatus
from verathos.validator.audit import EndpointResult
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState, windowed_verdict

log = logging.getLogger(__name__)


def build(*, epoch: int, epoch_blocks: int, endpoints: list[DiscoveredEndpoint], results: dict[str, EndpointResult],
          scores: dict[str, float], failed: Collection[str], invalid_hotkeys: Collection[str],
          credited: dict[str, float], scored_models: Collection[str], state: ScoreState,
          model_shares: dict[str, float] | None = None, weights: dict[int, float] | None = None,
          drifted: Collection[str] = (), user_availability: Mapping[str, float] | None = None,
          latency_factors: Mapping[str, float] | None = None,
          taken: Mapping[str, EndpointStatus] | None = None) -> dict[str, HotkeyStatus]:
    """Status per hotkey after ``update`` has folded the epoch into ``state``; ``weights``: per UID, as set;
    ``drifted``, ``user_availability`` and ``latency_factors`` per endpoint key, as scored; ``taken``: the source's
    status of the endpoints whose results it gave (``validator.fallback``), whose checks and state they show."""
    out: dict[str, HotkeyStatus] = {}
    for ep in endpoints:
        key, r = ep.key, results.get(ep.key)
        source = (taken or {}).get(key) if r is None else None
        status = EndpointStatus(endpoint_id=ep.endpoint.endpoint_id, model_id=ep.endpoint.model_id,
                                state="scored", declared_rps=ep.endpoint.capacity_rps,
                                credited_rps=credited.get(key, 0.0), score=scores.get(key, 0.0),
                                ema=state.ema.get(key, 0.0), probation_epochs=state.probation.get(key, 0),
                                drift=key in drifted, user_availability=(user_availability or {}).get(key),
                                latency_factor=(latency_factors or {}).get(key))
        if r is not None:
            status.sent, status.answered, status.mean_abs = r.sent, r.answered, r.tally.mean_abs
            status.verdict, status.errors = windowed_verdict(state, key, r).value, sorted(set(r.errors))
        elif source is not None:
            status.sent, status.answered, status.mean_abs = source.sent, source.answered, source.mean_abs
            status.verdict, status.errors = source.verdict, list(source.errors)
        if ep.hotkey in invalid_hotkeys:
            status.state = "invalid"
        elif key in failed:
            status.state = "failed"
        elif status.probation_epochs:
            status.state = "probation"
        elif source is not None:
            status.state = source.state
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
    """The latest epoch's status, replaced as a whole after each epoch (``revision`` counts the replacements). The
    network's document is built once per epoch: it is public, and every request for it would otherwise cost the
    validator's process a full rebuild. With a ``path`` it is also kept on disk, so a restarted validator serves the
    last epoch's status, and the weights it set, until its next epoch."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.revision = 0
        self._by_hotkey: dict[str, HotkeyStatus] = {}
        saved = NetworkStatus()
        if path is not None and path.exists():
            try:
                saved = NetworkStatus.model_validate_json(path.read_bytes())
                self._by_hotkey = {h.hotkey: h for h in saved.hotkeys}
            except (OSError, ValueError) as exc:
                log.warning("the saved status in %s is not used: %s", path, exc)
        self._network = saved.model_dump_json().encode()
        self._lock = threading.Lock()

    def publish(self, by_hotkey: dict[str, HotkeyStatus], *, epoch: int | None = None, epoch_blocks: int | None = None,
                model_shares: dict[str, float] | None = None, weights: dict[int, float] | None = None) -> None:
        """Replace the status with ``by_hotkey``. The network's document also states ``weights`` (per UID, as set)
        and the epoch, its length and the emission shares (by default the first hotkey's)."""
        network = _network(by_hotkey, epoch=epoch, epoch_blocks=epoch_blocks, model_shares=model_shares,
                           weights=weights).model_dump_json().encode()
        with self._lock:
            self._by_hotkey, self._network = by_hotkey, network
            self.revision += 1
        if self.path is not None:
            try:
                partial = self.path.with_suffix(".partial")
                partial.write_bytes(network)
                os.replace(partial, self.path)
            except OSError as exc:
                log.warning("the status could not be saved to %s: %s", self.path, exc)

    def get(self, hotkey: str) -> HotkeyStatus | None:
        with self._lock:
            return self._by_hotkey.get(hotkey)

    def network(self) -> bytes:
        """Every hotkey's status (``NetworkStatus``) as JSON."""
        with self._lock:
            return self._network

    def versioned_network(self) -> tuple[int, bytes]:
        """``(revision, network())``, read together."""
        with self._lock:
            return self.revision, self._network


def _network(by_hotkey: dict[str, HotkeyStatus], **given) -> NetworkStatus:
    hotkeys = sorted(by_hotkey.values(), key=lambda h: (h.uid is None, h.uid, h.hotkey))
    first = hotkeys[0] if hotkeys else None
    fields = {}
    if first is not None:
        fields = {"epoch": first.epoch, "epoch_blocks": first.epoch_blocks, "model_shares": first.model_shares}
    fields.update({k: v for k, v in given.items() if v is not None})
    return NetworkStatus(**fields, hotkeys=hotkeys)
