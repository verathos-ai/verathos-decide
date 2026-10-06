"""The miner's own standing, read from the capacity auditor's status page and written to the log.

Once per epoch the log gets one line per endpoint (scored, qualifying, gated, ...), and one line for every capacity
audit of this hotkey once its outcome is final.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from verathos.capacity.core import DEFAULT_BLOCK_TIME_S, AuditPolicy
from verathos.protocol.http import get_json
from verathos.protocol.status import STATUS_PATH, AuditOutcome, EndpointStatus, HotkeyStatus

log = logging.getLogger("verathos.miner.status")
POLL_S = 120.0
FINAL_WINDOWS = ("confirmed", "void")

_GATE_RULES = {"timing": "its timed work missed the deadline", "missing_evidence": "audits were not completed",
               "invalid_proof": "a proof was invalid"}


def _gate_reason(rule: str) -> str:
    hotkey_wide = rule.startswith("hotkey_")
    reason = _GATE_RULES.get(rule.removeprefix("hotkey_"), rule)
    return f"{reason} (counted across this hotkey's endpoints)" if hotkey_wide else reason


def _answers(e: EndpointStatus) -> str:
    text = f"answers {e.verdict or 'unchecked'} {e.answered}/{e.sent}, mean diff {e.mean_abs:.4f}"
    return text + (f"; errors: {', '.join(e.errors)}" if e.errors else "")


def describe_endpoint(e: EndpointStatus, *, epoch_blocks: int, head: int) -> str:
    """One log line for an endpoint's state in the validator's last epoch."""
    name = f"{e.endpoint_id} ({e.model_id})"
    probation = f"zero score for {e.probation_epochs} more epochs"
    if e.state == "scored":
        return f"{name}: scored {e.score:.2f} this epoch, average {e.ema:.2f}; {_answers(e)}"
    if e.state == "qualifying":
        minutes = max(0, (e.scored_from_epoch or 0) * epoch_blocks - head) * DEFAULT_BLOCK_TIME_S / 60
        return (f"{name}: qualifying, checked but not scored until epoch {e.scored_from_epoch} "
                f"(in about {minutes:.0f} min); {_answers(e)}")
    if e.state == "candidate":
        return f"{name}: candidate model, checked but not scored; {_answers(e)}"
    if e.state == "gated":
        return f"{name}: GATED by the capacity audit, {_gate_reason(e.capacity_rule)}; {probation}"
    if e.state == "failed":
        return f"{name}: answers FAILED verification this epoch; {probation}; {_answers(e)}"
    if e.state == "probation":
        return f"{name}: on probation, {probation}; {_answers(e)}"
    if e.state == "unauditable":
        return f"{name}: not scored, its GPU has no class in the catalog and cannot be audited"
    if e.state == "unanswered":
        return f"{name}: not scored, it did not answer or failed the identity check; {_answers(e)}"
    return f"{name}: not audited this epoch (the validator has no reference for this model); score held"


def describe_audit(a: AuditOutcome, *, limit_s: float) -> str:
    """One log line for a finished capacity audit."""
    where = f"capacity audit at block {a.selection_block}, {a.endpoint_id}"
    if a.outcome == "passed":
        return f"{where}: passed, timed work done in {a.final_s} s"
    if a.outcome == "timing_miss":
        return f"{where}: MISSED the deadline, final receipt after {a.final_s} s (limit {limit_s:.0f} s)"
    if a.outcome == "no_show":
        return f"{where}: MISSED, no final receipt arrived"
    if a.outcome == "missing_proof":
        return f"{where}: MISSED, no valid proof delivery ({a.reason})"
    if a.outcome == "invalid":
        return f"{where}: INVALID proof ({a.reason})"
    return f"{where}: not counted ({a.reason})"


class StatusWatcher:
    """Polls the auditor's status page for this hotkey and logs what changed."""

    def __init__(self, *, hotkey: str, auditor_url: Callable[[], str | None], head: Callable[[], int],
                 listed: Callable[[], list[str]], policy: Callable[[], AuditPolicy]) -> None:
        self.hotkey, self.auditor_url, self.head, self.listed, self.policy = hotkey, auditor_url, head, listed, policy
        self._epoch: int | None = None
        self._audits_logged: set[tuple[int, str]] = set()

    def start(self, interval_s: float = POLL_S) -> None:
        def run() -> None:
            while True:
                time.sleep(interval_s)
                try:
                    self.poll_once()
                except Exception as exc:
                    log.info("status unavailable: %s", exc)

        threading.Thread(target=run, name="miner-status", daemon=True).start()

    def poll_once(self) -> None:
        base = self.auditor_url()
        if not base:
            return
        status = HotkeyStatus.model_validate(
            get_json(base.rstrip("/") + STATUS_PATH.format(hotkey=self.hotkey), limit=1 << 20, timeout=15))
        for line in self.lines(status, self.head()):
            log.info(line)

    def lines(self, status: HotkeyStatus, head: int) -> list[str]:
        out = []
        if status.epoch is not None and status.epoch != self._epoch:
            self._epoch = status.epoch
            out.append(f"validator's epoch {status.epoch}:")
            seen = set()
            for e in status.endpoints:
                seen.add(e.endpoint_id)
                out.append("  " + describe_endpoint(e, epoch_blocks=status.epoch_blocks or 1, head=head))
            for endpoint_id in self.listed():
                if endpoint_id not in seen:
                    out.append(f"  {endpoint_id}: not in that epoch (manifest not accepted yet, or published after "
                               "the epoch started)")
        p = self.policy()
        for a in sorted(status.audits, key=lambda a: a.selection_block):
            key = (a.selection_block, a.endpoint_id)
            if key not in self._audits_logged and (a.window in FINAL_WINDOWS or a.outcome == "neutral"):
                self._audits_logged.add(key)
                out.append(describe_audit(a, limit_s=p.deadline_s + p.transport_grace_s))
        return out
