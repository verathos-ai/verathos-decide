"""The miner's own standing, read from the throughput auditor's status page and written to the log.

Once per epoch the log gets one line per endpoint (scored, qualifying, failed, ...), and one line for every
throughput test of this hotkey once its outcome is final.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from verathos.protocol.http import get_json
from verathos.protocol.status import STATUS_PATH, EndpointStatus, HotkeyStatus, ThroughputOutcome

log = logging.getLogger("verathos.miner.status")
POLL_S = 120.0
FINAL_STATES = ("confirmed", "void")


def _answers(e: EndpointStatus) -> str:
    text = f"answers {e.verdict or 'unchecked'} {e.answered}/{e.sent}, mean diff {e.mean_abs:.4f}"
    return text + (f"; errors: {', '.join(e.errors)}" if e.errors else "")


def describe_endpoint(e: EndpointStatus) -> str:
    """One log line for an endpoint's state in the validator's last epoch."""
    name = f"{e.endpoint_id} ({e.model_id})"
    probation = f"zero score for {e.probation_epochs} more epochs"
    capacity = f"proven {e.credited_rps:.1f} of {e.declared_rps:.1f} requests/s declared"
    if e.state == "scored":
        return f"{name}: scored {e.score:.2f} this epoch, average {e.ema:.2f}; {capacity}; {_answers(e)}"
    if e.state == "qualifying":
        return f"{name}: qualifying, checked but not scored until it passes a throughput test; {_answers(e)}"
    if e.state == "candidate":
        return f"{name}: candidate model, checked but not scored; {capacity}; {_answers(e)}"
    if e.state == "invalid":
        return (f"{name}: a WRONG ANSWER (in a throughput test or to the gateway) was proven against this hotkey; "
                f"average reset, {probation}")
    if e.state == "failed":
        return f"{name}: answers FAILED verification this epoch; {probation}; {_answers(e)}"
    if e.state == "probation":
        return f"{name}: on probation, {probation}; {_answers(e)}"
    if e.state == "unanswered":
        return f"{name}: not scored, it did not answer or failed the identity check; {_answers(e)}"
    return f"{name}: not audited this epoch (the validator has no reference for this model); score held"


def describe_test(t: ThroughputOutcome) -> str:
    """One log line for a finished throughput test."""
    where = f"throughput test of epoch {t.epoch} (block {t.start_block}), {t.endpoint_id}"
    rates = f"delivered {t.delivered_rps:.1f} of {t.tested_rps:.1f} requests/s, p95 latency {t.latency_p95_ms:.0f} ms"
    if t.state == "void" or t.outcome == "void":
        return f"{where}: not counted ({t.reason})"
    if t.outcome == "passed":
        return f"{where}: passed, {rates}; credited {t.credited_rps:.1f}"
    if t.outcome == "partial":
        return f"{where}: BELOW the tested rate, {rates}; credited {t.credited_rps:.1f} ({t.reason})"
    return f"{where}: INVALID, {t.reason}"


class StatusWatcher:
    """Polls the auditor's status page for this hotkey and logs what changed."""

    def __init__(self, *, hotkey: str, auditor_url: Callable[[], str | None], listed: Callable[[], list[str]]) -> None:
        self.hotkey, self.auditor_url, self.listed = hotkey, auditor_url, listed
        self._epoch: int | None = None
        self._tests_logged: set[tuple[int, str]] = set()

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
        for line in self.lines(status):
            log.info(line)

    def lines(self, status: HotkeyStatus) -> list[str]:
        out = []
        if status.epoch is not None and status.epoch != self._epoch:
            self._epoch = status.epoch
            out.append(f"validator's epoch {status.epoch}:")
            seen = set()
            for e in status.endpoints:
                seen.add(e.endpoint_id)
                out.append("  " + describe_endpoint(e))
            for endpoint_id in self.listed():
                if endpoint_id not in seen:
                    out.append(f"  {endpoint_id}: not in that epoch (manifest not accepted yet, or published after "
                               "the epoch started)")
        for t in sorted(status.tests, key=lambda t: (t.epoch, t.endpoint_id)):
            key = (t.epoch, t.endpoint_id)
            if key not in self._tests_logged and t.state in FINAL_STATES:
                self._tests_logged.add(key)
                out.append(describe_test(t))
        return out
