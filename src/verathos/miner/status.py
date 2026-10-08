"""The miner's own standing, read from the throughput auditor's status page and written to the log.

Once per epoch the log gets one line per endpoint this machine serves (scored, qualifying, failed, ...), and one line
for every throughput test of those endpoints once its outcome is final (after a start, the latest of each endpoint).
The status page itself lists every endpoint of the hotkey, on all its machines.

The miner acts on the test results (``shortfall``) only from a status the subnet owner hotkey signed for this hotkey
and netuid, recently and no earlier than one it read before (``protocol.status.unverified``). Any other status is
logged all the same, with one warning line until a status verifies again.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from verathos.protocol.http import get_json
from verathos.protocol.status import (
    MAX_STATUS_BYTES,
    STATUS_DOMAIN,
    STATUS_PATH,
    EndpointStatus,
    HotkeyStatus,
    ThroughputOutcome,
    unverified,
)

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
        users = "" if e.user_availability is None else f"; ready for users {e.user_availability:.0%}"
        latency = "" if e.latency_factor is None else f", latency factor {e.latency_factor:.2f}"
        return (f"{name}: scored {e.score:.2%} of the epoch's emission, average {e.ema:.2%}; {capacity}{users}"
                f"{latency}; {_answers(e)}")
    if e.state == "drifted":
        return (f"{name}: NO CREDIT, the auditor's recomputes of its answers (in tests or to users) drifted beyond the "
                f"model's tolerance over recent checks; serve the catalog's weights at their precision; {_answers(e)}")
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
        return f"{where}: passed, {rates}; credited {t.credited_rps:.1f}" + (f" ({t.reason})" if t.reason else "")
    if t.outcome == "partial":
        return f"{where}: BELOW the tested rate, {rates}; credited {t.credited_rps:.1f} ({t.reason})"
    return f"{where}: INVALID, {t.reason}"


class StatusWatcher:
    """Polls the auditor's status page for this hotkey, logs what changed and hands the test results of a verified
    status to ``on_tests`` (the miner measures again an endpoint that fell short). ``auditor_hotkey``: the subnet
    owner hotkey, as read from the chain."""

    def __init__(self, *, hotkey: str, netuid: int, auditor_url: Callable[[], str | None],
                 auditor_hotkey: Callable[[], str | None], listed: Callable[[], list[str]],
                 on_tests: Callable[[list[ThroughputOutcome]], None] = lambda tests: None,
                 clock: Callable[[], float] = time.time) -> None:
        self.hotkey, self.netuid, self.auditor_url, self.auditor_hotkey = hotkey, netuid, auditor_url, auditor_hotkey
        self.listed, self.on_tests, self.clock = listed, on_tests, clock
        self._epoch: int | None = None
        self._tests_logged: set[tuple[int, str]] | None = None  # None until the first status
        self._issued_ms = 0  # when the latest verified status was signed
        self._refused: str | None = None  # why the latest status was not acted on

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
        document = get_json(base.rstrip("/") + STATUS_PATH.format(hotkey=self.hotkey), limit=MAX_STATUS_BYTES,
                            timeout=15)
        status = HotkeyStatus.model_validate(document)
        for line in self.lines(status):
            log.info(line)
        why = self.refused(document, status)
        if why is None:
            if self._refused is not None:
                log.info("the auditor's status verifies again: the miner acts on its test results")
            self._refused, self._issued_ms = None, document["issued_ms"]
            self.on_tests(status.tests)
        elif why != self._refused:
            self._refused = why
            log.warning("the auditor's status is logged but not acted on: %s. Until a status verifies, its test "
                        "results never make the miner measure again or cap a declaration", why)

    def refused(self, document: Any, status: HotkeyStatus) -> str | None:
        """Why the miner does not act on ``document`` (as received; ``status``: as read); None if it does."""
        why = unverified(document, domain=STATUS_DOMAIN, auditor=self.auditor_hotkey(), netuid=self.netuid,
                         now_ms=int(self.clock() * 1000))
        if why is None and status.hotkey != self.hotkey:
            return "it is another hotkey's status"
        if why is None and document["issued_ms"] < self._issued_ms:
            return "it was signed before a status already read"
        return why

    def lines(self, status: HotkeyStatus) -> list[str]:
        out, listed = [], self.listed()
        if status.epoch is not None and status.epoch != self._epoch:
            self._epoch = status.epoch
            out.append(f"validator's epoch {status.epoch}:")
            if status.model_shares:
                out.append("  emission shares: " + ", ".join(f"{m} {s:.1%}" for m, s in
                                                              sorted(status.model_shares.items())))
            if status.miner_share is not None and status.miner_share < 1:
                out.append(f"  miner share {status.miner_share:.2%}: the scores below are shares of it; the subnet "
                           "owner's UID gets the rest of the miners' emission")
            seen = set()
            for e in status.endpoints:
                if e.endpoint_id in listed:
                    seen.add(e.endpoint_id)
                    out.append("  " + describe_endpoint(e))
            for endpoint_id in listed:
                if endpoint_id not in seen:
                    out.append(f"  {endpoint_id}: not in that epoch (manifest not accepted yet, or published after "
                               "the epoch started)")
        final = sorted((t for t in status.tests if t.state in FINAL_STATES), key=lambda t: (t.epoch, t.endpoint_id))
        if self._tests_logged is None:  # the first status: earlier tests than each endpoint's latest go unlogged
            latest = {t.endpoint_id: (t.epoch, t.endpoint_id) for t in final}
            self._tests_logged = {(t.epoch, t.endpoint_id) for t in final} - set(latest.values())
        for t in final:
            key = (t.epoch, t.endpoint_id)
            if t.endpoint_id in listed and key not in self._tests_logged:
                self._tests_logged.add(key)
                out.append(describe_test(t))
        return out
