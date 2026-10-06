"""Audit holds as seen by validators that do not run the audit.

An endpoint drafted into an audit window answers 503 while its bench runs. A validator that is not the auditor
derives the same windows and cohorts from public chain data, like the miners do, and holds its canaries for that
endpoint until the hold ends, so an audit is never counted as unavailability.
"""

from __future__ import annotations

import logging
import threading
import time

from verathos.capacity import core
from verathos.capacity.miner import ChainView

log = logging.getLogger(__name__)


class HoldTracker:
    """Follows current-head blocks and tracks which slots are inside an audit hold."""

    def __init__(self, *, netuid: int, policy: core.AuditPolicy, chain: ChainView,
                 block_time_s: float = core.DEFAULT_BLOCK_TIME_S) -> None:
        self.netuid, self.policy, self.chain, self.block_time_s = netuid, policy, chain, block_time_s
        self._lock = threading.Lock()
        self._slots: list[core.Slot] = []
        self._drafted_at: dict[str, int] = {}
        self._busy_until: dict[str, int] = {}
        self._last_block = 0
        self._running = False

    def update_slots(self, slots: list[core.Slot]) -> None:
        with self._lock:
            self._slots = list(slots)

    def start(self, poll_s: float = 1.0) -> None:
        self._running = True
        threading.Thread(target=self._run, args=(poll_s,), name="capacity-holds", daemon=True).start()

    def _run(self, poll_s: float) -> None:
        while self._running:
            try:
                self.poll_once()
            except Exception:
                log.exception("audit hold tracker error")
            time.sleep(poll_s)

    def poll_once(self) -> None:
        head, head_hash = self.chain.head()
        if head <= 0:
            return
        if self._last_block <= 0:
            self._last_block = head - 1
        for n in range(self._last_block + 1, head + 1):
            h = head_hash if n == head and head_hash else self.chain.block_hash(n)
            if h is None:
                return
            self._last_block = n
            window = core.open_window(self.netuid, n, h, self.chain.epoch_blocks(), self.policy)
            if window is None:
                continue
            with self._lock:
                busy = {s for s, until in self._busy_until.items() if n <= until}
                cooldown = core.payload_cooldown_blocks(self.policy, self.block_time_s)
                for slot, _row in core.draft(window, self._slots, self.policy, busy):
                    self._drafted_at[slot.slot_id] = n
                    self._busy_until[slot.slot_id] = window.proof_block + cooldown

    def drained(self, hotkey: str, endpoint_id: str) -> bool:
        slot_id = core.Slot(self.netuid, hotkey, endpoint_id, "", "").slot_id
        hold_blocks = core.active_hold_seconds(self.policy, self.block_time_s) / self.block_time_s
        with self._lock:
            drafted = self._drafted_at.get(slot_id)
        return drafted is not None and self._last_block - drafted <= hold_blocks
