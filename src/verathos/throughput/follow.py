"""Following the chain head and the test schedule."""

from __future__ import annotations

import logging
import threading
import time

from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.policy import AuditPolicy, Schedule, ScheduledTest, eligible_range

log = logging.getLogger(__name__)
POLL_S = 2.0


class ScheduleFollower:
    """Feeds every block to the schedule in order, from the first block of the epoch it starts in, and knows whether
    a test's traffic is running at the head."""

    def __init__(self, chain: SubtensorChainView, policy: AuditPolicy) -> None:
        self.chain = chain
        self.head = 0
        self._lock = threading.Lock()
        self._reset(policy)

    def _reset(self, policy: AuditPolicy) -> None:
        self.policy, self.schedule, self._next = policy, Schedule(policy), None

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt a new policy; the current epoch is replayed under it."""
        with self._lock:
            if policy != self.policy:
                self._reset(policy)

    def poll(self) -> list[ScheduledTest]:
        """Read the head and feed the blocks up to it; returns the tests they select."""
        head, head_hash = self.chain.head()
        if head_hash is None:
            return []
        with self._lock:
            epoch_start = eligible_range(head // self.policy.epoch_blocks, self.policy)[0]
            self._next = epoch_start if self._next is None else max(self._next, epoch_start)  # earlier epochs are over
            selected = []
            while self._next <= head:
                block_hash = head_hash if self._next == head else self.chain.block_hash(self._next)
                if block_hash is None:
                    break
                test = self.schedule.observe(self._next, block_hash)
                if test is not None:
                    selected.append(test)
                self._next += 1
            self.head = head
        return selected

    def holding(self) -> bool:
        """Whether a test's traffic runs at the head (validators hold their canaries). Never waits for a poll."""
        return self.schedule.held(self.head)

    def start(self, interval_s: float = POLL_S) -> None:
        """Follow in a background thread (validators that only hold canaries)."""

        def run() -> None:
            while True:
                try:
                    for test in self.poll():
                        log.info("throughput test of epoch %d at block %d", test.epoch, test.start_block)
                except Exception as exc:
                    log.warning("following the test schedule failed: %s", exc)
                time.sleep(interval_s)

        threading.Thread(target=run, name="throughput-schedule", daemon=True).start()
