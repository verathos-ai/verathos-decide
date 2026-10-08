"""When a miner measures: not into a throughput test it knows of.

A measurement loads every engine of the machine for a few minutes. A test of its endpoints meanwhile finds them busy
and credits less than they serve, and the measurement counts less than they serve too. The miner follows the public
test schedule (``throughput.follow``) and, whatever the reason to measure (the start, a new catalog, a changed
engine, a shortfall, ``--remeasure``), starts no measurement from a test's selection block until its traffic is over
(``ScheduledTest.held``'s block of margin after it): the test starts ``lead_blocks`` after its selection. Before an
epoch's test is selected nobody knows when it comes, so the miner measures then, and test traffic that reaches the
measurement stops it within one request, inside the test's unmeasured warmup (``throughput.measure``); it is tried
again after the test. A machine that serves no endpoint yet measures at once: no test reaches it.

The schedule is read from the chain only when a measurement is about to start: the head, and the blocks since the
last read (an epoch's worth after a pause).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from verathos.subnet.chain import CHAIN_ERRORS, one_line
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.policy import AuditPolicy

log = logging.getLogger("verathos.miner")


@dataclass(frozen=True)
class Deferral:
    """Why a measurement waits: the epoch of the test it waits for, and the log line."""

    epoch: int
    line: str


class MeasurementDeferred(Exception):
    def __init__(self, deferral: Deferral) -> None:
        super().__init__(deferral.line)
        self.deferral = deferral


class MeasurementWindow:
    """Tells a measurement about to start whether a selected test's traffic is still to come or running."""

    def __init__(self, follower: ScheduleFollower) -> None:
        self.follower = follower

    def set_policy(self, policy: AuditPolicy) -> None:
        if policy != self.follower.policy:
            self.follower.set_policy(policy)

    def deferral(self) -> Deferral | None:
        """Why a measurement must not start now; None if it may. With the chain unreachable it may: test traffic
        still stops it."""
        try:
            self.follower.poll()
        except CHAIN_ERRORS as exc:
            log.warning("the test schedule could not be read (%s); measuring now, test traffic stops it", one_line(exc))
            return None
        if self.follower.head <= 0:
            log.warning("the chain head could not be read; measuring now, test traffic stops it")
            return None
        test = self.follower.due()
        if test is None:
            return None
        return Deferral(test.epoch, f"the throughput test of epoch {test.epoch} is selected, its traffic from block "
                                    f"{test.start_block}: measuring after it")
