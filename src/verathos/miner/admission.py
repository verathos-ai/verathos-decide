"""Admission to an endpoint's engine.

At most ``limit`` requests reach the engine at once (the concurrency at which it measured its best throughput; more
only slows it down); the others wait in order. Regular requests from validators and gateways go ahead of the
auditor's test traffic: a test request is admitted only while no regular request waits, and a regular request that
finds the queue full takes the place of the newest waiting test request.
"""

from __future__ import annotations

import asyncio
from collections import deque

QUEUE_FACTOR = 4  # requests that may wait, per engine slot
WAIT_S = 30.0


class Admission:
    def __init__(self, limit: int, *, wait_s: float = WAIT_S) -> None:
        self.limit = limit  # may be replaced at any time (a new measurement)
        self.wait_s = wait_s
        self.inflight = 0
        self._regular: deque[asyncio.Future[bool]] = deque()
        self._test: deque[asyncio.Future[bool]] = deque()

    def _waiting(self) -> int:
        return len(self._regular) + len(self._test)

    async def acquire(self, *, test: bool) -> bool:
        """Wait for an engine slot; False if the queue is full or the wait times out. Call ``release`` after a
        True."""
        if self.inflight < self.limit and not self._regular and not (test and self._test):
            self.inflight += 1
            return True
        if self._waiting() >= QUEUE_FACTOR * self.limit:
            if test or not self._test:
                return False
            self._test.pop().set_result(False)
        queue = self._test if test else self._regular
        waiter = asyncio.get_running_loop().create_future()
        queue.append(waiter)
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), self.wait_s)
        except TimeoutError:
            return self._withdraw(waiter, queue)
        except asyncio.CancelledError:
            if self._withdraw(waiter, queue):
                self.release()
            raise

    def _withdraw(self, waiter: asyncio.Future[bool], queue: deque[asyncio.Future[bool]]) -> bool:
        """Take a waiter out of its queue; True if it had been handed a slot already."""
        if waiter.done():
            return waiter.result()
        waiter.cancel()
        queue.remove(waiter)
        return False

    def release(self) -> None:
        self.inflight -= 1
        while self.inflight < self.limit and (self._regular or self._test):
            self.inflight += 1
            (self._regular or self._test).popleft().set_result(True)
