"""Admission to an endpoint's engine.

At most ``limit`` requests reach the engine at once (the concurrency at which it measured its best throughput; more
only slows it down); the others wait by class, then in order. Regular requests from validators and gateways go first,
then the auditor's test traffic, then validators' requests while test traffic arrives (``BEHIND_TESTS``). A request is
admitted only while no request of its class or a higher one waits, and a request that finds the queue full takes the
place of the newest waiting request of the lowest class below its own.
"""

from __future__ import annotations

import asyncio
from collections import deque

QUEUE_FACTOR = 4  # requests that may wait, per engine slot
WAIT_S = 30.0
REGULAR, TEST, BEHIND_TESTS = 0, 1, 2  # classes, first served first


class Admission:
    def __init__(self, limit: int, *, wait_s: float = WAIT_S) -> None:
        self.limit = limit  # may be replaced at any time (a new measurement)
        self.wait_s = wait_s
        self.inflight = 0
        self._queues: tuple[deque[asyncio.Future[bool]], ...] = (deque(), deque(), deque())

    async def acquire(self, cls: int = REGULAR, *, wait_s: float | None = None) -> bool:
        """Wait for an engine slot, at most ``wait_s`` (default ``self.wait_s``); False if the queue is full or the
        wait times out. Call ``release`` after a True."""
        if self.inflight < self.limit and not any(self._queues[:cls + 1]):
            self.inflight += 1
            return True
        if sum(map(len, self._queues)) >= QUEUE_FACTOR * self.limit:
            lower = next((q for q in reversed(self._queues[cls + 1:]) if q), None)
            if lower is None:
                return False
            lower.pop().set_result(False)
        queue = self._queues[cls]
        waiter = asyncio.get_running_loop().create_future()
        queue.append(waiter)
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), self.wait_s if wait_s is None else wait_s)
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
        while self.inflight < self.limit and any(self._queues):
            self.inflight += 1
            next(q for q in self._queues if q).popleft().set_result(True)
