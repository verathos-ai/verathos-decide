"""Log lines about what can happen on every request: each kind at most once per interval, with how often it happened."""

from __future__ import annotations

import time
from collections.abc import Callable

MAX_KINDS = 10_000


class Throttle:
    def __init__(self, every_s: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.every_s, self.clock = every_s, clock
        self._kinds: dict[str, list] = {}  # kind -> [when last logged, times since]

    def __call__(self, kind: str) -> int:
        """How many times ``kind`` happened (this time included) since it was last logged, if it should be logged
        now; 0 if not."""
        now = self.clock()
        state = self._kinds.get(kind)
        if state is None:
            if len(self._kinds) >= MAX_KINDS:
                self._kinds = {k: s for k, s in self._kinds.items() if now - s[0] < self.every_s}
            state = self._kinds[kind] = [now - self.every_s, 0]
        state[1] += 1
        if now - state[0] < self.every_s:
            return 0
        times, state[0], state[1] = state[1], now, 0
        return times
