"""Rate limits: the generic cell rate algorithm (GCRA) per key.

A key may send ``rate`` requests per second, with up to ``burst`` at once. State is one timestamp per key, exact
within this process; with several gateway processes, route each API key to one of them.
"""

from __future__ import annotations

MAX_KEYS = 200_000


class RateLimiter:
    def __init__(self, max_keys: int = MAX_KEYS) -> None:
        self._tat: dict[object, float] = {}  # key -> theoretical arrival time of its next request
        self._max_keys = max_keys

    def check(self, key: object, rate: float, burst: int, now: float) -> float:
        """0.0 if a request of ``key`` is allowed now (and counts it); otherwise the seconds until it would be."""
        if rate <= 0:
            return float("inf")
        interval = 1.0 / rate
        tat = max(self._tat.get(key, now), now)
        wait = tat - now - interval * max(burst - 1, 0)
        if wait > 0:
            return wait
        if len(self._tat) >= self._max_keys:
            self._tat = {k: t for k, t in self._tat.items() if t > now}
        self._tat[key] = tat + interval
        return 0.0
