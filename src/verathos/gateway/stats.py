"""What users get from each model, as this gateway process sees it, for the public status (``GET /v1/status``).

Per model and minute: the answered and failed user requests and whether the model was unavailable; the answers'
latencies of the last ``LATENCY_WINDOW_S``. Kept for a day, in memory: a restarted process starts afresh, so the
figures cover ``since`` until now.
"""

from __future__ import annotations

import collections
import time
from collections.abc import Callable
from typing import Any

DAY_MINUTES = 24 * 60
LATENCY_WINDOW_S = 900
MAX_LATENCIES = 50_000  # per model, within the window
DEGRADED_SUCCESS = 0.98  # below this share of answered requests in the last 15 minutes, a model is degraded
MIN_REQUESTS = 20  # fewer requests than this in the window say nothing about the success rate


class ModelStats:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.since = clock()
        self._minutes: dict[str, collections.OrderedDict[int, list]] = {}  # model -> minute -> [answered, failed, down]
        self._latencies: dict[str, collections.deque] = {}  # model -> (time, seconds)

    def _minute(self, model: str) -> list:
        minute = int(self.clock() // 60)
        minutes = self._minutes.setdefault(model, collections.OrderedDict())
        while minutes and next(iter(minutes)) <= minute - DAY_MINUTES:
            minutes.popitem(last=False)
        counts = minutes.get(minute)
        if counts is None:
            counts = minutes[minute] = [0, 0, False]
        return counts

    def record(self, model: str, *, answered: bool, seconds: float = 0.0) -> None:
        """One user request's outcome; requests the miners refused as invalid are not recorded."""
        self._minute(model)[0 if answered else 1] += 1
        if answered:
            self._latencies.setdefault(model, collections.deque(maxlen=MAX_LATENCIES)).append((self.clock(), seconds))

    def mark(self, model: str, *, unavailable: bool) -> None:
        """The model's state now; a minute in which it was unavailable at any check counts as unavailable."""
        counts = self._minute(model)
        counts[2] = counts[2] or unavailable

    def requests(self, model: str, minutes: int) -> tuple[int, int]:
        """(answered, failed) user requests in the last ``minutes``."""
        now = int(self.clock() // 60)
        answered = failed = 0
        for minute, counts in self._minutes.get(model, {}).items():
            if minute > now - minutes:
                answered, failed = answered + counts[0], failed + counts[1]
        return answered, failed

    def success_rate(self, model: str, minutes: int) -> float | None:
        answered, failed = self.requests(model, minutes)
        return None if answered + failed < MIN_REQUESTS else round(answered / (answered + failed), 4)

    def summary(self, model: str) -> dict[str, Any]:
        cutoff = self.clock() - LATENCY_WINDOW_S
        latencies = self._latencies.get(model, collections.deque())
        while latencies and latencies[0][0] < cutoff:
            latencies.popleft()
        recent = sorted(s for _, s in latencies)
        minutes = self._minutes.get(model, {})
        return {
            "latency_ms": None if not recent else {
                "p50": round(1000 * recent[len(recent) // 2]), "p95": round(1000 * recent[int(len(recent) * 0.95)]),
                "window_s": LATENCY_WINDOW_S},
            "success_rate_1h": self.success_rate(model, 60),
            "requests_1h": sum(self.requests(model, 60)),
            "availability_24h": None if not minutes else round(
                sum(1 for counts in minutes.values() if not counts[2]) / len(minutes), 4),
        }
