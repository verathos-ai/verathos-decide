"""What users get from each model, as this gateway process sees it, for the public status (``GET /v1/status``).

Per model and minute: the user requests answered, failed and refused for want of free capacity (busy), and whether
the model was unavailable; the answers' latencies (at most ``MAX_LATENCIES``). Kept for a day, in memory: a
restarted process starts afresh, so the figures cover ``since`` until now. The success rate is of the requests the
model had capacity for: one user bursting beyond it says nothing about what the others get. The latency is of the
last ``LATENCY_WINDOW_S``, or, with fewer than ``MIN_LATENCIES`` answers there, of the fewest whole hours that hold
them (all of the day's answers when fewer), so a quiet quarter of an hour does not hide it.
"""

from __future__ import annotations

import collections
import math
import time
from collections.abc import Callable
from typing import Any

DAY_MINUTES = 24 * 60
LATENCY_WINDOW_S = 900
MIN_LATENCIES = 20  # fewer answers than this in LATENCY_WINDOW_S: the latency is of a longer window
MAX_LATENCIES = 50_000  # per model
DEGRADED_SUCCESS = 0.98  # below this share of answered requests in the last 15 minutes (busy left out): degraded
OUTCOMES = ("answered", "failed", "busy")
MIN_REQUESTS = 20  # fewer requests than this in the window say nothing about the success rate


class ModelStats:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self.since = clock()
        # model -> minute -> [answered, failed, busy, unavailable]
        self._minutes: dict[str, collections.OrderedDict[int, list]] = {}
        self._latencies: dict[str, collections.deque] = {}  # model -> (time, seconds)

    def _minute(self, model: str) -> list:
        minute = int(self.clock() // 60)
        minutes = self._minutes.setdefault(model, collections.OrderedDict())
        while minutes and next(iter(minutes)) <= minute - DAY_MINUTES:
            minutes.popitem(last=False)
        counts = minutes.get(minute)
        if counts is None:
            counts = minutes[minute] = [0, 0, 0, False]
        return counts

    def record(self, model: str, outcome: str, seconds: float = 0.0) -> None:
        """One user request's outcome: answered (after ``seconds``), failed or busy."""
        self._minute(model)[OUTCOMES.index(outcome)] += 1
        if outcome == "answered":
            self._latencies.setdefault(model, collections.deque(maxlen=MAX_LATENCIES)).append((self.clock(), seconds))

    def mark(self, model: str, *, unavailable: bool) -> None:
        """The model's state now; a minute in which it was unavailable at any check counts as unavailable."""
        counts = self._minute(model)
        counts[3] = counts[3] or unavailable

    def requests(self, model: str, minutes: int) -> tuple[int, int, int]:
        """(answered, failed, busy) user requests in the last ``minutes``."""
        now = int(self.clock() // 60)
        totals = [0, 0, 0]
        for minute, counts in self._minutes.get(model, {}).items():
            if minute > now - minutes:
                totals = [t + c for t, c in zip(totals, counts[:3])]
        return totals[0], totals[1], totals[2]

    def success_rate(self, model: str, minutes: int) -> float | None:
        """The share answered of the requests the model had capacity for."""
        answered, failed, _ = self.requests(model, minutes)
        return None if answered + failed < MIN_REQUESTS else round(answered / (answered + failed), 4)

    def busy_rate(self, model: str, minutes: int) -> float | None:
        """The share of the requests refused for want of free capacity."""
        answered, failed, busy = self.requests(model, minutes)
        total = answered + failed + busy
        return None if total < MIN_REQUESTS else round(busy / total, 4)

    def latency(self, model: str) -> dict[str, int] | None:
        """p50 and p95 in milliseconds over ``window_s`` (see the module docstring); None without answers today."""
        now = self.clock()
        latencies = self._latencies.get(model, collections.deque())
        while latencies and latencies[0][0] < now - 60 * DAY_MINUTES:
            latencies.popleft()
        if not latencies:
            return None
        age = now - latencies[-min(MIN_LATENCIES, len(latencies))][0]  # of the MIN_LATENCIES-th latest answer
        window = LATENCY_WINDOW_S if age <= LATENCY_WINDOW_S else min(60 * DAY_MINUTES, 3600 * math.ceil(age / 3600))
        recent = sorted(s for t, s in latencies if t >= now - window)
        return {"p50": round(1000 * recent[len(recent) // 2]), "p95": round(1000 * recent[int(len(recent) * 0.95)]),
                "window_s": window}

    def summary(self, model: str) -> dict[str, Any]:
        minutes = self._minutes.get(model, {})
        return {
            "latency_ms": self.latency(model),
            "success_rate_1h": self.success_rate(model, 60),
            "busy_rate_1h": self.busy_rate(model, 60),
            "requests_1h": sum(self.requests(model, 60)),
            "availability_24h": None if not minutes else round(
                sum(1 for counts in minutes.values() if not counts[3]) / len(minutes), 4),
        }
