"""Which endpoint answers a request.

Per model, the endpoints of the routes with their proven capacity (credited requests/s). A request goes to the better
of two endpoints drawn in proportion to capacity (power of two choices): the one with fewer requests in flight per
unit of capacity. In this process an endpoint takes at most ``credited x max_latency_s x share`` requests at once:
what it serves within the latency bound (Little's law), for this process's share of its capacity.

A circuit breaker per endpoint stops traffic after ``BREAKER_FAILURES`` failures in a row, for ``OPEN_S``; then one
probe request decides, and each failed probe doubles the time, up to ``MAX_OPEN_S``. A miner refusing work because it
is busy gets no requests for ``BUSY_S``. A slow endpoint gets fewer: its requests stay in flight longer.

The router tells an observer (the served counter) each time it considers an endpoint for a user request (draws it)
and whether it was ready then: not busy, not behind an open breaker, and not slow (its latest answer within
``SLOW_FACTOR`` times the latency bound it declared its capacity at); and each request the endpoint answered busy or
failed. Being drawn while full is neither, as long as its latest answer came within the latency bound: full beyond
it, the endpoint serves less than its capacity at the bound, so it was not ready. What users get from an endpoint
between tests is scored from that: the share of the times it was considered that it was ready.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import random
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from verathos.gateway import metrics
from verathos.gateway.logs import Throttle
from verathos.protocol.gateway import Route, Routes
from verathos.protocol.manifest import endpoint_host

log = logging.getLogger(__name__)

BREAKER_FAILURES = 3
OPEN_S = 5.0
MAX_OPEN_S = 120.0
BUSY_S = 1.0
SLOW_FACTOR = 2.0  # an endpoint whose latest answer took longer than this times max_latency_s is not ready
DRAWS = 8  # weighted draws before every endpoint is scanned


class Outcome(Enum):
    ANSWERED = "answered"
    BUSY = "busy"
    FAILED = "failed"


class Observer(Protocol):
    def offered(self, route: Route, refused: bool) -> None: ...

    def attempted(self, route: Route) -> None: ...

    def refused(self, route: Route) -> None: ...


class Breaker:
    def __init__(self) -> None:
        self.failures, self.open_until, self.open_s, self.probing = 0, 0.0, OPEN_S, False

    @property
    def is_open(self) -> bool:
        return self.failures >= BREAKER_FAILURES

    def allows(self, now: float) -> bool:
        return not self.is_open or (now >= self.open_until and not self.probing)

    def started(self) -> None:
        if self.is_open:
            self.probing = True

    def succeeded(self) -> None:
        self.failures, self.open_s, self.probing = 0, OPEN_S, False

    def failed(self, now: float) -> None:
        if self.probing:
            self.open_s = min(self.open_s * 2, MAX_OPEN_S)
        self.failures += 1
        self.probing = False
        if self.is_open:
            self.open_until = now + self.open_s


@dataclass(eq=False)
class Endpoint:
    route: Route
    cap: int  # requests at once in this process
    breaker: Breaker = field(default_factory=Breaker)
    in_flight: int = 0
    busy_until: float = 0.0  # it refused work: no requests until then
    slow: bool = False  # its latest answer took more than SLOW_FACTOR x max_latency_s
    latest_s: float = 0.0  # how long its latest answer took
    last_failure: str = ""  # how its latest failed attempt failed (kind and status or exception class, no content)

    @property
    def key(self) -> str:
        return f"{self.route.hotkey}/{self.route.endpoint_id}"

    @property
    def load(self) -> float:
        return self.in_flight / self.route.credited_rps


class Pool:
    """The endpoints of one model, drawn in proportion to capacity."""

    def __init__(self, endpoints: list[Endpoint]) -> None:
        self.endpoints = endpoints
        self._cumulative = list(itertools.accumulate(e.route.credited_rps for e in endpoints))

    def draw(self, rng: random.Random) -> Endpoint:
        return self.endpoints[bisect.bisect_right(self._cumulative, rng.random() * self._cumulative[-1])]


class Router:
    def __init__(self, *, share: float, max_latency_s: float, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.monotonic, observer: Observer | None = None) -> None:
        self.share, self.max_latency_s = share, max_latency_s
        self.rng, self.clock, self.observer = rng or random.Random(), clock, observer
        self._pools: dict[str, Pool] = {}
        self._current: set[int] = set()
        self._throttled = Throttle(clock=clock)

    def update(self, routes: Routes) -> None:
        """Adopt a new table; endpoints that stay (same URL and certificate pin) keep their load and breaker. A route is
        used only at a URL a manifest may list (``protocol.manifest.endpoint_host``: a public address, any machine of
        the miner), so not even a table the auditor signed can send this process's requests into a private
        network."""
        known = {(e.key, e.route.url, e.route.tls): e for pool in self._pools.values() for e in pool.endpoints}
        grouped: dict[str, list[Endpoint]] = {}
        for route in routes.routes:
            if endpoint_host(route.url, route.endpoint_id) is None:
                if times := self._throttled(f"{route.url} refused"):
                    log.error("route %s/%s refused (%d times since last logged): %s is not a public endpoint URL",
                              route.hotkey, route.endpoint_id, times, route.url)
                continue
            cap = max(1, math.ceil(route.credited_rps * self.max_latency_s * self.share))
            endpoint = known.get((f"{route.hotkey}/{route.endpoint_id}", route.url, route.tls))
            if endpoint is None:
                endpoint = Endpoint(route, cap)
            else:
                endpoint.route, endpoint.cap = route, cap
            grouped.setdefault(route.model_id, []).append(endpoint)
        self._pools = {model: Pool(endpoints) for model, endpoints in grouped.items()}
        self._current = {id(e) for endpoints in grouped.values() for e in endpoints}
        metrics.BREAKERS_OPEN.set(sum(e.breaker.is_open for pool in self._pools.values() for e in pool.endpoints))
        metrics.ROUTES.clear()
        for model, pool in self._pools.items():
            metrics.ROUTES.labels(model).set(len(pool.endpoints))

    def models(self) -> set[str]:
        return set(self._pools)

    def find(self, model_id: str, hotkey: str, endpoint_id: str) -> Endpoint | None:
        pool = self._pools.get(model_id)
        return next((e for e in pool.endpoints if (e.route.hotkey, e.route.endpoint_id) == (hotkey, endpoint_id)),
                    None) if pool else None

    def describe(self) -> list[dict[str, object]]:
        """Every endpoint with the router's state of it."""
        now = self.clock()
        return [{"model": model, "endpoint": e.key, "url": e.route.url, "tls": e.route.tls,
                 "credited_rps": e.route.credited_rps,
                 "latency_ms": e.route.latency_ms, "cap": e.cap, "in_flight": e.in_flight,
                 "busy_for_s": round(max(e.busy_until - now, 0.0), 1), "slow": e.slow, "last_failure": e.last_failure,
                 "breaker": {"open": e.breaker.is_open, "failures": e.breaker.failures, "probing": e.breaker.probing,
                             "open_for_s": round(max(e.breaker.open_until - now, 0.0), 1) if e.breaker.is_open
                             else 0.0}}
                for model, pool in sorted(self._pools.items()) for e in pool.endpoints]

    def health(self, model_id: str) -> dict[str, float]:
        """The model's routable endpoints, their credited capacity, and how many are failing (breaker open)."""
        pool = self._pools.get(model_id)
        endpoints = pool.endpoints if pool else []
        return {"endpoints": len(endpoints), "capacity_rps": round(sum(e.route.credited_rps for e in endpoints), 2),
                "failing": sum(e.breaker.is_open for e in endpoints)}

    def capacity(self, model_id: str) -> float:
        pool = self._pools.get(model_id)
        return sum(e.route.credited_rps for e in pool.endpoints) * self.share if pool else 0.0

    def pick(self, model_id: str, exclude: Collection[str] = (), *, observe: bool = True) -> Endpoint | None:
        """An endpoint for one attempt, not of a hotkey in ``exclude``; None if none can take it now. ``observe``: the
        request is user traffic, so each endpoint drawn is reported with whether it was ready."""
        pool = self._pools.get(model_id)
        if pool is None:
            return None
        now = self.clock()

        def usable(e: Endpoint) -> bool:
            return (e.in_flight < e.cap and e.busy_until <= now and e.route.hotkey not in exclude
                    and e.breaker.allows(now))

        picks: list[Endpoint] = []
        drawn: set[int] = set()
        for _ in range(DRAWS):
            e = pool.draw(self.rng)
            late = e.in_flight >= e.cap and e.latest_s > self.max_latency_s  # full, and beyond the bound
            if observe and self.observer is not None and id(e) not in drawn and e.route.hotkey not in exclude and (
                    e.in_flight < e.cap or e.slow or late):
                drawn.add(id(e))
                ready = e.busy_until <= now and e.breaker.allows(now) and not e.slow and not late
                self.observer.offered(e.route, refused=not ready)
            if usable(e) and e not in picks:
                picks.append(e)
                if len(picks) == 2:
                    break
        if not picks:
            usable_ = [e for e in pool.endpoints if usable(e)]
            if not usable_:
                return None
            picks = self.rng.choices(usable_, weights=[e.route.credited_rps for e in usable_], k=2)
        chosen = min(picks, key=lambda e: e.load)
        if observe and self.observer is not None and id(chosen) not in drawn:  # found by the scan, not drawn
            self.observer.offered(chosen.route, refused=chosen.slow)
        return chosen

    def down(self, model_id: str) -> bool:
        """Every endpoint of the model failed until its breaker opened: none is answering, as opposed to busy."""
        pool = self._pools.get(model_id)
        return pool is not None and bool(pool.endpoints) and all(e.breaker.is_open for e in pool.endpoints)

    def started(self, endpoint: Endpoint, *, observe: bool = True) -> None:
        """An attempt is sent; ``observe``: it is user traffic (the auditor judges a test against what was sent)."""
        endpoint.in_flight += 1
        endpoint.breaker.started()
        if observe and self.observer is not None:
            self.observer.attempted(endpoint.route)

    def finished(self, endpoint: Endpoint, outcome: Outcome | None, seconds: float | None = None, *,
                 observe: bool = True, why: str = "") -> None:
        """One attempt is over after ``seconds``; ``outcome`` None: it counts neither way yet, until ``penalize`` or
        ``settled`` (a probe stays the breaker's one probe until then, so a failed one doubles the open time)."""
        endpoint.in_flight -= 1
        endpoint.last_failure = why or endpoint.last_failure
        slow = seconds is not None and seconds >= SLOW_FACTOR * self.max_latency_s
        if outcome is Outcome.ANSWERED or slow:  # a timeout at the attempt's limit is slow too
            endpoint.slow = slow
        if outcome is Outcome.ANSWERED and seconds is not None:
            endpoint.latest_s = seconds
        if outcome is Outcome.ANSWERED:
            self._breaker(endpoint, endpoint.breaker.succeeded)
        elif outcome is Outcome.FAILED:
            self._breaker(endpoint, lambda: endpoint.breaker.failed(self.clock()))
        elif outcome is Outcome.BUSY:
            endpoint.breaker.probing = False
            endpoint.busy_until = self.clock() + BUSY_S
        if outcome in (Outcome.BUSY, Outcome.FAILED) and observe and self.observer is not None:
            self.observer.refused(endpoint.route)

    def penalize(self, endpoint: Endpoint, *, observe: bool = True) -> None:
        """Count a failure whose cause was unclear, once another miner has answered the same request."""
        self._breaker(endpoint, lambda: endpoint.breaker.failed(self.clock()))
        self.refused(endpoint, observe=observe)

    def settled(self, endpoint: Endpoint) -> None:
        """An attempt that ended with no outcome counts neither way after all (no other miner answered)."""
        endpoint.breaker.probing = False

    def refused(self, endpoint: Endpoint, *, observe: bool = True) -> None:
        """Report that the endpoint refused a user request, without a breaker failure."""
        if observe and self.observer is not None:
            self.observer.refused(endpoint.route)

    def _breaker(self, endpoint: Endpoint, change: Callable[[], None]) -> None:
        was_open = endpoint.breaker.is_open
        change()
        if endpoint.breaker.is_open == was_open:
            return
        if id(endpoint) in self._current:
            metrics.BREAKERS_OPEN.inc() if endpoint.breaker.is_open else metrics.BREAKERS_OPEN.dec()
        if endpoint.breaker.is_open and (times := self._throttled(f"{endpoint.key} opened")):
            log.warning("the breaker of %s opened (%d times since last logged) after %d failures, the latest: %s",
                        endpoint.key, times, endpoint.breaker.failures, endpoint.last_failure or "unknown")
        elif not endpoint.breaker.is_open and (times := self._throttled(f"{endpoint.key} closed")):
            log.info("the breaker of %s closed (%d times since last logged)", endpoint.key, times)
