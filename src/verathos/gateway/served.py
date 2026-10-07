"""What the miners did for this gateway process's users, per endpoint and second, reported to the auditor: the
requests they answered with their input tokens and durations, how often the router offered them a request, and how
often they refused (see ``router``).

Miners serve the gateway before the auditor's test traffic; the auditor counts what an endpoint answered the gateway
inside a test's measured window as delivered too, so serving users never costs a miner credit, and refusing users
during a test costs it. The input tokens are the users' demand per model, which sets the models' emission shares.
Refusals and durations are scored: what users get between tests. Every ``REPORT_S`` the counts of finished seconds
are posted (signed by the gateway); a failed post is retried with the next, for counts up to ``KEEP_S`` old.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from bittensor_wallet import Keypair

from verathos.protocol.gateway import (
    MAX_SERVED_COUNTS,
    SERVED_PATH,
    Route,
    ServedCount,
    ServedReport,
    latency_bucket,
)
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
REPORT_S = 5.0
KEEP_S = 120


@dataclass
class _Second:
    answered: int = 0
    input_tokens: int = 0
    offered: int = 0
    refused: int = 0
    latency: dict[int, int] = field(default_factory=dict)

    def add(self, other: _Second) -> None:
        self.answered += other.answered
        self.input_tokens += other.input_tokens
        self.offered += other.offered
        self.refused += other.refused
        for bucket, n in other.latency.items():
            self.latency[bucket] = self.latency.get(bucket, 0) + n


class ServedCounter:
    """Counts per endpoint and second; the router's observer (``router.Observer``)."""

    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor_url: Callable[[], Awaitable[str | None]], auditor_hotkey: str,
                 clock: Callable[[], float] = time.time) -> None:
        self.keypair, self.client, self.auditor_url, self.auditor_hotkey = keypair, client, auditor_url, auditor_hotkey
        self.clock = clock
        self._counts: dict[tuple[str, str, int], _Second] = {}  # (hotkey, endpoint, second) -> counts

    def _second(self, route: Route) -> _Second:
        return self._counts.setdefault((route.hotkey, route.endpoint_id, int(self.clock())), _Second())

    def record(self, route: Route, input_tokens: int, seconds: float | None = None) -> None:
        """A user request the endpoint answered, after ``seconds`` (the attempt's duration)."""
        counts = self._second(route)
        counts.answered += 1
        counts.input_tokens += input_tokens
        if seconds is not None:
            bucket = latency_bucket(seconds)
            counts.latency[bucket] = counts.latency.get(bucket, 0) + 1

    def offered(self, route: Route, refused: bool) -> None:
        counts = self._second(route)
        counts.offered += 1
        counts.refused += refused

    def refused(self, route: Route) -> None:
        self._second(route).refused += 1

    def pending(self) -> int:
        """Per-endpoint seconds not reported yet."""
        return len(self._counts)

    async def report(self) -> None:
        now = int(self.clock())
        done = {k: n for k, n in self._counts.items() if k[2] < now}
        if not done:
            return
        for key in done:
            del self._counts[key]
        url = await self.auditor_url()
        items = list(done.items())
        for i in range(0, len(items), MAX_SERVED_COUNTS):
            part = items[i:i + MAX_SERVED_COUNTS]
            try:
                if not url:
                    raise RuntimeError("auditor unknown")
                report = ServedReport(counts=[ServedCount(hotkey=hk, endpoint_id=ep, second=s, answered=n.answered,
                                                          input_tokens=n.input_tokens, offered=n.offered,
                                                          refused=n.refused, latency=n.latency)
                                              for (hk, ep, s), n in part])
                body = report.model_dump_json().encode()
                headers = {"Content-Type": "application/json",
                           **sign_request(self.keypair, "POST", SERVED_PATH, body, self.auditor_hotkey)}
                r = await self.client.post(url.rstrip("/") + SERVED_PATH, content=body, headers=headers, timeout=10)
                r.raise_for_status()
            except Exception as exc:
                for key, n in part:  # retried with the next report while still useful
                    if key[2] >= now - KEEP_S:
                        self._counts.setdefault(key, _Second()).add(n)
                log.warning("reporting served counts failed: %s", exc)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(REPORT_S)
            await self.report()
