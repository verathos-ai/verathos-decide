"""What the miners did for this gateway process's users, per endpoint and second, reported to the auditor: the
user requests sent to them and those they answered (with input tokens and durations), how often the router
considered them for a request, and how often they were not ready then or refused it (see ``router``).

Miners serve the gateway before the auditor's test traffic; the auditor counts what an endpoint answered the gateway
inside a test's measured window as delivered too, so serving users never costs a miner credit, and refusing users
during a test costs it. The input tokens are the users' demand per model, which sets the models' emission shares.
Refusals and durations are scored: what users get between tests. Every ``REPORT_S`` the counts of finished seconds
are posted (signed by the gateway) under a report id the auditor takes once; a failed post is sent again as it is,
same id, with the next report, while its counts are up to ``KEEP_S`` old, so a post the auditor took but whose answer
was lost is not counted twice.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from bittensor_wallet import Keypair

from verathos.protocol.gateway import (
    MAX_SERVED_BYTES,
    MAX_SERVED_COUNTS,
    SERVED_PATH,
    Route,
    ServedCount,
    latency_bucket,
)
from verathos.protocol.http import post_idempotent
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
REPORT_S = 5.0
KEEP_S = 120


@dataclass
class _Second:
    attempts: int = 0
    answered: int = 0
    input_tokens: int = 0
    offered: int = 0
    refused: int = 0
    latency: dict[int, int] = field(default_factory=dict)


class ServedCounter:
    """Counts per endpoint and second; the router's observer (``router.Observer``)."""

    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor_url: Callable[[], Awaitable[str | None]], auditor_hotkey: str,
                 clock: Callable[[], float] = time.time) -> None:
        self.keypair, self.client, self.auditor_url, self.auditor_hotkey = keypair, client, auditor_url, auditor_hotkey
        self.clock = clock
        self._counts: dict[tuple[str, str, int], _Second] = {}  # (hotkey, endpoint, second) -> counts
        self._unsent: list[tuple[int, int, bytes]] = []  # failed posts: (their latest second, counts, body)

    def _second(self, route: Route) -> _Second:
        return self._counts.setdefault((route.hotkey, route.endpoint_id, int(self.clock())), _Second())

    def record(self, route: Route, input_tokens: int, seconds: float = 0.0) -> None:
        """A user request the endpoint answered after ``seconds`` (the attempt's duration). It counts in the second
        its request was sent, so an answer held back cannot be moved into a test's window."""
        counts = self._counts.setdefault((route.hotkey, route.endpoint_id, int(self.clock() - seconds)), _Second())
        counts.answered += 1
        counts.input_tokens += input_tokens
        bucket = latency_bucket(seconds)
        counts.latency[bucket] = counts.latency.get(bucket, 0) + 1

    def attempted(self, route: Route) -> None:
        self._second(route).attempts += 1

    def offered(self, route: Route, refused: bool) -> None:
        counts = self._second(route)
        counts.offered += 1
        counts.refused += refused

    def refused(self, route: Route) -> None:
        self._second(route).refused += 1

    def pending(self) -> int:
        """Per-endpoint seconds not reported yet."""
        return len(self._counts) + sum(counts for _, counts, _ in self._unsent)

    async def report(self, *, final: bool = False) -> None:
        """Post the posts that failed before, then the counts of finished seconds (``final``: all of them); a post
        that fails is kept for the next report, whatever ended it."""
        now = int(self.clock())
        done = {k: n for k, n in self._counts.items() if final or k[2] < now}
        for key in done:
            del self._counts[key]
        self._unsent = [post for post in self._unsent if post[0] >= now - KEEP_S] + [
            (max(k[2] for k, _ in part), len(part), ('{"report_id": "' + secrets.token_hex(16) + '", "counts": ['
                                                     + ",".join(json for _, json in part) + "]}").encode())
            for part in _parts(done)]
        if not self._unsent:
            return
        try:
            url = await self.auditor_url()
        except Exception as exc:
            log.warning("the auditor's address is unknown: %s", exc)
            return
        if not url:
            log.warning("the auditor's address is unknown: %d served counts wait", self.pending())
            return
        for post in list(self._unsent):
            try:
                await self._post(url.rstrip("/") + SERVED_PATH, post[2])
            except Exception as exc:
                log.warning("reporting served counts failed: %s", exc)
                continue
            self._unsent.remove(post)

    async def _post(self, url: str, body: bytes) -> None:
        def headers() -> dict[str, str]:
            return {"Content-Type": "application/json",
                    **sign_request(self.keypair, "POST", SERVED_PATH, body, self.auditor_hotkey)}
        await post_idempotent(self.client, url, body, headers, timeout=10)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(REPORT_S)
            try:
                await self.report()
            except Exception:
                log.exception("reporting served counts failed")


def _parts(counts: dict[tuple[str, str, int], _Second]):
    """The counts as posts of at most ``MAX_SERVED_COUNTS`` entries and ``MAX_SERVED_BYTES``: [(key, json)]."""
    part: list[tuple[tuple[str, str, int], str]] = []
    size = 0
    for (hk, ep, second), n in counts.items():
        entry = ServedCount(hotkey=hk, endpoint_id=ep, second=second, attempts=n.attempts, answered=n.answered,
                            input_tokens=n.input_tokens, offered=n.offered, refused=n.refused,
                            latency=n.latency).model_dump_json()
        if part and (len(part) >= MAX_SERVED_COUNTS or size + len(entry) + 1 > MAX_SERVED_BYTES - 1024):
            yield part
            part, size = [], 0
        part.append(((hk, ep, second), entry))
        size += len(entry) + 1
    if part:
        yield part
