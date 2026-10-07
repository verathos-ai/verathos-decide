"""What the miners answered this gateway process, per endpoint and second (requests and their input tokens), reported
to the auditor.

Miners serve the gateway before the auditor's test traffic; the auditor counts what an endpoint answered the gateway
inside a test's measured window as delivered too, so serving users never costs a miner credit, and refusing users
during a test gains nothing. The input tokens are the users' demand per model, which sets the models' emission
shares. Every ``REPORT_S`` the counts of finished seconds are posted (signed by the gateway); a
failed post is retried with the next, for counts up to ``KEEP_S`` old.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import httpx
from bittensor_wallet import Keypair

from verathos.protocol.gateway import MAX_SERVED_COUNTS, SERVED_PATH, Route, ServedCount, ServedReport
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
REPORT_S = 5.0
KEEP_S = 120


class ServedCounter:
    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor_url: Callable[[], Awaitable[str | None]], auditor_hotkey: str,
                 clock: Callable[[], float] = time.time) -> None:
        self.keypair, self.client, self.auditor_url, self.auditor_hotkey = keypair, client, auditor_url, auditor_hotkey
        self.clock = clock
        self._counts: dict[tuple[str, str, int], list[int]] = {}  # (hotkey, endpoint, second) -> [answered, tokens]

    def record(self, route: Route, input_tokens: int) -> None:
        counts = self._counts.setdefault((route.hotkey, route.endpoint_id, int(self.clock())), [0, 0])
        counts[0] += 1
        counts[1] += input_tokens

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
                report = ServedReport(counts=[ServedCount(hotkey=hk, endpoint_id=ep, second=s, answered=n[0],
                                                          input_tokens=n[1]) for (hk, ep, s), n in part])
                body = report.model_dump_json().encode()
                headers = {"Content-Type": "application/json",
                           **sign_request(self.keypair, "POST", SERVED_PATH, body, self.auditor_hotkey)}
                r = await self.client.post(url.rstrip("/") + SERVED_PATH, content=body, headers=headers, timeout=10)
                r.raise_for_status()
            except Exception as exc:
                for key, n in part:  # retried with the next report while still useful
                    if key[2] >= now - KEEP_S:
                        counts = self._counts.setdefault(key, [0, 0])
                        counts[0] += n[0]
                        counts[1] += n[1]
                log.warning("reporting served counts failed: %s", exc)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(REPORT_S)
            await self.report()
