"""The exchanges the auditor checks (``protocol.gateway``).

Per endpoint and window, one exchange drawn uniformly from the window's answered ones: a reservoir of one, filled
after each answer is in, so a miner cannot tell which answers will be checked. Signed malformed answers are proof on
their own and always go. At the end of each window the picks are posted to the auditor, signed by the gateway.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

import httpx
from bittensor_wallet import Keypair

from verathos.gateway import metrics
from verathos.gateway.upstream import Exchange
from verathos.protocol.gateway import MAX_SAMPLES_PER_POST, SAMPLES_PATH, GatewaySample, GatewaySamples
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
MAX_PROOFS = 256


class Sampler:
    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor_url: Callable[[], Awaitable[str | None]], auditor_hotkey: str, window_s: float,
                 rng: random.Random | None = None) -> None:
        self.keypair, self.client, self.auditor_url, self.auditor_hotkey = keypair, client, auditor_url, auditor_hotkey
        self.window_s, self.rng = window_s, rng or random.Random()
        self._seen: dict[str, int] = {}
        self._picked: dict[str, Exchange] = {}
        self._proofs: list[Exchange] = []

    def offer(self, exchange: Exchange) -> None:
        key = f"{exchange.route.hotkey}/{exchange.route.endpoint_id}"
        seen = self._seen[key] = self._seen.get(key, 0) + 1
        if self.rng.random() * seen < 1:
            self._picked[key] = exchange

    def proof(self, exchange: Exchange) -> None:
        if len(self._proofs) < MAX_PROOFS:
            self._proofs.append(exchange)

    async def flush(self) -> None:
        exchanges = self._proofs + list(self._picked.values())
        self._seen, self._picked, self._proofs = {}, {}, []
        if not exchanges:
            return
        url = await self.auditor_url()
        if not url:
            metrics.SAMPLES.labels("failed").inc(len(exchanges))
            log.warning("auditor unknown: %d samples dropped", len(exchanges))
            return
        samples = [GatewaySample(hotkey=e.route.hotkey, endpoint_id=e.route.endpoint_id, model_id=e.route.model_id,
                                 request=e.request, response=e.response) for e in exchanges]
        for i in range(0, len(samples), MAX_SAMPLES_PER_POST):
            batch = samples[i:i + MAX_SAMPLES_PER_POST]
            body = GatewaySamples(samples=batch).model_dump_json().encode()
            headers = {"Content-Type": "application/json",
                       **sign_request(self.keypair, "POST", SAMPLES_PATH, body, self.auditor_hotkey)}
            try:
                r = await self.client.post(url.rstrip("/") + SAMPLES_PATH, content=body, headers=headers,
                                           timeout=30.0)
                r.raise_for_status()
                metrics.SAMPLES.labels("submitted").inc(len(batch))
                metrics.SAMPLES.labels("kept").inc(int(r.json().get("kept", 0)))
            except (httpx.HTTPError, ValueError) as exc:
                metrics.SAMPLES.labels("failed").inc(len(batch))
                log.warning("submitting %d samples to the auditor failed: %s", len(batch), exc)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.window_s)
            try:
                await self.flush()
            except Exception:
                log.exception("sampling window failed")
