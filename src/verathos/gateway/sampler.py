"""The exchanges the auditor checks (``protocol.gateway``).

Per endpoint and window, one exchange drawn uniformly from the window's answered ones: a reservoir of one, filled
after each answer is in, so a miner cannot tell which answers will be checked. Signed malformed answers are proof on
their own: they always go, and a failed post keeps them for the next window. At the end of each window the picks
are posted to the auditor in batches within its size limit, signed by the gateway.
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
from verathos.protocol.gateway import MAX_SAMPLES_BYTES, MAX_SAMPLES_PER_POST, SAMPLES_PATH, GatewaySample
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
MAX_PROOFS = 256
BATCH_BYTES = MAX_SAMPLES_BYTES - (64 << 10)  # room for the envelope


def batches(samples: list[str]) -> list[list[str]]:
    """Serialized samples grouped into posts within the auditor's count and size limits; a sample too large for any
    post is left out."""
    out: list[list[str]] = []
    size = 0
    for sample in samples:
        if len(sample) > BATCH_BYTES:
            continue
        if not out or len(out[-1]) >= MAX_SAMPLES_PER_POST or size + len(sample) + 1 > BATCH_BYTES:
            out.append([])
            size = 0
        out[-1].append(sample)
        size += len(sample) + 1
    return out


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

    def pending(self) -> dict[str, int]:
        return {"picked": len(self._picked), "proofs": len(self._proofs)}

    def proof(self, exchange: Exchange) -> None:
        if len(self._proofs) < MAX_PROOFS:
            self._proofs.append(exchange)

    async def flush(self) -> None:
        proofs, picked = self._proofs, list(self._picked.values())
        self._seen, self._picked, self._proofs = {}, {}, []
        url = await self.auditor_url() if proofs or picked else None
        for exchanges, is_proof in ((proofs, True), (picked, False)):
            if not exchanges:
                continue
            serialized = [GatewaySample(hotkey=e.route.hotkey, endpoint_id=e.route.endpoint_id,
                                        model_id=e.route.model_id, request=e.request,
                                        response=e.response).model_dump_json() for e in exchanges]
            failed = 0
            for batch in batches(serialized):
                if not await self._post(url, batch):
                    failed += len(batch)
            if failed and is_proof:  # proofs are kept for the next window
                self._proofs = (exchanges + self._proofs)[:MAX_PROOFS]

    async def _post(self, url: str | None, batch: list[str]) -> bool:
        if not url:
            metrics.SAMPLES.labels("failed").inc(len(batch))
            log.warning("auditor unknown: %d samples not submitted", len(batch))
            return False
        body = ('{"samples": [' + ",".join(batch) + "]}").encode()
        headers = {"Content-Type": "application/json",
                   **sign_request(self.keypair, "POST", SAMPLES_PATH, body, self.auditor_hotkey)}
        try:
            r = await self.client.post(url.rstrip("/") + SAMPLES_PATH, content=body, headers=headers, timeout=30.0)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            metrics.SAMPLES.labels("failed").inc(len(batch))
            log.warning("submitting %d samples to the auditor failed: %s", len(batch), exc)
            return False
        metrics.SAMPLES.labels("submitted").inc(len(batch))
        try:
            metrics.SAMPLES.labels("kept").inc(int(r.json().get("kept", 0)))
        except ValueError:
            pass
        return True

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.window_s)
            try:
                await self.flush()
            except Exception:
                log.exception("sampling window failed")
