"""The exchanges the auditor checks (``protocol.gateway``).

Per endpoint and window, one exchange drawn uniformly from the window's answered ones without images and one from
those with images: reservoirs of one, filled after each answer is in, so a miner cannot tell which answers will be
checked, and answers with images reach the auditor however small their share. The picks with images held at once are
bounded (``MEDIA_PICKS_BYTES``). Signed malformed answers are proof on their own: they always go, and a failed post
keeps them for the next window. At the end of each window the picks are posted to the auditor in batches within its
size limit, signed by the gateway.

Throughput tests: the auditor counts what an endpoint answers users during a test only once it has checked one of
those answers, since a miner knows when a test runs. The gateway follows the public test schedule
(``TestFollower``): from when it sees a test's start block until its traffic ends, it draws per endpoint one of the
exchanges sent then, the same way, and posts it marked with the test's id once the test's sessions end
(``AuditPolicy.session_s`` after the start), again on failure for ``TEST_SAMPLE_KEEP_S``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from bittensor_wallet import Keypair

from verathos.gateway import metrics
from verathos.gateway.logs import Throttle
from verathos.gateway.upstream import Exchange
from verathos.protocol.gateway import MAX_SAMPLES_BYTES, MAX_SAMPLES_PER_POST, SAMPLES_PATH, GatewaySample
from verathos.protocol.http import post_idempotent
from verathos.protocol.requests import sign_request
from verathos.protocol.tls import Service
from verathos.throughput.follow import POLL_S, ScheduleFollower
from verathos.throughput.policy import BLOCK_TIME_S, AuditPolicy, ScheduledTest

log = logging.getLogger(__name__)
MAX_PROOFS = 256
MEDIA_PICKS_BYTES = 2 << 30  # the images of the window's picks held at once, at most: a pick beyond is not taken
BATCH_BYTES = MAX_SAMPLES_BYTES - (64 << 10)  # room for the envelope
TEST_SAMPLE_KEEP_S = 300.0  # a test's picks are posted again on failure for this long after its sessions end
_throttled = Throttle()


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


@dataclass
class _Test:
    """A throughput test's traffic as this gateway saw it, and its picks: endpoint -> [answers seen, the pick]."""

    start: float
    end: float
    post_at: float
    picks: dict[str, list] = field(default_factory=dict)


class Sampler:
    def __init__(self, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor: Callable[[], Awaitable[Service | None]], auditor_hotkey: str, window_s: float,
                 rng: random.Random | None = None, clock: Callable[[], float] = time.time) -> None:
        self.keypair, self.client, self.auditor, self.auditor_hotkey = keypair, client, auditor, auditor_hotkey
        self.window_s, self.rng, self.clock = window_s, rng or random.Random(), clock
        self._seen: dict[tuple[str, bool], int] = {}  # (endpoint, with images) -> answers seen in the window
        self._picked: dict[tuple[str, bool], Exchange] = {}
        self._media_bytes = 0  # of the window's picks with images
        self._proofs: list[Exchange] = []
        self._tests: dict[str, _Test] = {}  # test id -> its traffic and picks

    def offer(self, exchange: Exchange) -> None:
        key = f"{exchange.route.hotkey}/{exchange.route.endpoint_id}"
        pool = (key, bool(exchange.request.get("images")))
        seen = self._seen[pool] = self._seen.get(pool, 0) + 1
        if self.rng.random() * seen < 1:
            self._pick(pool, exchange)
        sent = self.clock() - exchange.seconds
        for test in self._tests.values():
            if test.start <= sent <= test.end:
                pick = test.picks.setdefault(key, [0, None])
                pick[0] += 1
                if self.rng.random() * pick[0] < 1:
                    pick[1] = exchange

    def _pick(self, pool: tuple[str, bool], exchange: Exchange) -> None:
        if pool[1]:
            held = self._media_bytes - _image_bytes(self._picked.get(pool)) + _image_bytes(exchange)
            if held > MEDIA_PICKS_BYTES:
                if times := _throttled("media picks"):
                    log.warning("an answer with images is not sampled: the window's picks hold %d MiB of images "
                                "(%d times since last logged)", self._media_bytes >> 20, times)
                return
            self._media_bytes = held
        self._picked[pool] = exchange

    def begin_test(self, test_id: str, start: float, traffic_s: float, session_s: float) -> None:
        """A test whose traffic started at ``start`` (wall clock) and lasts ``traffic_s``; its sessions end
        ``session_s`` after the start."""
        if test_id not in self._tests:
            self._tests[test_id] = _Test(start, start + traffic_s, start + session_s)

    async def post_tests(self) -> None:
        """Post the picks of the tests whose sessions ended; a test whose post fails is kept for the next call, for
        ``TEST_SAMPLE_KEEP_S``."""
        now = self.clock()
        for test_id, test in list(self._tests.items()):
            if now < test.post_at:
                continue
            picks = [e for _, e in test.picks.values()]
            try:
                auditor = await self.auditor()
            except Exception as exc:
                log.warning("the auditor's address is unknown: %s", exc)
                auditor = None
            serialized = [GatewaySample(hotkey=e.route.hotkey, endpoint_id=e.route.endpoint_id,
                                        model_id=e.route.model_id, request=e.request, response=e.response,
                                        test_id=test_id).model_dump_json() for e in picks]
            sent = [await self._post(auditor, batch) for batch in batches(serialized)]
            if all(sent) or now > test.post_at + TEST_SAMPLE_KEEP_S:
                if not all(sent):
                    log.warning("the users' answers in test %s were not submitted: %d picks dropped", test_id[:12],
                                len(picks))
                del self._tests[test_id]

    def pending(self) -> dict[str, int]:
        return {"picked": len(self._picked), "proofs": len(self._proofs),
                "test_picks": sum(len(t.picks) for t in self._tests.values())}

    def proof(self, exchange: Exchange) -> None:
        if len(self._proofs) < MAX_PROOFS:
            self._proofs.append(exchange)

    async def flush(self) -> None:
        proofs, picked = self._proofs, list(self._picked.values())
        self._seen, self._picked, self._proofs, self._media_bytes = {}, {}, [], 0
        if not (proofs or picked):
            return
        sent_proofs = False
        try:
            try:
                auditor = await self.auditor()
            except Exception as exc:
                log.warning("the auditor's address is unknown: %s", exc)
                auditor = None
            for exchanges, is_proof in ((proofs, True), (picked, False)):
                if not exchanges:
                    continue
                serialized = [GatewaySample(hotkey=e.route.hotkey, endpoint_id=e.route.endpoint_id,
                                            model_id=e.route.model_id, request=e.request,
                                            response=e.response).model_dump_json() for e in exchanges]
                failed = 0
                for batch in batches(serialized):
                    if not await self._post(auditor, batch):
                        failed += len(batch)
                if is_proof:
                    sent_proofs = not failed
        finally:
            if proofs and not sent_proofs:  # proofs are kept for the next window, whatever stopped this one
                self._proofs = (proofs + self._proofs)[:MAX_PROOFS]

    async def _post(self, auditor: Service | None, batch: list[str]) -> bool:
        if auditor is None:
            metrics.SAMPLES.labels("failed").inc(len(batch))
            log.warning("auditor unknown: %d samples not submitted", len(batch))
            return False
        body = ('{"samples": [' + ",".join(batch) + "]}").encode()

        def headers() -> dict[str, str]:
            return {"Content-Type": "application/json",
                    **sign_request(self.keypair, "POST", SAMPLES_PATH, body, self.auditor_hotkey)}
        try:  # the auditor keeps a sample once, however often it arrives
            r = await post_idempotent(self.client, auditor.url.rstrip("/") + SAMPLES_PATH, body, headers,
                                      timeout=30.0, tls=auditor.pin)
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


def _image_bytes(exchange: Exchange | None) -> int:
    """The base64 of an exchange's images."""
    images = exchange.request.get("images") if exchange is not None else None
    return sum(len(i.get("base64", "")) for i in images if isinstance(i, dict)) if images else 0


class TestFollower:
    """Follows the public test schedule on the chain (``ScheduleFollower``) and tells the sampler when a test's
    traffic starts: when the head reaches its start block (dated back by the blocks the head is past it, if it is
    seen late)."""

    def __init__(self, sampler: Sampler, follower: ScheduleFollower, policy: Callable[[], AuditPolicy | None]) -> None:
        self.sampler, self.follower, self.policy = sampler, follower, policy
        self._coming: dict[str, ScheduledTest] = {}

    async def step(self) -> None:
        policy = self.policy()
        if policy is None:
            return
        if policy != self.follower.policy:
            self.follower.set_policy(policy)
        for test in await asyncio.to_thread(self.follower.poll):
            self._coming[test.test_id] = test
        head, now = self.follower.head, self.sampler.clock()
        for test_id, test in list(self._coming.items()):
            if head < test.start_block:
                continue
            del self._coming[test_id]
            if head - test.start_block <= policy.traffic_blocks:  # seen while its traffic runs
                start = now - (head - test.start_block) * BLOCK_TIME_S
                self.sampler.begin_test(test_id, start, 2 * policy.warmup_s + policy.test_s + POLL_S,
                                        policy.session_s)
        await self.sampler.post_tests()

    async def run(self) -> None:
        while True:
            try:
                await self.step()
            except Exception as exc:
                log.warning("following the test schedule failed: %s: %s", type(exc).__name__, exc)
            await asyncio.sleep(POLL_S)
