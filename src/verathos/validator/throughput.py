"""The validator's side of the throughput audit.

Every validator follows the public test schedule and holds its canaries while a test runs. The subnet owner's validator
is the auditor: it runs the tests and serves signed verdicts (``validator.service``); every other validator fetches them
over TLS, accepting the auditor's certificate by the pin the catalog names. A validator applies a piece of evidence only
after checking the miner's receipt in it, and, when it runs a reference engine for the model, only if its own reference
also proves the answer wrong.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import bittensor as bt
import httpx

from verathos import catalog
from verathos.miner.engine import Engine
from verathos.protocol import tls
from verathos.protocol.manifest import Endpoint
from verathos.protocol.receipts import issued_receipt, parse_response, signed_answers
from verathos.throughput import feed
from verathos.throughput.auditor import Auditor, Slot
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.policy import BLOCK_TIME_S, AuditPolicy
from verathos.validator import chain as validator_chain
from verathos.validator.config import ReferenceEngine
from verathos.validator.discovery import DiscoveredMiner, Identities, ManifestCache, Neuron, discover

log = logging.getLogger(__name__)
SLOT_REFRESH_S = 120.0
MANIFEST_HOLD_EPOCHS = 3  # how long a manifest accepted before stands in while its axon is out
VERDICTS_RETRY_S = 30.0
VERDICTS_WAIT_S = 300.0


def testable(e: Endpoint) -> bool:
    """Whether the auditor tests the endpoint: open, with a declared capacity."""
    return e.tier == "open" and e.capacity_rps > 0


def slots(miners: list[DiscoveredMiner]) -> list[Slot]:
    """Every endpoint the auditor tests, with its machine's pin."""
    return [Slot(e.hotkey, e.endpoint.endpoint_id, e.url, e.endpoint.model_id, e.endpoint.capacity_rps, e.tls)
            for m in miners for e in m.endpoints() if testable(e.endpoint)]


class Throughput:
    def __init__(self, *, netuid: int, policy: AuditPolicy, auditor_hotkey: str, chain: SubtensorChainView,
                 auditor: Auditor | None = None, follower: ScheduleFollower | None = None,
                 auditor_pin: Callable[[], str] | None = None) -> None:
        """``auditor_pin``: the pin of the auditor's certificate, as the catalog names it (``Catalog.auditor_pin``)."""
        self.netuid, self.policy, self.auditor_hotkey, self.chain = netuid, policy, auditor_hotkey, chain
        self.auditor, self.follower, self.auditor_pin = auditor, follower, auditor_pin
        # what the validator knows of the miners, shared by its canary audit and the auditor's refresh
        self.identities = Identities()
        self.manifests = ManifestCache(MANIFEST_HOLD_EPOCHS * policy.epoch_blocks * BLOCK_TIME_S)
        self._last: feed.Verdicts | None = None

    @classmethod
    def start(cls, *, network: str, netuid: int, current: Callable[[], catalog.Catalog],
              st: bt.Subtensor, wallet: bt.Wallet, references: dict[str, ReferenceEngine],
              screens: dict[str, ReferenceEngine], state_dir: Path) -> Throughput:
        """The auditor role if this validator's hotkey is the subnet owner's, else a schedule follower."""
        policy = current().audit_policy()
        auditor_hotkey = st.get_subnet_owner_hotkey(netuid)
        chain = SubtensorChainView(network, netuid)
        if wallet.hotkey.ss58_address != auditor_hotkey:
            follower = ScheduleFollower(chain, policy)
            follower.start()
            log.info("throughput tests are run by %s; applying its verdicts", auditor_hotkey)
            return cls(netuid=netuid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain, follower=follower,
                       auditor_pin=lambda: current().auditor_pin())
        auditor = Auditor(netuid=netuid, keypair=wallet.hotkey, chain=chain, policy=policy,
                          catalog=current, references=references, screens=screens,
                          db_path=state_dir / "throughput.sqlite",
                          usage_chain=SubtensorChainView(network, netuid))  # apart: the schedule waits for none of it
        auditor.start()
        log.info("throughput auditor running (%s mode)", policy.mode)
        return cls(netuid=netuid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain, auditor=auditor)

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt the policy of a newly committed catalog."""
        self.policy = policy
        self.manifests.hold_s = MANIFEST_HOLD_EPOCHS * policy.epoch_blocks * BLOCK_TIME_S
        (self.auditor or self.follower).set_policy(policy)

    def holding(self) -> bool:
        """Whether a test runs now (canaries wait)."""
        return (self.auditor or self.follower).holding()

    # ---- the auditor's endpoints ----

    def start_slot_refresh(self, network: str, catalogs: Callable[[], Mapping[str, int]],
                           interval_s: float = SLOT_REFRESH_S) -> None:
        """Re-discover miners every ``interval_s`` so new or changed endpoints are tested within minutes rather than
        from the next epoch (auditor only); ``catalogs`` as ``discover`` takes them."""
        if self.auditor is None:
            return

        def run() -> None:
            st: bt.Subtensor | None = None
            while True:  # at once, then every interval: a restarted auditor tests and routes from the start
                st = self.slot_round(st, network, catalogs)
                time.sleep(interval_s)

        threading.Thread(target=run, name="throughput-slots", daemon=True).start()

    def slot_round(self, st: bt.Subtensor | None, network: str,
                   catalogs: Callable[[], Mapping[str, int]]) -> bt.Subtensor | None:
        """One round of ``start_slot_refresh`` over ``st`` (connecting first without one). -> the connection for the
        next round: None after a failure, so the next round connects again."""
        try:
            st = st or bt.Subtensor(network=network)
            neurons, _ = validator_chain.snapshot(st, self.netuid)
            epoch = st.get_current_block() // self.policy.epoch_blocks
            self.update_slots(asyncio.run(self._refresh(neurons, catalogs(), epoch)), neurons, epoch)
            return st
        except Exception:
            log.exception("throughput slot refresh failed")
            return None

    async def _refresh(self, neurons: list[Neuron], catalogs: Mapping[str, int],
                       epoch: int) -> list[DiscoveredMiner]:
        async with tls.client() as client:
            return await self.refresh(neurons, catalogs, epoch, client)

    async def refresh(self, neurons: list[Neuron], catalogs: Mapping[str, int], epoch: int,
                      client: httpx.AsyncClient) -> list[DiscoveredMiner]:
        """Discover the miners, and challenge the endpoints a test would target that were not challenged at their URL
        this epoch yet (auditor only): the epoch's canary audit tries again, within the miners' request budget."""
        miners = await discover(neurons, self.netuid, client, catalogs, self.manifests)
        await self.identities.verify([e for m in miners for e in m.endpoints() if testable(e.endpoint)], epoch,
                                     self.auditor.keypair, client, retry=False)
        return miners

    def update_slots(self, miners: list[DiscoveredMiner], neurons: list[Neuron], epoch: int) -> None:
        """Replace the tested endpoints with the discovered ones (``manifests`` holds a hotkey's while its axon is
        out) that proved control at their URL in ``epoch`` or the one before."""
        if self.auditor is None:
            return
        if not miners and any(n.commitment for n in neurons):
            log.warning("no miner could be discovered; keeping the endpoints known before")
            return  # a failed discovery must not empty the tests and the gateways' routes
        self.auditor.update_slots([s for s in slots(miners) if self.identities.verified(s.key, s.where, epoch - 1)])

    # ---- verdicts ----

    async def verdicts(self, client: httpx.AsyncClient, epoch: int) -> feed.Verdicts | None:
        """The verdicts for scoring epoch ``epoch``, waiting up to ``VERDICTS_WAIT_S`` while its tests settle. If
        they stay unavailable, the last verdicts fetched while they still cover a test inside ``credit_ttl_epochs``;
        else None."""
        deadline = time.monotonic() + VERDICTS_WAIT_S
        while True:
            try:
                verdicts = await self._fetch(client, epoch)
            except feed.VerdictsUnavailable as exc:
                if time.monotonic() + VERDICTS_RETRY_S > deadline:
                    log.warning("throughput verdicts for epoch %d unavailable: %s", epoch, exc)
                    break
                await asyncio.sleep(VERDICTS_RETRY_S)
            else:
                self._last = verdicts
                return verdicts
        if self._last is not None and self._last.epoch >= epoch - self.policy.credit_ttl_epochs + 1:
            log.warning("using the throughput verdicts of epoch %d", self._last.epoch)
            return self._last
        return None

    async def _fetch(self, client: httpx.AsyncClient, epoch: int) -> feed.Verdicts:
        if self.auditor is not None:
            signed = self.auditor.verdicts(epoch)
            if signed is None:
                raise feed.VerdictsUnavailable("the epoch's tests are not settled yet")
            return signed.verdicts
        try:
            url = await asyncio.to_thread(self.chain.auditor_url)
            pin = self.auditor_pin() if self.auditor_pin is not None else None
            return await feed.fetch(client, url, pin, auditor_hotkey=self.auditor_hotkey, netuid=self.netuid,
                                    epoch=epoch)
        except feed.VerdictsUnavailable:
            raise
        except Exception as exc:  # chain or transport trouble is an unavailable feed, never a failed epoch
            raise feed.VerdictsUnavailable(f"{type(exc).__name__}: {exc}") from None

    async def proven_invalid(self, evidence: list[feed.Evidence], references: dict[str, Engine],
                             cat: catalog.Catalog, budget_s: float | None = None) -> list[feed.Evidence]:
        """The evidence this validator accepts. From a test: the miner's receipt binds the request and the answers,
        and the validator's own reference (if it has one for the model and it answers) also proves the answers wrong.
        From a sampled user request (not published): the miner's receipt shows it answered one of the catalog's
        gateways, and the auditor's word stands for the wrongness, as it does for capacity. Rechecks run within
        ``budget_s``; past it, as when the reference is unavailable, the auditor's verdict applies."""
        accepted = []
        deadline = None if budget_s is None else time.monotonic() + budget_s
        for e in evidence:
            model = cat.get(e.model_id)
            response = parse_response(e.response)
            if e.from_gateway:
                receipt = None if response is None or model is None else issued_receipt(
                    response.get("receipt"), request_id=e.request_id, hotkey=e.hotkey, endpoint_id=e.endpoint_id,
                    model_id=e.model_id, callers=set(cat.gateway_hotkeys()), netuid=self.netuid)
                if receipt is None:
                    log.warning("evidence against %s/%s rejected: no receipt issued to a gateway", e.hotkey,
                                e.endpoint_id)
                else:
                    accepted.append(e)
                continue
            answers = None if response is None else signed_answers(
                response, e.request, request_id=e.request_id, hotkey=e.hotkey, endpoint_id=e.endpoint_id,
                model_id=e.model_id, callers={self.auditor_hotkey}, netuid=self.netuid)
            if answers is None or model is None:
                log.warning("evidence against %s/%s rejected: %s", e.hotkey, e.endpoint_id,
                            "unknown model" if model is None else "the receipt does not bind it")
                continue
            if self.auditor is None and e.model_id in references:
                left = None if deadline is None else deadline - time.monotonic()
                try:
                    if left is not None and left <= 0:
                        raise feed.ReferenceUnavailable("no time left this epoch")
                    recheck = feed.answer_fault(e.request, answers, references[e.model_id], model.tolerance,
                                                feed.claimed_tokens(response))
                    fault = await asyncio.wait_for(recheck, left)
                except (feed.ReferenceUnavailable, TimeoutError) as exc:
                    log.warning("evidence against %s/%s not rechecked, reference unavailable (%s); applying the "
                                "auditor's verdict", e.hotkey, e.endpoint_id, exc)
                    fault = e.reason
                if fault is None:
                    log.warning("evidence against %s/%s rejected: this validator's reference accepts the answers",
                                e.hotkey, e.endpoint_id)
                    continue
            accepted.append(e)
        return accepted
