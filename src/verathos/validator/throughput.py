"""The validator's side of the throughput audit.

Every validator follows the public test schedule and holds its canaries while a test runs. The validator whose
hotkey is the policy's auditor (by default the subnet owner) runs the tests and serves signed verdicts
(``validator.service``); every other validator fetches them. A validator applies a piece of evidence only after
checking the miner's receipt in it, and, when it runs a reference engine for the model, only if its own reference also
proves the answer wrong.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Collection
from pathlib import Path

import bittensor as bt
import httpx

from verathos import catalog
from verathos.miner.engine import Engine
from verathos.protocol.receipts import parse_response, signed_answers
from verathos.throughput import feed
from verathos.throughput.auditor import Auditor, Slot
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.policy import AuditPolicy
from verathos.validator import chain as validator_chain
from verathos.validator.config import ReferenceEngine
from verathos.validator.discovery import DiscoveredMiner, Neuron, discover

log = logging.getLogger(__name__)
SLOT_REFRESH_S = 120.0
VERDICTS_RETRY_S = 30.0
VERDICTS_WAIT_S = 300.0


def slots(miners: list[DiscoveredMiner]) -> list[Slot]:
    """Every open endpoint with a declared capacity: what the auditor tests."""
    return [Slot(m.hotkey, e.endpoint_id, e.url, e.model_id, e.capacity_rps)
            for m in miners for e in m.manifest.endpoints if e.tier == "open" and e.capacity_rps > 0]


class Throughput:
    def __init__(self, *, netuid: int, mechid: int, policy: AuditPolicy, auditor_hotkey: str,
                 chain: SubtensorChainView, auditor: Auditor | None = None,
                 follower: ScheduleFollower | None = None) -> None:
        self.netuid, self.mechid, self.policy, self.auditor_hotkey, self.chain = (netuid, mechid, policy,
                                                                                  auditor_hotkey, chain)
        self.auditor, self.follower = auditor, follower
        self._known: dict[str, tuple[str | None, list[Slot]]] = {}
        self._last: feed.Verdicts | None = None

    @classmethod
    def start(cls, *, network: str, netuid: int, mechid: int, current: Callable[[], catalog.Catalog],
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
            return cls(netuid=netuid, mechid=mechid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain,
                       follower=follower)
        auditor = Auditor(netuid=netuid, mechid=mechid, keypair=wallet.hotkey, chain=chain, policy=policy,
                          catalog=current, references=references, screens=screens,
                          db_path=state_dir / "throughput.sqlite")
        auditor.start()
        log.info("throughput auditor running (%s mode)", policy.mode)
        return cls(netuid=netuid, mechid=mechid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain,
                   auditor=auditor)

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt the policy of a newly committed catalog."""
        self.policy = policy
        (self.auditor or self.follower).set_policy(policy)

    def holding(self) -> bool:
        """Whether a test runs now (canaries wait)."""
        return (self.auditor or self.follower).holding()

    # ---- the auditor's endpoints ----

    def start_slot_refresh(self, network: str, catalog_digests: Callable[[], Collection[str]],
                           interval_s: float = SLOT_REFRESH_S) -> None:
        """Re-discover miners every ``interval_s`` so new or changed endpoints are tested within minutes rather than
        from the next epoch (auditor only)."""
        if self.auditor is None:
            return

        def run() -> None:
            st = bt.Subtensor(network=network)
            while True:
                time.sleep(interval_s)
                try:
                    neurons, _ = validator_chain.snapshot(st, self.netuid)
                    self.update_slots(asyncio.run(self._discover(neurons, catalog_digests())), neurons)
                except Exception:
                    log.exception("throughput slot refresh failed")

        threading.Thread(target=run, name="throughput-slots", daemon=True).start()

    async def _discover(self, neurons: list[Neuron], catalog_digests: Collection[str]) -> list[DiscoveredMiner]:
        async with httpx.AsyncClient() as client:
            return await discover(neurons, self.netuid, self.mechid, client, catalog_digests)

    def update_slots(self, miners: list[DiscoveredMiner], neurons: list[Neuron]) -> None:
        """Replace the tested endpoints with the discovered ones. A hotkey whose manifest could not be fetched this
        time keeps its endpoints while its commitment is unchanged, so one failed fetch does not drop it from a
        test."""
        if self.auditor is None:
            return
        commitments = {n.hotkey: n.commitment for n in neurons}
        known = {m.hotkey: (commitments.get(m.hotkey), slots([m])) for m in miners}
        for hotkey, (commitment, kept) in self._known.items():
            if hotkey not in known and commitment is not None and commitments.get(hotkey) == commitment:
                known[hotkey] = (commitment, kept)
        self._known = known
        self.auditor.update_slots([s for _, kept in known.values() for s in kept])

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
            return await feed.fetch(client, url, auditor_hotkey=self.auditor_hotkey, netuid=self.netuid, epoch=epoch)
        except feed.VerdictsUnavailable:
            raise
        except Exception as exc:  # chain or transport trouble is an unavailable feed, never a failed epoch
            raise feed.VerdictsUnavailable(f"{type(exc).__name__}: {exc}") from None

    async def proven_invalid(self, evidence: list[feed.Evidence], references: dict[str, Engine],
                             cat: catalog.Catalog) -> list[feed.Evidence]:
        """The evidence this validator accepts: the miner's receipt binds the request and the answers, and the
        validator's own reference (if it has one for the model and it answers) also proves the answers wrong."""
        accepted = []
        for e in evidence:
            model = cat.get(e.model_id)
            response = parse_response(e.response)
            # A test answer was issued to the auditor, a sampled gateway answer to one of the catalog's gateways.
            callers = set(cat.gateways) if e.from_gateway else {self.auditor_hotkey}
            answers = None if response is None else signed_answers(
                response, e.request, request_id=e.request_id, hotkey=e.hotkey, endpoint_id=e.endpoint_id,
                model_id=e.model_id, callers=callers, netuid=self.netuid, mechid=self.mechid)
            if answers is None or model is None:
                log.warning("evidence against %s/%s rejected: %s", e.hotkey, e.endpoint_id,
                            "unknown model" if model is None else "the receipt does not bind it")
                continue
            if self.auditor is None and e.model_id in references:
                try:
                    fault = await feed.answer_fault(e.request, answers, references[e.model_id], model.tolerance)
                except feed.ReferenceUnavailable as exc:
                    log.warning("evidence against %s/%s not rechecked, reference unavailable (%s); applying the "
                                "auditor's verdict", e.hotkey, e.endpoint_id, exc)
                    fault = e.reason
                if fault is None:
                    log.warning("evidence against %s/%s rejected: this validator's reference accepts the answers",
                                e.hotkey, e.endpoint_id)
                    continue
            accepted.append(e)
        return accepted
