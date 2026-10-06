"""The validator's side of the capacity audit.

Every validator drafts the audit's slots from the manifests it discovers. The validator whose hotkey is the
policy's auditor (by default the subnet owner) runs the audit: it times receipts, verifies proofs and gates
endpoints, and serves the ingest and verdicts (``validator.service``). Every other validator tracks audit holds
from public chain data and applies the auditor's signed verdicts.
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

from verathos.capacity.auditor import Auditor
from verathos.capacity.chain import SubtensorChainView
from verathos.capacity.core import AuditPolicy, Slot, gpu_support_status
from verathos.capacity.holds import HoldTracker
from verathos.capacity.verdicts import fetch_verdicts
from verathos.validator import chain as validator_chain
from verathos.validator.discovery import DiscoveredMiner, Neuron, discover

log = logging.getLogger(__name__)
SLOT_REFRESH_S = 120.0


def capacity_slots(miners: list[DiscoveredMiner], netuid: int) -> list[Slot]:
    """Every open CUDA endpoint with declared hardware: the slots the capacity audit drafts from."""
    return [Slot(netuid, m.hotkey, e.endpoint_id, e.url, e.model_id, e.hardware.gpu_name, e.hardware.vram_gb)
            for m in miners for e in m.manifest.endpoints
            if e.tier == "open" and e.hardware.kind == "cuda" and e.hardware.gpu_name and e.hardware.vram_gb]


def unauditable_endpoints(miners: list[DiscoveredMiner], policy: AuditPolicy) -> set[str]:
    """Open endpoints the capacity audit cannot draft: not CUDA, or no calibrated GPU class declared."""
    return {f"{m.hotkey}/{e.endpoint_id}" for m in miners for e in m.manifest.endpoints if e.tier == "open"
            and (e.hardware.kind != "cuda"
                 or not gpu_support_status(e.hardware.gpu_name or "", e.hardware.vram_gb or 0, policy)[0])}


class Capacity:
    def __init__(self, *, netuid: int, policy: AuditPolicy, auditor_hotkey: str, chain: SubtensorChainView,
                 auditor: Auditor | None = None, holds: HoldTracker | None = None) -> None:
        self.netuid, self.policy, self.auditor_hotkey, self.chain = netuid, policy, auditor_hotkey, chain
        self.auditor, self.holds = auditor, holds
        self._known: dict[str, tuple[str | None, list[Slot]]] = {}
        self._known_lock = threading.Lock()

    @classmethod
    def start(cls, *, network: str, netuid: int, policy: AuditPolicy, st: bt.Subtensor, wallet: bt.Wallet,
              state_dir: Path) -> Capacity:
        """The auditor role if this validator's hotkey is the auditor, else the hold tracker."""
        auditor_hotkey = policy.auditor_hotkey or st.get_subnet_owner_hotkey(netuid)
        chain = SubtensorChainView(network, netuid, auditor_hotkey)
        if wallet.hotkey.ss58_address != auditor_hotkey:
            holds = HoldTracker(netuid=netuid, policy=policy, chain=chain)
            holds.start()
            log.info("capacity audits are evaluated by %s; applying its verdicts", auditor_hotkey)
            return cls(netuid=netuid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain, holds=holds)
        auditor = Auditor(netuid=netuid, policy=policy, chain=chain, db_path=state_dir / "capacity.sqlite")
        auditor.start()
        log.info("capacity auditor running (%s mode)", policy.mode)
        return cls(netuid=netuid, policy=policy, auditor_hotkey=auditor_hotkey, chain=chain, auditor=auditor)

    def set_policy(self, policy: AuditPolicy) -> None:
        """Adopt the policy of a newly committed catalog. A change of auditor takes effect at the next restart."""
        if (policy.auditor_hotkey or self.auditor_hotkey) != self.auditor_hotkey and policy.auditor_hotkey:
            log.warning("the catalog names a new capacity auditor %s; restart to adopt it", policy.auditor_hotkey)
        self.policy = policy
        (self.auditor or self.holds).set_policy(policy)

    def start_slot_refresh(self, network: str, mechid: int, catalog_digests: Callable[[], Collection[str]],
                           interval_s: float = SLOT_REFRESH_S) -> None:
        """Re-discover miners every ``interval_s`` so new or changed endpoints join the audit within minutes rather
        than at the next epoch."""

        def run() -> None:
            st = bt.Subtensor(network=network)
            while True:
                time.sleep(interval_s)
                try:
                    neurons, _ = validator_chain.snapshot(st, self.netuid)
                    self.update_slots(asyncio.run(self._discover(neurons, mechid, catalog_digests())), neurons)
                except Exception:
                    log.exception("capacity slot refresh failed")

        threading.Thread(target=run, name="capacity-slots", daemon=True).start()

    async def _discover(self, neurons: list, mechid: int, catalog_digests: Collection[str]) -> list[DiscoveredMiner]:
        async with httpx.AsyncClient() as client:
            return await discover(neurons, self.netuid, mechid, client, catalog_digests)

    def update_slots(self, miners: list[DiscoveredMiner], neurons: list[Neuron]) -> None:
        """Replace the audit's slots with the discovered ones. A hotkey whose manifest could not be fetched this
        time keeps its last slots while its commitment is unchanged, so a failed fetch never makes the auditor and
        the miner disagree on a draft."""
        commitments = {n.hotkey: n.commitment for n in neurons}
        known = {m.hotkey: (commitments.get(m.hotkey), capacity_slots([m], self.netuid)) for m in miners}
        with self._known_lock:
            for hotkey, (commitment, slots) in self._known.items():
                if hotkey not in known and commitment is not None and commitments.get(hotkey) == commitment:
                    known[hotkey] = (commitment, slots)
            self._known = known
        (self.auditor or self.holds).update_slots([s for _, slots in known.values() for s in slots])

    def drained(self, hotkey: str, endpoint_id: str) -> bool:
        return (self.auditor or self.holds).drained(hotkey, endpoint_id)

    async def gated(self, client: httpx.AsyncClient, epoch: int) -> dict[tuple[str, str], str]:
        """Endpoints gated in audit epoch ``epoch``: the auditor's own evaluation, or its signed verdicts."""
        if self.auditor is not None:
            self.auditor.prune(epoch)
            return self.auditor.gated(epoch)
        if self.policy.mode != "score_gate":
            return {}
        return await fetch_verdicts(client, self.chain.auditor_url(), auditor_hotkey=self.auditor_hotkey,
                                    netuid=self.netuid, epoch=epoch)
