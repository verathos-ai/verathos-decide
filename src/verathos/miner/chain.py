"""Miner chain operations (bittensor SDK 10.x): publish the axon and the manifest commitment, and know which
callers are permitted."""

from __future__ import annotations

import logging
import threading
import time

import bittensor as bt

from verathos.miner.config import EndpointConfig, MinerConfig
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value
from verathos.subnet.chain import serve_axon

log = logging.getLogger(__name__)


def build_manifest(config: MinerConfig, endpoints: list[EndpointConfig], capacity_rps: dict[str, float], keypair,
                   catalog_digest: str) -> SignedManifest:
    """Sign a manifest for ``endpoints`` with their declared capacities. The same configuration, measurements and
    catalog always give the same manifest, so a restart does not change the on-chain commitment."""
    listed = [
        Endpoint(endpoint_id=e.endpoint_id, url=config.endpoint_url(e.endpoint_id), model_id=e.model_id, tier=e.tier,
                 capacity_rps=capacity_rps[e.endpoint_id])
        for e in endpoints
    ]
    manifest = Manifest(netuid=config.netuid, mechid=config.mechid, hotkey=keypair.ss58_address,
                        catalog_digest=catalog_digest, endpoints=listed)
    return SignedManifest.create(manifest, keypair)


def publish(subtensor: bt.Subtensor, wallet: bt.Wallet, config: MinerConfig, signed: SignedManifest) -> None:
    """Point the axon at this service and commit the manifest digest; each call is made only if it changes state."""
    uid = serve_axon(subtensor, wallet, config.netuid, config.public_host, config.port)
    value = commitment_value(signed.manifest.digest())
    if subtensor.get_commitment(config.netuid, uid) != value:
        result = subtensor.set_commitment(wallet, config.netuid, value)
        if not getattr(result, "success", False):
            raise RuntimeError(f"set_commitment failed: {getattr(result, 'message', result)}")
        log.info("manifest committed on chain (%s)", value.rsplit(":", 1)[1][:12])
    else:
        log.info("manifest already committed on chain")


class PermitAuthorizer:
    """Allows validators with a validator permit plus configured gateway hotkeys; refreshed in the background over
    its own chain connection."""

    def __init__(self, network: str, netuid: int, extra: list[str], refresh_s: float = 600.0) -> None:
        self._subtensor, self._netuid, self._extra = bt.Subtensor(network=network), netuid, set(extra)
        self._allowed: set[str] = set(extra)
        self._logged = False
        self._refresh_s = refresh_s
        self.refresh()
        threading.Thread(target=self._loop, daemon=True).start()

    def refresh(self) -> None:
        mg = self._subtensor.metagraph(self._netuid, lite=True)
        permitted = {hk for hk, permit in zip(mg.hotkeys, mg.validator_permit) if bool(permit)}
        allowed = permitted | self._extra
        if allowed != self._allowed or not self._logged:
            log.info("accepting requests from %d validators and %d gateway hotkeys", len(permitted), len(self._extra))
            self._logged = True
        self._allowed = allowed

    def _loop(self) -> None:
        while True:
            time.sleep(self._refresh_s)
            try:
                self.refresh()
            except Exception:
                log.exception("permit refresh failed; keeping the previous set")

    def __call__(self, hotkey: str) -> bool:
        return hotkey in self._allowed
