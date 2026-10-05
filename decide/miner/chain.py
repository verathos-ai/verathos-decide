"""Miner chain operations (bittensor SDK 10.x): publish the axon and the manifest commitment, and know which
callers are permitted."""

from __future__ import annotations

import logging
import threading
import time

import bittensor as bt

from decide.miner.config import MinerConfig
from decide.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value

log = logging.getLogger(__name__)


def build_manifest(config: MinerConfig, keypair, catalog_digest: str, block: int) -> SignedManifest:
    """Sign a manifest for the configured endpoints. ``seq`` is the block it was issued at, so it only grows."""
    endpoints = [
        Endpoint(endpoint_id=e.endpoint_id, url=config.endpoint_url(e.endpoint_id), model_id=e.model_id, tier=e.tier,
                 hardware=e.hardware, max_concurrency=e.max_concurrency)
        for e in config.endpoints
    ]
    manifest = Manifest(netuid=config.netuid, mechid=config.mechid, hotkey=keypair.ss58_address, seq=block,
                        issued_block=block, catalog_digest=catalog_digest, endpoints=endpoints)
    return SignedManifest.create(manifest, keypair)


def publish(subtensor: bt.Subtensor, wallet: bt.Wallet, config: MinerConfig, signed: SignedManifest) -> None:
    """Point the axon at this service and commit the manifest digest; each call is made only if it changes state."""
    hotkey = wallet.hotkey.ss58_address
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(hotkey, config.netuid)
    if neuron is None or neuron.is_null:
        raise RuntimeError(f"hotkey {hotkey} is not registered on netuid {config.netuid}")
    axon = neuron.axon_info
    if (axon.ip, axon.port) != (config.public_host, config.port):
        result = subtensor.serve_axon(netuid=config.netuid, axon=bt.Axon(wallet=wallet, port=config.port,
                                      external_ip=config.public_host, external_port=config.port))
        if not getattr(result, "success", False):
            raise RuntimeError(f"serve_axon failed: {getattr(result, 'message', result)}")
        log.info("serve_axon %s:%d", config.public_host, config.port)
    value = commitment_value(signed.manifest.digest())
    if subtensor.get_commitment(config.netuid, neuron.uid) != value:
        result = subtensor.set_commitment(wallet, config.netuid, value)
        if not getattr(result, "success", False):
            raise RuntimeError(f"set_commitment failed: {getattr(result, 'message', result)}")
        log.info("set_commitment %s", value)


class PermitAuthorizer:
    """Allows validators with a validator permit plus configured gateway hotkeys; refreshed in the background."""

    def __init__(self, subtensor: bt.Subtensor, netuid: int, extra: list[str], refresh_s: float = 600.0) -> None:
        self._subtensor, self._netuid, self._extra = subtensor, netuid, set(extra)
        self._allowed: set[str] = set(extra)
        self._refresh_s = refresh_s
        self.refresh()
        threading.Thread(target=self._loop, daemon=True).start()

    def refresh(self) -> None:
        mg = self._subtensor.metagraph(self._netuid, lite=True)
        permitted = {hk for hk, permit in zip(mg.hotkeys, mg.validator_permit) if bool(permit)}
        self._allowed = permitted | self._extra
        log.info("permitted callers: %d", len(self._allowed))

    def _loop(self) -> None:
        while True:
            time.sleep(self._refresh_s)
            try:
                self.refresh()
            except Exception:
                log.exception("permit refresh failed; keeping the previous set")

    def __call__(self, hotkey: str) -> bool:
        return hotkey in self._allowed
