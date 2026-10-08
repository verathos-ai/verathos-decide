"""Miner chain operations (bittensor SDK 10.x): publish the axon and the manifest commitment, and know which
callers are permitted."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Collection

import bittensor as bt

from verathos.miner.config import EndpointConfig, MinerConfig
from verathos.protocol.manifest import (
    Endpoint,
    HostEndpoints,
    Manifest,
    SignedHostEndpoints,
    SignedManifest,
    commitment_value,
)
from verathos.subnet.chain import serve_axon

log = logging.getLogger(__name__)


def own_endpoints(config: MinerConfig, endpoints: list[EndpointConfig],
                  capacity_rps: dict[str, float]) -> list[Endpoint]:
    """This machine's ``endpoints`` as a manifest lists them, with their declared capacities."""
    return [Endpoint(endpoint_id=e.endpoint_id, url=config.endpoint_url(e.endpoint_id), model_id=e.model_id,
                     tier=e.tier, capacity_rps=capacity_rps[e.endpoint_id]) for e in endpoints]


def build_manifest(config: MinerConfig, endpoints: list[Endpoint], keypair, catalog_digest: str) -> SignedManifest:
    """Sign the hotkey's manifest. The same endpoints and catalog always give the same manifest, so a restart does not
    change the on-chain commitment."""
    manifest = Manifest(netuid=config.netuid, mechid=config.mechid, hotkey=keypair.ss58_address,
                        catalog_digest=catalog_digest, endpoints=endpoints)
    return SignedManifest.create(manifest, keypair)


def build_host_endpoints(config: MinerConfig, endpoints: list[Endpoint], keypair,
                         catalog_digest: str) -> SignedHostEndpoints:
    """Sign what this machine serves, for the primary (``GET /v1/endpoints``)."""
    host = HostEndpoints(netuid=config.netuid, mechid=config.mechid, hotkey=keypair.ss58_address,
                         catalog_digest=catalog_digest, url=config.base_url, endpoints=endpoints)
    return SignedHostEndpoints.create(host, keypair)


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
    """Allows validators with a validator permit (refreshed in the background over its own chain connection) and the
    gateways ``gateways`` names at the time of the request."""

    def __init__(self, network: str, netuid: int, gateways: Callable[[], Collection[str]],
                 refresh_s: float = 600.0) -> None:
        self._subtensor, self._netuid, self._gateways = bt.Subtensor(network=network), netuid, gateways
        self._permitted: set[str] = set()
        self._logged = False
        self._refresh_s = refresh_s
        self.refresh()
        threading.Thread(target=self._loop, daemon=True).start()

    def refresh(self) -> None:
        mg = self._subtensor.metagraph(self._netuid, lite=True)
        permitted = {hk for hk, permit in zip(mg.hotkeys, mg.validator_permit) if bool(permit)}
        if permitted != self._permitted or not self._logged:
            log.info("accepting requests from %d validators and %d gateways", len(permitted), len(self._gateways()))
            self._logged = True
        self._permitted = permitted

    def _loop(self) -> None:
        while True:
            time.sleep(self._refresh_s)
            try:
                self.refresh()
            except Exception:
                log.exception("permit refresh failed; keeping the previous set")

    def __call__(self, hotkey: str) -> bool:
        return hotkey in self._permitted or hotkey in self._gateways()
