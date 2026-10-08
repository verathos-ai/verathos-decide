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
from verathos.subnet.chain import CHAIN_ERRORS, one_line, serve_axon
from verathos.throughput.policy import BLOCK_TIME_S

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
    """Allows validators with a validator permit and at least ``min_stake()`` of stake, the metagraph's S (both
    refreshed in the background over its own chain connection), and the callers ``allowed`` names at the time of the
    request (the gateways and the auditor). It also follows the chain's head, for the epoch requests are counted in
    (``epoch``)."""

    def __init__(self, network: str, netuid: int, allowed: Callable[[], Collection[str]],
                 min_stake: Callable[[], float] = lambda: 0.0, refresh_s: float = 600.0) -> None:
        self._subtensor, self._netuid, self._allowed = bt.Subtensor(network=network), netuid, allowed
        self._min_stake = min_stake
        self._stake: dict[str, float] = {}  # validators with a permit -> their stake
        self._head = (0, time.monotonic())  # (block, when it was read)
        self._logged: tuple[int, ...] | None = None
        self._refresh_s = refresh_s
        self.refresh()
        threading.Thread(target=self._loop, daemon=True).start()

    def refresh(self) -> None:
        mg = self._subtensor.metagraph(self._netuid, lite=True)
        self._stake = {hk: float(s) for hk, permit, s in zip(mg.hotkeys, mg.validator_permit, mg.S) if bool(permit)}
        self._head = (int(mg.block), time.monotonic())
        floor = self._min_stake()
        counts = (sum(s >= floor for s in self._stake.values()), sum(s < floor for s in self._stake.values()))
        if counts != self._logged:
            log.info("accepting requests from %d validators, the gateways and the auditor%s", counts[0],
                     f"; refusing {counts[1]} validators with a permit but less stake than validator_min_stake "
                     f"{floor:,.0f}" if counts[1] else "")
            self._logged = counts

    def _loop(self) -> None:
        while True:
            time.sleep(self._refresh_s)
            self.try_refresh()

    def try_refresh(self) -> None:
        """``refresh``; if it fails, the previous set stays."""
        try:
            self.refresh()
        except CHAIN_ERRORS as exc:
            log.warning("permit refresh failed (%s); keeping the previous set", one_line(exc))
        except Exception:
            log.exception("permit refresh failed; keeping the previous set")

    def __call__(self, hotkey: str) -> bool:
        stake = self._stake.get(hotkey)
        return (stake is not None and stake >= self._min_stake()) or hotkey in self._allowed()

    def epoch(self, epoch_blocks: int) -> tuple[int, float]:
        """(the current epoch of ``epoch_blocks`` blocks, seconds until the next), from the head last read."""
        block, read = self._head
        now = block + (time.monotonic() - read) / BLOCK_TIME_S
        epoch = int(now // epoch_blocks)
        return epoch, ((epoch + 1) * epoch_blocks - now) * BLOCK_TIME_S
