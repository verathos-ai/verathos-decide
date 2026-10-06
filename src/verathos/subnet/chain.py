"""Chain operations shared by miners and validators (bittensor SDK 10.x)."""

from __future__ import annotations

import logging

import bittensor as bt

from verathos import catalog
from verathos.catalog import Catalog, CatalogStore

log = logging.getLogger(__name__)


def serve_axon(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, host: str, port: int) -> int:
    """Point this hotkey's axon at ``host:port`` unless it already does. Returns the hotkey's UID."""
    hotkey = wallet.hotkey.ss58_address
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    if neuron is None or neuron.is_null:
        raise RuntimeError(f"hotkey {hotkey} is not registered on netuid {netuid}")
    if (neuron.axon_info.ip, neuron.axon_info.port) != (host, port):
        result = subtensor.serve_axon(netuid=netuid, axon=bt.Axon(wallet=wallet, port=port, external_ip=host,
                                                                   external_port=port))
        if not getattr(result, "success", False):
            raise RuntimeError(f"serve_axon failed: {getattr(result, 'message', result)}")
        log.info("serve_axon %s:%d", host, port)
    return neuron.uid


def committed_catalog(subtensor: bt.Subtensor, netuid: int) -> str | None:
    """The catalog digest the subnet owner hotkey currently commits, or None."""
    owner = subtensor.get_subnet_owner_hotkey(netuid)
    uid = subtensor.get_uid_for_hotkey_on_subnet(owner, netuid)
    return catalog.committed_digest(subtensor.get_commitment(netuid, uid)) if uid is not None else None


def axon_url(subtensor: bt.Subtensor, netuid: int, hotkey: str) -> str | None:
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    if neuron is None or neuron.is_null or neuron.axon_info.ip in ("", "0.0.0.0") or not neuron.axon_info.port:
        return None
    return f"http://{neuron.axon_info.ip}:{neuron.axon_info.port}"


def current_catalog(subtensor: bt.Subtensor, netuid: int, store: CatalogStore, mirrors: list[str]) -> Catalog:
    """The catalog the subnet owner currently commits: from the shipped copy, the local store, the owner's axon or
    a mirror (URL templates with ``{digest}``). Raises CatalogUnavailable."""
    committed = committed_catalog(subtensor, netuid)
    if committed is None:
        raise catalog.CatalogUnavailable("the subnet owner has not committed a catalog")
    owner = axon_url(subtensor, netuid, subtensor.get_subnet_owner_hotkey(netuid))
    sources = ([owner + catalog.CATALOG_PATH] if owner else []) + list(mirrors)
    return catalog.resolve(committed, store=store, sources=sources)
