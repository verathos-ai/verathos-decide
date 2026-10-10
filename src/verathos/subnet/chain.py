"""Chain operations shared by miners and validators (bittensor SDK 10.x)."""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable

import bittensor as bt
from async_substrate_interface.errors import ConnectionClosed, InvalidHandshake, SubstrateRequestException

from verathos import catalog
from verathos.catalog import Catalog, CatalogStore
from verathos.protocol.manifest import axon_base

log = logging.getLogger(__name__)
# Failures of the connection to the chain's node (a closed websocket, a keepalive timeout, no route) and errors its RPC
# returns: a later attempt, on a new connection, may succeed. Loops log them in one line (``one_line``); anything else
# is a programming error and keeps its traceback.
CHAIN_ERRORS = (OSError, ConnectionClosed, InvalidHandshake, SubstrateRequestException)


CONNECT_WAITS_S = (5, 15, 30, 60)  # then every 60 s: a public node refuses connections for a while under load


def one_line(exc: BaseException) -> str:
    """An exception as its type and message."""
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def connect(network: str, *, sleep: Callable[[float], None] = time.sleep) -> bt.Subtensor:
    """A connection to the chain for a process that runs on. While the node refuses one (a public node answers HTTP 429
    under load) or cannot be reached, it is tried again after 5, 15 and 30 s, then every 60 s, each failure logged in
    one line: a process that exited instead would be restarted by its manager, but supervisord gives up on a program
    that exits within seconds of its start ten times in a row."""
    for attempt in itertools.count():
        try:
            return bt.Subtensor(network=network)
        except CHAIN_ERRORS as exc:
            wait = CONNECT_WAITS_S[min(attempt, len(CONNECT_WAITS_S) - 1)]
            log.warning("cannot connect to the %s chain (%s); trying again in %d s", network, one_line(exc), wait)
            sleep(wait)
    raise AssertionError("unreachable")


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


def committed_catalog(subtensor: bt.Subtensor, netuid: int, block: int | None = None) -> str | None:
    """The catalog digest the subnet owner hotkey commits (at ``block``, default now), or None."""
    owner = subtensor.get_subnet_owner_hotkey(netuid, block=block)
    uid = subtensor.get_uid_for_hotkey_on_subnet(owner, netuid, block=block)
    return catalog.committed_digest(subtensor.get_commitment(netuid, uid, block=block)) if uid is not None else None


def axon_url(subtensor: bt.Subtensor, netuid: int, hotkey: str) -> str | None:
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    if neuron is None or neuron.is_null or neuron.axon_info.ip in ("", "0.0.0.0") or not neuron.axon_info.port:
        return None
    return axon_base(neuron.axon_info.ip, neuron.axon_info.port)


def current_catalog(subtensor: bt.Subtensor, netuid: int, store: CatalogStore, mirrors: list[str],
                    block: int | None = None) -> Catalog:
    """The catalog the subnet owner commits (at ``block``, default now): from the shipped copy, the local store, the
    owner's axon or a mirror (URL templates with ``{digest}``). Raises CatalogUnavailable."""
    committed = committed_catalog(subtensor, netuid, block)
    if committed is None:
        raise catalog.CatalogUnavailable("the subnet owner has not committed a catalog")
    owner = axon_url(subtensor, netuid, subtensor.get_subnet_owner_hotkey(netuid))
    sources = ([owner + catalog.CATALOG_PATH] if owner else []) + list(mirrors)
    return catalog.resolve(committed, store=store, sources=sources)
