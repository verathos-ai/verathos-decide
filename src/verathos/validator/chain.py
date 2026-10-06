"""Validator chain operations (bittensor SDK 10.x)."""

from __future__ import annotations

import logging

import bittensor as bt

from verathos.validator.discovery import Neuron

log = logging.getLogger(__name__)


def snapshot(subtensor: bt.Subtensor, netuid: int) -> tuple[list[Neuron], dict[int, str]]:
    """Neurons with axon and current commitment, plus hotkey per UID."""
    mg = subtensor.metagraph(netuid, lite=True)
    commitments = subtensor.get_all_commitments(netuid)
    neurons = [Neuron(uid, hk, ax.ip, ax.port, commitments.get(hk))
               for uid, (hk, ax) in enumerate(zip(mg.hotkeys, mg.axons))]
    return neurons, dict(enumerate(mg.hotkeys))


def owner_uid(subtensor: bt.Subtensor, netuid: int, hotkey_of_uid: dict[int, str]) -> int | None:
    owner = subtensor.get_subnet_owner_hotkey(netuid)
    return next((uid for uid, hk in hotkey_of_uid.items() if hk == owner), None)


def serve_axon(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, host: str, port: int) -> None:
    """Point this hotkey's axon at ``host:port`` unless it already does."""
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(wallet.hotkey.ss58_address, netuid)
    if neuron is None or neuron.is_null:
        raise RuntimeError(f"hotkey {wallet.hotkey.ss58_address} is not registered on netuid {netuid}")
    if (neuron.axon_info.ip, neuron.axon_info.port) == (host, port):
        return
    result = subtensor.serve_axon(netuid=netuid, axon=bt.Axon(wallet=wallet, port=port, external_ip=host,
                                                               external_port=port))
    if not getattr(result, "success", False):
        raise RuntimeError(f"serve_axon failed: {getattr(result, 'message', result)}")
    log.info("serve_axon %s:%d", host, port)


def mechanism_ready(subtensor: bt.Subtensor, netuid: int, mechid: int) -> bool:
    return subtensor.get_mechanism_count(netuid) > mechid


def submit_weights(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, mechid: int,
                   weights: dict[int, float]) -> bool:
    """Set weights on ``mechid`` with the subnet's weights version key as the floor."""
    version_key = int(subtensor.substrate.query("SubtensorModule", "WeightsVersionKey", [netuid]).value or 0)
    uids = sorted(weights)
    result = subtensor.set_weights(wallet, netuid, uids, [weights[u] for u in uids], mechid=mechid,
                                   version_key=max(version_key, 1))
    ok = bool(getattr(result, "success", False))
    log.info("set_weights mechid=%d uids=%d -> %s %s", mechid, len(uids), ok, getattr(result, "message", ""))
    return ok
