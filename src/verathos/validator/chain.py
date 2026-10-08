"""Validator chain operations (bittensor SDK 10.x)."""

from __future__ import annotations

import logging

import bittensor as bt

from verathos.validator.discovery import Neuron

log = logging.getLogger(__name__)


def snapshot(subtensor: bt.Subtensor, netuid: int) -> tuple[list[Neuron], dict[int, str]]:
    """Neurons with axon, current commitment, stake and validator permit, plus hotkey per UID."""
    mg = subtensor.metagraph(netuid, lite=True)
    commitments = subtensor.get_all_commitments(netuid)
    neurons = [Neuron(uid, hk, ax.ip, ax.port, commitments.get(hk), float(stake), bool(permit))
               for uid, (hk, ax, stake, permit) in enumerate(zip(mg.hotkeys, mg.axons, mg.S, mg.validator_permit))]
    return neurons, dict(enumerate(mg.hotkeys))


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
