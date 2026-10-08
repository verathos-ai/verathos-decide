"""Validator chain operations (bittensor SDK 10.x)."""

from __future__ import annotations

import logging

import bittensor as bt
from bittensor.core.extrinsics.weights import commit_timelocked_weights_extrinsic, set_weights_extrinsic
from bittensor.core.settings import DEFAULT_PERIOD
from bittensor.utils import get_mechid_storage_index

from verathos.throughput.policy import BLOCK_TIME_S
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


def weights_wait(subtensor: bt.Subtensor, netuid: int, mechid: int, uid: int) -> int:
    """Blocks until ``uid`` may set weights on ``mechid`` again; 0 if it may now. The chain limits how often a UID
    sets weights per mechanism, from its last set on that mechanism."""
    block = subtensor.get_current_block()
    last = subtensor.substrate.query("SubtensorModule", "LastUpdate", [get_mechid_storage_index(netuid, mechid)]).value
    if uid >= len(last or []) or not last[uid]:
        return 0
    return max(0, int(subtensor.weights_rate_limit(netuid)) - (block - int(last[uid])))


def submit_weights(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, mechid: int,
                   weights: dict[int, float]) -> bool:
    """Set weights on ``mechid`` with the subnet's weights version key as the floor, committed for a timed reveal
    where the subnet has commit-reveal on. The extrinsics are called directly: the SDK's ``set_weights`` checks the
    rate limit of mechanism 0 only, which holds back a set on another mechanism after one on mechanism 0 and lets
    through one the chain refuses."""
    version_key = max(int(subtensor.substrate.query("SubtensorModule", "WeightsVersionKey", [netuid]).value or 0), 1)
    uids = sorted(weights)
    values = [weights[u] for u in uids]
    if subtensor.commit_reveal_enabled(netuid=netuid):
        result = commit_timelocked_weights_extrinsic(subtensor, wallet, netuid, mechid, uids, values,
                                                     block_time=BLOCK_TIME_S, version_key=version_key,
                                                     period=DEFAULT_PERIOD)
    else:
        result = set_weights_extrinsic(subtensor, wallet, netuid, mechid, uids, values, version_key,
                                       period=DEFAULT_PERIOD)
    ok = bool(getattr(result, "success", False))
    log.info("set_weights mechid=%d uids=%d -> %s %s", mechid, len(uids), ok, getattr(result, "message", ""))
    return ok
