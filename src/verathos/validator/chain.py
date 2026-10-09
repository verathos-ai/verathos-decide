"""Validator chain operations (bittensor SDK 10.x)."""

from __future__ import annotations

import logging

import bittensor as bt
from bittensor.core.extrinsics.weights import commit_timelocked_weights_extrinsic, set_weights_extrinsic
from bittensor.core.settings import DEFAULT_PERIOD
from bittensor.utils import get_mechid_storage_index

from verathos import __version__, release
from verathos.protocol.callers import ValidatorIdentity
from verathos.protocol.manifest import committed_digest
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.validator.discovery import Neuron

log = logging.getLogger(__name__)
MAX_UNREVEALED_COMMITS = 10  # a hotkey's unrevealed weight commits per epoch of the subnet, the most the chain keeps


class ReleaseTooOld(Exception):
    """The subnet's weights version key is above this release's: the chain refuses its weights."""


class CommitsUnrevealed(Exception):
    """The chain holds as many unrevealed weight commits of this hotkey as it keeps: it refuses another until it runs
    the subnet's epoch that reveals them. ``epoch``: the subnet's epoch counter then."""

    def __init__(self, message: str, epoch: int) -> None:
        super().__init__(message)
        self.epoch = epoch


def snapshot(subtensor: bt.Subtensor, netuid: int) -> tuple[list[Neuron], dict[int, str]]:
    """Neurons with axon, current commitment, stake and validator permit, plus hotkey per UID."""
    mg = subtensor.metagraph(netuid, lite=True)
    commitments = subtensor.get_all_commitments(netuid)
    neurons = [Neuron(uid, hk, ax.ip, ax.port, commitments.get(hk), float(stake), bool(permit))
               for uid, (hk, ax, stake, permit) in enumerate(zip(mg.hotkeys, mg.axons, mg.S, mg.validator_permit))]
    return neurons, dict(enumerate(mg.hotkeys))


def publish_identity(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, identity: ValidatorIdentity) -> None:
    """Commit the validator's identity (its service's TLS pin, its egress addresses; ``protocol.callers``) unless the
    hotkey commits it already. A hotkey whose commitment is a miner's manifest keeps it: one hotkey is either."""
    value = identity.commitment()
    uid = subtensor.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
    if uid is None:
        raise RuntimeError(f"hotkey {wallet.hotkey.ss58_address} is not registered on netuid {netuid}")
    current = subtensor.get_commitment(netuid, uid)
    if current == value:
        log.info("validator identity already committed on chain (%s)", value)
        return
    if committed_digest(current) is not None:
        raise RuntimeError("this hotkey commits a miner's manifest; a validator publishes its identity with a hotkey "
                           "of its own")
    result = subtensor.set_commitment(wallet, netuid, value)
    if not getattr(result, "success", False):
        raise RuntimeError(f"set_commitment failed: {getattr(result, 'message', result)}")
    log.info("validator identity committed on chain: %s", value)


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


def unrevealed_commits(subtensor: bt.Subtensor, netuid: int, mechid: int, hotkey: str) -> tuple[int, int]:
    """The weight commits of ``hotkey`` on ``mechid`` the chain holds unrevealed where its next commit goes, under the
    subnet's epoch counter or the next value while that epoch is due (the higher count of the two), and the counter.
    The chain reveals them as it runs the subnet's epochs."""
    index = get_mechid_storage_index(netuid, mechid)
    epoch = int(subtensor.substrate.query("SubtensorModule", "SubnetEpochIndex", [netuid]).value or 0)
    counts = []
    for key in (epoch, epoch + 1):
        commits = subtensor.substrate.query("SubtensorModule", "TimelockedWeightCommits", [index, key]).value or ()
        counts.append(sum(1 for commit in commits if commit[0] == hotkey))
    return max(counts), epoch


def weights_version_key(subtensor: bt.Subtensor, netuid: int) -> int:
    """This release's weights version key (``release``). Raises ReleaseTooOld when the subnet's
    ``WeightsVersionKey``, the lowest key the chain accepts weights with, is above it."""
    floor = int(subtensor.substrate.query("SubtensorModule", "WeightsVersionKey", [netuid]).value or 0)
    own = release.weights_version_key()
    if floor > own:
        raise ReleaseTooOld(f"this validator's release is too old to set weights: the subnet's weights version key is "
                            f"{floor}, above its own {own} (verathos {__version__}); update it")
    return own


def submit_weights(subtensor: bt.Subtensor, wallet: bt.Wallet, netuid: int, mechid: int,
                   weights: dict[int, float]) -> bool:
    """Set weights on ``mechid`` with this release's weights version key, committed for a timed reveal where the
    subnet has commit-reveal on; raises ReleaseTooOld if the chain would refuse the key, and CommitsUnrevealed if it
    would refuse another commit, so that no transaction it refuses is sent (and paid for). The extrinsics are called
    directly: the SDK's ``set_weights`` checks the rate limit of mechanism 0 only, which holds back a set on another
    mechanism after one on mechanism 0 and lets through one the chain refuses."""
    version_key = weights_version_key(subtensor, netuid)
    uids = sorted(weights)
    values = [weights[u] for u in uids]
    if subtensor.commit_reveal_enabled(netuid=netuid):
        held, epoch = unrevealed_commits(subtensor, netuid, mechid, wallet.hotkey.ss58_address)
        if held >= MAX_UNREVEALED_COMMITS:
            raise CommitsUnrevealed(f"the chain holds {held} unrevealed weight commits of this hotkey on mechanism "
                                    f"{mechid}, the most it keeps, until it runs the subnet's epoch that reveals them",
                                    epoch)
        result = commit_timelocked_weights_extrinsic(subtensor, wallet, netuid, mechid, uids, values,
                                                     block_time=BLOCK_TIME_S, version_key=version_key,
                                                     period=DEFAULT_PERIOD)
    else:
        result = set_weights_extrinsic(subtensor, wallet, netuid, mechid, uids, values, version_key,
                                       period=DEFAULT_PERIOD)
    ok = bool(getattr(result, "success", False))
    log.info("set_weights mechid=%d uids=%d -> %s %s", mechid, len(uids), ok, getattr(result, "message", ""))
    return ok
