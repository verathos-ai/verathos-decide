"""Results a full validator takes from its source for the models it does not check itself in an epoch (references one
at a time, ``validator.sequential``).

The source is the subnet owner's validator (the auditor), or the validator ``source_hotkey`` names. Its network
document (``GET /v1/network``) for the epoch is read and verified as light validators read it (``validator.light``):
signed by the source hotkey for this netuid within ``STATUS_MAX_AGE_S``, for the catalog's epoch length and for this
epoch, asked every ``light.POLL_S`` until shortly before the epoch ends. Each endpoint of such a model the source lists
enters this validator's scoring by the state the source gave it:
- ``not_audited`` (the source could not check it either) or ``invalid`` (its hotkey gave a wrong answer the auditor
  proved, which reaches this validator through the auditor's verdicts as for every model): its average is held, as
  for an endpoint without reference answers (``scoring.MAX_HELD_EPOCHS``);
- ``failed`` (the source's canaries failed it): failed here too: it scores zero, and every endpoint of its hotkey
  starts probation with its average reset, as for this validator's own failures;
- ``probation``: zero for the epoch, as the source counts it;
- any other state: the score the source gave it, its share of its model's emission in the epoch.
That score then enters the endpoint's average, probation and the weights exactly as one computed here: a validator
that checks every model scores as it would without a source, and one that checks none sets the source's weights. An
endpoint of such a model that the source does not list keeps its average, and so does every endpoint of such a model
when no document verifies.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

import bittensor as bt
import httpx

from verathos.protocol.http import ResponseTooLarge, get_json
from verathos.protocol.status import MAX_NETWORK_BYTES, NETWORK_PATH, EndpointStatus, NetworkStatus
from verathos.subnet.chain import CHAIN_ERRORS, one_line
from verathos.validator.light import FETCH_TIMEOUT_S, POLL_S, Refused, check, find_source

HELD = ("not_audited", "invalid")


@dataclass
class Taken:
    """What the source's document gives the endpoints of the models taken from it, by endpoint key."""

    scores: dict[str, float] = field(default_factory=dict)  # the epoch's score
    failed: set[str] = field(default_factory=set)
    held: set[str] = field(default_factory=set)
    uids: dict[str, int] = field(default_factory=dict)  # every endpoint taken: its UID
    status: dict[str, EndpointStatus] = field(default_factory=dict)  # as the source published it


def take(network: NetworkStatus, models: Collection[str]) -> Taken:
    """The results of ``models``' endpoints in the source's ``network`` document (see the module)."""
    out = Taken()
    for hotkey in network.hotkeys:
        if hotkey.uid is None:
            continue
        for e in hotkey.endpoints:
            if e.model_id not in models:
                continue
            key = f"{hotkey.hotkey}/{e.endpoint_id}"
            out.uids[key], out.status[key] = hotkey.uid, e
            if e.state in HELD:
                out.held.add(key)
            elif e.state == "failed":
                out.failed.add(key)
                out.scores[key] = 0.0
            else:
                out.scores[key] = 0.0 if e.state == "probation" else e.score
    return out


def scores_valid(network: NetworkStatus) -> str | None:
    """Why the document's endpoint scores cannot be scored with: each must be a share of the emission. None if they
    can."""
    for hotkey in network.hotkeys:
        for e in hotkey.endpoints:
            if not (math.isfinite(e.score) and 0.0 <= e.score <= 1.0):
                return f"the score of {hotkey.hotkey}/{e.endpoint_id} is not a share of the emission"
    return None


async def source_results(st: bt.Subtensor, *, netuid: int, hotkey: str, source_hotkey: str | None, epoch: int,
                         epoch_blocks: int, deadline: float, fetch: Callable[..., Any] = get_json,
                         clock: Callable[[], float] = time.time, poll_s: float = POLL_S) -> tuple[str, NetworkStatus]:
    """The source hotkey and its network document for ``epoch``, once it verifies, asked every ``poll_s`` until
    ``deadline`` (``time.monotonic``; at least once). Raises Refused with the last reason."""
    while True:
        try:
            return await _attempt(st, netuid=netuid, hotkey=hotkey, source_hotkey=source_hotkey, epoch=epoch,
                                  epoch_blocks=epoch_blocks, fetch=fetch, now_ms=int(clock() * 1000))
        except Refused:
            if time.monotonic() + poll_s > deadline:
                raise
        await asyncio.sleep(poll_s)


async def _attempt(st: bt.Subtensor, *, netuid: int, hotkey: str, source_hotkey: str | None, epoch: int,
                   epoch_blocks: int, fetch: Callable[..., Any], now_ms: int) -> tuple[str, NetworkStatus]:
    try:
        source, url = find_source(st, netuid, st.get_subnet_owner_hotkey(netuid), source_hotkey)
    except CHAIN_ERRORS as exc:
        raise Refused(f"the chain is not available: {one_line(exc)}") from None
    if source == hotkey:
        raise Refused("the source is this validator")
    try:
        document = await asyncio.to_thread(fetch, url + NETWORK_PATH, limit=MAX_NETWORK_BYTES, timeout=FETCH_TIMEOUT_S)
    except (httpx.HTTPError, ResponseTooLarge, ValueError) as exc:
        raise Refused(f"the source's axon {url} did not serve it: {one_line(exc)}") from None
    network = check(document, source=source, netuid=netuid, epoch=epoch, epoch_blocks=epoch_blocks, done_block=None,
                    now_ms=now_ms)
    if network.epoch != epoch:
        raise Refused(f"the source has not published the results of epoch {epoch} yet (its latest are for epoch "
                      f"{network.epoch})")
    if why := scores_valid(network):
        raise Refused(why)
    return source, network
