"""The referral pool's periods (``catalog.ReferralPool``): what the auditor records for each, and how it is checked.

Period ``k`` is the epochs ``k x period_epochs`` up to the next period's first: blocks ``first_block`` up to
``end_block``, the next period's first block, which is the period's reference block. The gateways report each
referred account's top-up and refund as it counts (``gateway.referrals``), with the codes' signed registrations and
moves; the auditor takes each entry once, in the period it arrives in. Once ``end_block`` is final the auditor closes
the period into a ``ReferralRecord`` and states it with every verdicts document of period ``k + 1``, signed with them
(``referral``):

- ``codes``: each code with entries in the window (period ``k`` and the ``window_periods - 1`` before it), with its
  hotkey at the end of the period (as registered, then moved, a move only to a hotkey of the code's coldkey), that
  hotkey's coldkey at the reference block, and the sum of its entries in each period of the window, oldest first
  (micro-USD, signed);
- ``referrers``: per hotkey, ``referred_micros``, its codes' sums over the window, at least 0 (``R``); ``cap``,
  ``payout_per_usd x R / (window_periods x tao_usd x alpha_tao x period_emission)``, the share of the mechanism's miner
  emission that pays ``payout_per_usd / window_periods`` of R in USD; and ``share``, ``cap x share / the caps' sum``;
- ``share``: the caps' sum, at most ``max_share``: what the referrers get of the mechanism's miner emission in every
  epoch of period ``k + 1``, at most what the epoch's miner share leaves (``validator.referral``);
- the inputs: ``payout_per_usd``, ``max_share`` (the catalog's ``share``) and ``window_periods``; ``emission`` and
  ``period_emission`` (``throughput.usage``); ``tao_usd`` (from ``TAO_USD_URL`` when the period closed, once anything
  was referred) and ``alpha_tao`` (``SubnetMovingPrice`` at the reference block); and the subnet owner's hotkey and
  coldkey at the reference block, whose hotkeys are paid nothing.

A referred USD counts in at most ``window_periods`` periods at most ``payout_per_usd / window_periods`` of it each:
it earns at most ``payout_per_usd`` USD of emission, however it is timed or split over accounts and codes.

The registrations and moves are taken once, in the order they arrive; each move is decided once, when the first
period after it closes (``follow``), so later events on chain never change where a code went.

Validators recompute the record's figures (``ReferralRecord.inconsistency``), check what their own node holds of the
reference block (``contradiction``) and compare the record with the last one they took (``validator.referral``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from verathos.catalog import MAX_PAYOUT_PER_USD, ReferralPool
from verathos.throughput.usage import Emission, alpha_price_at, block_hash, coldkeys_at, emission_at, period_blocks

MICROS = 1_000_000  # micro-USD per USD


class CodeWindow(BaseModel):
    code: str
    hotkey: str  # at the end of the period
    coldkey: str | None = None  # that hotkey's owner at the reference block (None: not known there)
    periods: list[int]  # micro-USD per period of the window, oldest first: top-ups less refunds


class Referrer(BaseModel):
    hotkey: str
    coldkey: str | None = None
    referred_micros: int = Field(ge=0)
    cap: float = Field(ge=0)
    share: float = Field(ge=0, le=1)


class ReferralRecord(BaseModel):
    """One period's referral pool as the auditor closed it (see the module)."""

    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    period: int = Field(ge=0)
    first_block: int = Field(ge=0)
    end_block: int = Field(ge=0)  # the next period's first block: the reference block
    payout_per_usd: float = Field(gt=0, le=MAX_PAYOUT_PER_USD)
    max_share: float = Field(gt=0, le=1)
    window_periods: int = Field(ge=1)
    emission: Emission
    period_emission: float = Field(ge=0)  # alpha
    tao_usd: float | None = None
    alpha_tao: float | None = None
    owner_hotkey: str | None = None
    owner_coldkey: str | None = None
    codes: list[CodeWindow] = []
    referrers: list[Referrer] = []
    share: float = Field(ge=0, le=1)

    def referred(self) -> dict[str, int]:
        """Each hotkey's codes' sums over the window, at least 0."""
        sums: dict[str, int] = {}
        for c in self.codes:
            sums[c.hotkey] = sums.get(c.hotkey, 0) + sum(c.periods)
        return {hotkey: max(0, n) for hotkey, n in sums.items()}

    def inconsistency(self) -> str | None:
        """Why the record's figures do not follow from its inputs; None if they do."""
        if self.end_block <= self.first_block:
            return "its period ends before it starts"
        if not _close(self.period_emission, self.emission.miner_alpha_per_block() * (self.end_block -
                                                                                    self.first_block)):
            return "its period emission is not its per-block emission over the period's blocks"
        if len({c.code for c in self.codes}) != len(self.codes):
            return "it lists a code twice"
        if any(len(c.periods) != self.window_periods for c in self.codes):
            return "a code's periods are not its window's"
        coldkeys: dict[str, str | None] = {}
        for c in self.codes:
            if coldkeys.setdefault(c.hotkey, c.coldkey) != c.coldkey:
                return f"it states two coldkeys of {c.hotkey}"
        referred = self.referred()
        stated = {r.hotkey: (r.coldkey, r.referred_micros) for r in self.referrers}
        if len(self.referrers) != len(referred) or stated != {h: (coldkeys[h], n) for h, n in referred.items()}:
            return "its referrers are not its codes' hotkeys with their sums"
        if any(referred.values()) and not ((self.tao_usd or 0) > 0 and (self.alpha_tao or 0) > 0):
            return "it lacks its prices"
        caps, share, shares = sized(referred, payout_per_usd=self.payout_per_usd, window_periods=self.window_periods,
                                    max_share=self.max_share, tao_usd=self.tao_usd, alpha_tao=self.alpha_tao,
                                    period_emission=self.period_emission)
        if any(not (_close(r.cap, caps[r.hotkey]) and _close(r.share, shares[r.hotkey])) for r in self.referrers):
            return "its caps or shares do not follow from its sums and prices"
        if not _close(self.share, share):
            return "its share does not follow from its caps"
        return None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def sized(referred: dict[str, int], *, payout_per_usd: float, window_periods: int, max_share: float,
          tao_usd: float | None, alpha_tao: float | None,
          period_emission: float) -> tuple[dict[str, float], float, dict[str, float]]:
    """-> (each hotkey's cap, the period's share, each hotkey's share) for the referred micro-USD of each hotkey over
    the window (see the module)."""
    emission_usd = (tao_usd or 0.0) * (alpha_tao or 0.0) * period_emission  # a period's miner emission in USD
    caps = {hotkey: payout_per_usd * n / MICROS / (window_periods * emission_usd) if emission_usd > 0 else 0.0
            for hotkey, n in sorted(referred.items())}
    total = sum(caps.values())
    share = min(max_share, total)
    return caps, share, {hotkey: cap * share / total if total > 0 else 0.0 for hotkey, cap in caps.items()}


@dataclass(frozen=True)
class CodeState:
    """A code at the auditor: its registration (the hotkey that signed it, the order and block it arrived in) and,
    once a period after it was closed, the hotkey it is on (``at``) and the coldkey it belongs to."""

    code: str
    hotkey: str
    seq: int
    block: int
    at: str | None = None
    coldkey: str | None = None


@dataclass(frozen=True)
class Move:
    """A move the auditor has not decided yet: every code on ``source`` to ``target``."""

    seq: int
    source: str
    target: str


def follow(codes: Iterable[CodeState], moves: Iterable[Move], coldkeys: dict[str, str | None],
           end_block: int) -> tuple[dict[str, tuple[str, str | None]], dict[int, str]]:
    """Each code's hotkey and coldkey at a period's end, from what the auditor held before (``codes``' ``at`` and
    ``coldkey``) and the registrations and moves since that arrived before ``end_block``, each taken once, in the
    order they arrived. ``coldkeys``: the hotkeys' owners at the end block.

    A code belongs to the coldkey that owned the hotkey registering it, and follows that coldkey when the chain moves
    its hotkey to another (a coldkey swap). A move takes the codes on its source hotkey that arrived before it to its
    target only if the target belongs to the codes' coldkey, so a code never changes coldkey by a move, whatever
    happens to the source hotkey since (a hotkey swap leaves it without an owner). -> (code -> (hotkey, coldkey), each
    move -> moved, refused (its codes belong to another coldkey) or empty (no code on its source))."""
    states: dict[str, tuple[str, str | None]] = {}
    events: list[tuple[int, CodeState | Move]] = []
    for c in codes:
        if c.at is not None:
            states[c.code] = (c.at, coldkeys.get(c.at) or c.coldkey)
        elif c.block < end_block:
            events.append((c.seq, c))
    events += [(m.seq, m) for m in moves]
    outcomes: dict[int, str] = {}
    for _, event in sorted(events, key=lambda e: e[0]):
        if isinstance(event, CodeState):
            states[event.code] = (event.hotkey, coldkeys.get(event.hotkey))
            continue
        on_source = [code for code, (at, _) in states.items() if at == event.source]
        target = coldkeys.get(event.target)
        moved = [code for code in on_source if target is not None and states[code][1] == target]
        for code in moved:
            states[code] = (event.target, target)
        outcomes[event.seq] = "moved" if moved else "refused" if on_source else "empty"
    return states, outcomes


def close(period: int, *, pool: ReferralPool, epoch_blocks: int, entries: Iterable[tuple[str, int, int]],
          hotkeys: dict[str, str], coldkeys: dict[str, str | None], owner: tuple[str | None, str | None],
          emission: Emission, tao_usd: float | None = None, alpha_tao: float | None = None,
          since_block: int = 0) -> ReferralRecord:
    """The record of ``period``: ``entries`` (code, the block it arrived at, micro-USD) as the auditor took them,
    those from ``since_block`` on in the window counting; ``hotkeys`` each code's hotkey and ``coldkeys`` their owners,
    ``owner`` the subnet owner's hotkey and coldkey and ``emission`` the miners' emission, at the reference block; the
    prices, needed once anything was referred."""
    first, end = period_blocks(pool, epoch_blocks, period)
    length, window = end - first, pool.window_periods
    start = end - window * length
    sums: dict[str, list[int]] = {}
    for code, block, amount in entries:
        if max(start, since_block) <= block < end and code in hotkeys:
            sums.setdefault(code, [0] * window)[(block - start) // length] += amount
    codes = [CodeWindow(code=code, hotkey=hotkeys[code], coldkey=coldkeys.get(hotkeys[code]), periods=periods)
             for code, periods in sorted(sums.items()) if any(periods)]
    period_emission = emission.miner_alpha_per_block() * length
    record = ReferralRecord(period=period, first_block=first, end_block=end, payout_per_usd=pool.payout_per_usd,
                            max_share=pool.share, window_periods=window, emission=emission,
                            period_emission=period_emission, tao_usd=tao_usd, alpha_tao=alpha_tao,
                            owner_hotkey=owner[0], owner_coldkey=owner[1], codes=codes, share=0.0)
    referred = record.referred()
    caps, share, shares = sized(referred, payout_per_usd=pool.payout_per_usd, window_periods=window,
                                max_share=pool.share, tao_usd=tao_usd, alpha_tao=alpha_tao,
                                period_emission=period_emission)
    record.referrers = [Referrer(hotkey=hotkey, coldkey=coldkeys.get(hotkey), referred_micros=n, cap=caps[hotkey],
                                 share=shares[hotkey]) for hotkey, n in sorted(referred.items())]
    record.share = share
    return record


def record_of(document: Any) -> ReferralRecord | None:
    """The referral record a verdicts document states (its ``referral``), if well formed."""
    if not isinstance(document, dict):
        return None
    try:
        return ReferralRecord.model_validate(document)
    except ValueError:
        return None


def owner_at(st: Any, block: int, netuid: int) -> tuple[str | None, str | None]:
    """The subnet owner's hotkey and coldkey at ``block`` (``SubnetOwnerHotkey``, ``SubnetOwner``)."""
    at = block_hash(st, block)

    def value(name: str) -> str | None:
        return st.substrate.query("SubtensorModule", name, [netuid], block_hash=at).value or None
    return value("SubnetOwnerHotkey"), value("SubnetOwner")


def contradiction(record: ReferralRecord, st: Any, netuid: int,
                  previous: ReferralRecord | None = None) -> str | None:
    """What the chain, as this node still holds it at the reference block, says against ``record``: another
    emission, alpha price, subnet owner or referrer's coldkey, or, against ``previous`` (an earlier record whose
    window overlaps this one's), a code on another hotkey whose previous hotkey belongs to another coldkey than its new
    one. None if nothing; what the node no longer holds (or cannot read now) is not checked."""
    block = record.end_block
    try:
        emission = emission_at(st, block, netuid, record.emission.mechanism)
    except Exception:
        emission = None
    if emission is not None and emission != record.emission:
        return f"the chain's miner emission at block {block} is {emission}, not {record.emission}"
    if record.alpha_tao is not None:
        try:
            price = alpha_price_at(st, block, netuid)
        except Exception:
            price = None
        if price is not None and not math.isclose(price, record.alpha_tao, rel_tol=1e-9, abs_tol=1e-9):
            return f"the chain's moving alpha price at block {block} is {price}, not {record.alpha_tao}"
    try:
        owner = owner_at(st, block, netuid)
    except Exception:
        owner = None
    if owner is not None and owner != (record.owner_hotkey, record.owner_coldkey):
        return f"the subnet owner at block {block} is {owner[0]} of {owner[1]}, not as the record states"
    try:
        coldkeys = coldkeys_at(st, [r.hotkey for r in record.referrers], block)
    except Exception:
        coldkeys = {}
    for r in record.referrers:
        if r.hotkey in coldkeys and coldkeys[r.hotkey] != r.coldkey:
            return f"{r.hotkey}'s coldkey at block {block} is {coldkeys[r.hotkey]}, not {r.coldkey}"
    length = record.end_block - record.first_block
    if previous is None or previous.end_block <= record.end_block - record.window_periods * length:
        return None  # no earlier record whose window this one overlaps
    before = {c.code: c.hotkey for c in previous.codes}
    moved = {c.code: c for c in record.codes if before.get(c.code, c.hotkey) != c.hotkey}
    try:
        owners = coldkeys_at(st, sorted({before[code] for code in moved}), block)
    except Exception:
        owners = {}
    for code, c in sorted(moved.items()):
        owner = owners.get(before[code])
        if owner is not None and c.coldkey is not None and owner != c.coldkey:  # unknown: cannot tell
            return (f"it moves code {code} from {before[code]} to {c.hotkey}, a hotkey of another coldkey than "
                    f"{before[code]}'s at block {block}")
    return None
