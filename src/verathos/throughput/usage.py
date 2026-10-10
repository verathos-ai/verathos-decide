"""The usage pool's periods (``catalog.UsagePool``): what the auditor records for each, and how it is checked.

Period ``k`` is the epochs ``k x period_epochs`` up to the next period's first: blocks ``first_block`` up to
``end_block``, the next period's first block, which is the period's reference block. Once ``end_block`` is final the
auditor closes the period into a ``UsageRecord`` and states it with every verdicts document of period ``k + 1``, signed
with them (``usage``):

- ``served``: per hotkey, the input tokens of the answers its endpoints gave paying users in the period (the
  verdicts' ``paid_served`` of its epochs, summed), with its coldkey at the reference block; ``paid_input_tokens``
  the same per model;
- ``emission``: what the chain pays the mechanism's miners per block, read at the reference block, and
  ``period_emission``, that over the period's blocks, in alpha;
- ``basis``, in alpha. ``burned``: what ``buyback_coldkey`` burned on the subnet in the period, its ``AlphaBurned``
  events each listed with its block and position (``burns``; blocks the auditor's node no longer held when it read
  them are listed in ``unscanned`` and their burns do not count). ``revenue``: the period's paid input tokens per model
  at the catalog's ``prices`` (``revenue_usd``), converted at ``tao_usd`` (USD per TAO, read from ``TAO_USD_URL`` when
  the period closed) and ``alpha_tao`` (the subnet's moving alpha price in TAO at the reference block);
- ``share``: ``fraction x basis / period_emission``, at most 1, rounded to ``SHARE_DIGITS``: the share of the
  mechanism's miner emission the pool pays in every epoch of period ``k + 1``, at most what the epoch's miner share
  leaves (``validator.usage``).

A record states every input. Validators recompute its sums (``UsageRecord.inconsistency``) and check what their own
node still holds (``contradiction``): the burns, and the emission and price at the reference block. What their node no
longer holds they take on the auditor's word, as they take its capacity credits.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from verathos.catalog import SHARE_DIGITS, Catalog, ReferralPool, UsagePool
from verathos.protocol.http import get_json
from verathos.protocol.tls import CA

log = logging.getLogger(__name__)

RAO = 10**9  # rao per alpha (and per TAO)
U16_MAX = 65535  # the owner cut and the mechanisms' split are parts of it
MINER_PART = 0.5  # of what a block emits to a subnet's participants after the owner cut, the miners' part
PRICE_SCALE = 2**32  # SubnetMovingPrice is a fixed-point number with 32 fractional bits
TAO_USD_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
TOKENS_PER_PRICE = 1_000_000  # a model's price is in USD per million input tokens
NO_OWNER = "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"  # the zero account: ``Owner`` of a hotkey nobody owns


class Burn(BaseModel):
    """One ``AlphaBurned`` event: ``coldkey`` destroyed ``amount`` rao of its alpha staked to ``hotkey``."""

    block: int = Field(ge=0)
    event_index: int = Field(ge=0)  # its position among the block's events
    coldkey: str
    hotkey: str
    amount: int = Field(ge=0)  # rao


class Emission(BaseModel):
    """What the chain pays one mechanism's miners per block, from the subnet's state at a block: its alpha out per
    block, less the owner cut where the subnet has it on, the miners' half of it, split between the mechanisms (an
    even split where the subnet sets none, or one of another length)."""

    alpha_out: int = Field(ge=0)  # SubnetAlphaOutEmission, rao per block
    owner_cut: int = Field(ge=0, le=U16_MAX)  # SubnetOwnerCut, 0 where the subnet's OwnerCutEnabled is off
    mechanism: int = Field(ge=0)
    mechanisms: int = Field(ge=1)  # MechanismCountCurrent
    split: list[int] | None = None  # MechanismEmissionSplit

    def miner_alpha_per_block(self) -> float:
        if self.mechanism >= self.mechanisms:
            return 0.0
        if self.split is not None and len(self.split) == self.mechanisms:
            part = self.split[self.mechanism] / U16_MAX
        else:
            part = 1 / self.mechanisms
        return self.alpha_out / RAO * (1 - self.owner_cut / U16_MAX) * MINER_PART * part


class Served(BaseModel):
    hotkey: str
    coldkey: str | None = None  # its owner at the reference block (None: not known there)
    input_tokens: int = Field(ge=0)  # of the answers its endpoints gave paying users in the period


class UsageRecord(BaseModel):
    """One period's usage pool as the auditor closed it (see the module)."""

    model_config = ConfigDict(extra="allow")  # a newer auditor's fields stay signed
    period: int = Field(ge=0)
    first_block: int = Field(ge=0)
    end_block: int = Field(ge=0)  # the next period's first block: the reference block
    mode: Literal["burned", "revenue"]
    fraction: float = Field(gt=0, le=1)
    emission: Emission
    period_emission: float = Field(ge=0)  # alpha
    basis: float = Field(ge=0)  # alpha
    share: float = Field(ge=0, le=1)
    served: list[Served] = []
    paid_input_tokens: dict[str, int] = {}  # per model
    buyback_coldkey: str | None = None
    burns: list[Burn] = []
    unscanned: list[tuple[int, int]] = []  # [first, end) block ranges not read: their burns do not count
    prices: dict[str, float] = {}  # per model, USD per million input tokens
    revenue_usd: float | None = None
    tao_usd: float | None = None
    alpha_tao: float | None = None

    def inconsistency(self) -> str | None:
        """Why the record's figures do not follow from its inputs; None if they do."""
        if self.end_block <= self.first_block:
            return "its period ends before it starts"
        if not _close(self.period_emission, self.emission.miner_alpha_per_block() * (self.end_block -
                                                                                    self.first_block)):
            return "its period emission is not its per-block emission over the period's blocks"
        if sum(s.input_tokens for s in self.served) != sum(self.paid_input_tokens.values()):
            return "its hotkeys' paid input tokens do not add up to its models'"
        if self.mode == "burned":
            if self.buyback_coldkey is None or any(b.coldkey != self.buyback_coldkey for b in self.burns):
                return "it counts burns of another coldkey than its buyback coldkey"
            if any(not self.first_block <= b.block < self.end_block for b in self.burns):
                return "it counts burns outside its period"
            if len({(b.block, b.event_index) for b in self.burns}) != len(self.burns):
                return "it counts a burn twice"
            basis = burned(self.burns)
        else:
            if self.tao_usd is None or self.alpha_tao is None or self.revenue_usd is None:
                return "it lacks its revenue or prices"
            if not _close(self.revenue_usd, revenue(self.paid_input_tokens, self.prices)):
                return "its revenue is not its paid input tokens at its prices"
            basis = in_alpha(self.revenue_usd, self.tao_usd, self.alpha_tao)
        if not _close(self.basis, basis):
            return "its basis does not follow from its burns or revenue"
        if abs(self.share - sized_share(self.fraction, self.basis, self.period_emission)) > 10**-SHARE_DIGITS:
            return "its share does not follow from its basis"
        return None


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def period_blocks(pool: UsagePool | ReferralPool, epoch_blocks: int, period: int) -> tuple[int, int]:
    """The period's first block and the next period's (its reference block)."""
    length = pool.period_epochs * epoch_blocks
    return period * length, (period + 1) * length


def burned(burns: Iterable[Burn]) -> float:
    """The alpha the burns destroyed."""
    return sum(b.amount for b in burns) / RAO


def revenue(paid_input_tokens: dict[str, int], prices: dict[str, float]) -> float:
    """USD users paid for their input tokens, at each model's price (USD per million input tokens)."""
    return sum(tokens * prices.get(model, 0.0) for model, tokens in sorted(paid_input_tokens.items())) / (
        TOKENS_PER_PRICE)


def in_alpha(usd: float, tao_usd: float, alpha_tao: float) -> float:
    """``usd`` in alpha at ``tao_usd`` USD per TAO and ``alpha_tao`` TAO per alpha."""
    return usd / tao_usd / alpha_tao if tao_usd > 0 and alpha_tao > 0 else 0.0


def sized_share(fraction: float, basis: float, period_emission: float) -> float:
    """The share of the mechanism's miner emission that pays ``fraction`` of ``basis`` alpha over a period whose miner
    emission is ``period_emission`` alpha: at most 1, rounded to ``SHARE_DIGITS``."""
    if period_emission <= 0:
        return 0.0
    return round(min(1.0, fraction * basis / period_emission), SHARE_DIGITS)


def close(period: int, *, pool: UsagePool, cat: Catalog, epoch_blocks: int, paid: dict[tuple[str, str, str], int],
          coldkeys: dict[str, str | None], emission: Emission, burns: list[Burn] | None = None,
          unscanned: list[tuple[int, int]] | None = None, tao_usd: float | None = None,
          alpha_tao: float | None = None) -> UsageRecord:
    """The record of ``period``: ``paid`` holds the period's paid input tokens per (hotkey, endpoint, model),
    ``coldkeys`` the served hotkeys' owners and ``emission`` the miners' emission, at the reference block; in burned
    mode ``burns`` are the subnet's burns in the period (any coldkey's) and ``unscanned`` the ranges not read; in
    revenue mode ``tao_usd`` and ``alpha_tao`` the prices."""
    first, end = period_blocks(pool, epoch_blocks, period)
    per_hotkey: dict[str, int] = {}
    per_model: dict[str, int] = {}
    for (hotkey, _, model), tokens in sorted(paid.items()):
        per_hotkey[hotkey] = per_hotkey.get(hotkey, 0) + tokens
        per_model[model] = per_model.get(model, 0) + tokens
    period_emission = emission.miner_alpha_per_block() * (end - first)
    fields: dict[str, Any] = {}
    if pool.mode == "burned":
        counted = sorted((b for b in burns or [] if b.coldkey == pool.buyback_coldkey and first <= b.block < end),
                         key=lambda b: (b.block, b.event_index))
        basis = burned(counted)
        fields = {"buyback_coldkey": pool.buyback_coldkey, "burns": counted,
                  "unscanned": [(max(a, first), min(b, end)) for a, b in unscanned or [] if a < end and b > first]}
    else:
        prices = {model: m.price for model in per_model if (m := cat.get(model)) is not None}
        usd = revenue(per_model, prices)
        basis = in_alpha(usd, tao_usd or 0.0, alpha_tao or 0.0)
        fields = {"prices": prices, "revenue_usd": usd, "tao_usd": tao_usd, "alpha_tao": alpha_tao}
    return UsageRecord(period=period, first_block=first, end_block=end, mode=pool.mode, fraction=pool.fraction,
                       emission=emission, period_emission=period_emission, basis=basis,
                       share=sized_share(pool.fraction, basis, period_emission),
                       served=[Served(hotkey=hk, coldkey=coldkeys.get(hk), input_tokens=n)
                               for hk, n in sorted(per_hotkey.items())],
                       paid_input_tokens=dict(sorted(per_model.items())), **fields)


def record_of(document: Any) -> UsageRecord | None:
    """The usage record a verdicts document states (its ``usage``), if well formed."""
    if not isinstance(document, dict):
        return None
    try:
        return UsageRecord.model_validate(document)
    except ValueError:
        return None


# ---- the chain (bittensor SDK 10.x) ----

def block_hash(st: Any, block: int) -> str:
    """The block's hash. Raises LookupError where the node holds no such block (yet): a read without its hash would be
    of the chain's head, and the SDK keeps a missing hash for the connection's life."""
    found = st.substrate.get_block_hash(block)
    if not found:
        raise LookupError(f"the chain's node holds no block {block}")
    return found


def _int(value: Any) -> int:
    """A chain integer as the SDK decodes it: a number, or a one-field wrapper of one."""
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return int(value)


def burns_in(events: list[dict], block: int, netuid: int) -> list[Burn]:
    """The ``AlphaBurned(coldkey, hotkey, amount, netuid)`` events on ``netuid`` among a block's events."""
    out = []
    for index, event in enumerate(events):
        if event.get("module_id") != "SubtensorModule" or event.get("event_id") != "AlphaBurned":
            continue
        values = event.get("attributes")
        if isinstance(values, dict):
            values = list(values.values())
        try:
            coldkey, hotkey, amount, net = values
            if _int(net) == netuid:
                out.append(Burn(block=block, event_index=index, coldkey=str(coldkey), hotkey=str(hotkey),
                                amount=_int(amount)))
        except (TypeError, ValueError):
            log.warning("an AlphaBurned event at block %d is not (coldkey, hotkey, amount, netuid): %r", block, values)
    return out


def burns_at(st: Any, block: int, netuid: int) -> list[Burn]:
    """The burns on ``netuid`` in ``block``. Raises when the node no longer holds the block's state."""
    return burns_in(st.substrate.get_events(block_hash(st, block)), block, netuid)


def emission_at(st: Any, block: int, netuid: int, mechanism: int) -> Emission:
    """The mechanism's miner emission per block as the chain's state at ``block`` sets it."""
    at = block_hash(st, block)

    def value(name: str, params: list) -> Any:
        return st.substrate.query("SubtensorModule", name, params, block_hash=at).value
    cut = _int(value("SubnetOwnerCut", []) or 0)
    if st.substrate.get_metadata_storage_function("SubtensorModule", "OwnerCutEnabled", block_hash=at):
        cut = cut if value("OwnerCutEnabled", [netuid]) else 0
    split = value("MechanismEmissionSplit", [netuid])
    return Emission(alpha_out=_int(value("SubnetAlphaOutEmission", [netuid]) or 0), owner_cut=cut,
                    mechanism=mechanism, mechanisms=max(_int(value("MechanismCountCurrent", [netuid]) or 1), 1),
                    split=[_int(s) for s in split] if split else None)


def alpha_price_at(st: Any, block: int, netuid: int) -> float:
    """The subnet's moving alpha price in TAO at ``block`` (``SubnetMovingPrice``)."""
    raw = st.substrate.query("SubtensorModule", "SubnetMovingPrice", [netuid], block_hash=block_hash(st, block)).value
    return _int(raw.get("bits", raw) if isinstance(raw, dict) else raw) / PRICE_SCALE


def coldkeys_at(st: Any, hotkeys: Iterable[str], block: int) -> dict[str, str | None]:
    """Each hotkey's owner coldkey at ``block`` (``Owner``); None for a hotkey without one, which the chain reads as
    the zero account (``NO_OWNER``)."""
    at = block_hash(st, block)

    def owner(hotkey: str) -> str | None:
        value = st.substrate.query("SubtensorModule", "Owner", [hotkey], block_hash=at).value
        return value if value and value != NO_OWNER else None
    return {hk: owner(hk) for hk in hotkeys}


def tao_usd(fetch: Callable[..., Any] = get_json) -> float:
    """TAO's price in USD from ``TAO_USD_URL``. Raises if it does not give a positive number."""
    price = float(fetch(TAO_USD_URL, limit=4096, timeout=15, tls=CA)["bittensor"]["usd"])
    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"not a price: {price}")
    return price


def contradiction(record: UsageRecord, st: Any, netuid: int) -> str | None:
    """What the chain, as this node still holds it, says against ``record``: a listed burn it does not have, or
    another emission or alpha price at the reference block. None if nothing; what the node no longer holds (or cannot
    read now) is not checked."""
    by_block: dict[int, list[Burn]] = {}
    for burn in record.burns:
        by_block.setdefault(burn.block, []).append(burn)
    for block, listed in sorted(by_block.items()):
        try:
            held = burns_at(st, block, netuid)
        except Exception:
            continue  # beyond the node's state
        if missing := [b for b in listed if b not in held]:
            b = missing[0]
            return (f"the chain has no burn of {b.amount} rao by {b.coldkey} at block {block} (event {b.event_index}) "
                    "as the record lists")
    try:
        emission = emission_at(st, record.end_block, netuid, record.emission.mechanism)
    except Exception:
        emission = None
    if emission is not None and emission != record.emission:
        return f"the chain's miner emission at block {record.end_block} is {emission}, not {record.emission}"
    if record.mode == "revenue" and record.alpha_tao is not None:
        try:
            price = alpha_price_at(st, record.end_block, netuid)
        except Exception:
            price = None
        if price is not None and not _close(price, record.alpha_tao):
            return f"the chain's moving alpha price at block {record.end_block} is {price}, not {record.alpha_tao}"
    return None
