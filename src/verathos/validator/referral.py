"""The referral pool on the validators' side (``catalog.ReferralPool``, ``throughput.referral``).

Each epoch of period ``k + 1`` pays period ``k``'s record, which comes signed with the auditor's verdicts. A validator
takes the record once per period: it recomputes its figures, checks that it states the catalog's parameters and that
every code's sums in the periods its window shares with the last record it accepted are as that record stated them,
and, unless it is the auditor, what its node still holds of the reference block: the chain's state there, and that a
code now on another hotkey than in that record moved within a coldkey (``throughput.referral.contradiction``). A
record that fails any of these is refused for the whole period (no referral weights; the subnet owner's UID keeps the
share); once the windows no longer overlap, the next record is compared with nothing earlier. The last record accepted
is kept with the scores, so an epoch whose verdicts do not come still pays it.

The epoch's referral share is the record's share, at most what the epoch's miner share leaves. Each referrer gets its
share in the record, scaled down alike when the miner share leaves less, on the UID it holds now; it gets nothing, and
its part stays on the subnet owner's UID, if it holds no UID, if it is the subnet owner's (its hotkey or coldkey, as
the record states them), if the catalog's ``excluded`` names its hotkey or coldkey, or, with ``serving_only``, if it
has no positive score weight in the epoch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

from verathos.catalog import SHARE_DIGITS, Catalog, ReferralPool
from verathos.protocol.status import ReferralPeriod, ReferrerStatus
from verathos.throughput.feed import Verdicts
from verathos.throughput.referral import ReferralRecord, record_of
from verathos.throughput.usage import period_blocks
from verathos.validator.scoring import ScoreState

log = logging.getLogger("verathos.validator")


def pays(record: ReferralRecord, blocks: tuple[int, int]) -> bool:
    """Whether ``record`` is of the period of ``blocks`` (its first block, the next period's)."""
    return (record.first_block, record.end_block) == blocks


def parameters(record: ReferralRecord, pool: ReferralPool) -> str | None:
    """Why ``record`` does not state the catalog's parameters; None if it does."""
    if (record.payout_per_usd, record.max_share, record.window_periods) != (pool.payout_per_usd, pool.share,
                                                                            pool.window_periods):
        return "it states other parameters than the catalog's (payout_per_usd, share, window_periods)"
    return None


def rewrites(previous: ReferralRecord | None, record: ReferralRecord) -> str | None:
    """Why ``record`` contradicts ``previous``, an earlier record of periods as long and a window as long: a code's
    sum in a period both windows hold differs. None if it does not (or the two cannot be compared)."""
    length = record.end_block - record.first_block
    if (previous is None or previous.end_block >= record.end_block or previous.window_periods != record.window_periods
            or previous.end_block - previous.first_block != length):
        return None
    window = record.window_periods

    def sums(r: ReferralRecord) -> dict[tuple[str, int], int]:
        """(code, the end block of a period of the window) -> its sum."""
        return {(c.code, r.end_block - (window - 1 - i) * length): n for c in r.codes for i, n in enumerate(c.periods)}
    old, new = sums(previous), sums(record)
    ends = range(record.end_block - (window - 1) * length, previous.end_block + 1, length)
    for code in sorted({c.code for c in previous.codes} | {c.code for c in record.codes}):
        for end in ends:
            if old.get((code, end), 0) != new.get((code, end), 0):
                return f"it rewrites the sum of code {code} in the period ending at block {end}"
    return None


def take(state: ScoreState, verdicts: Verdicts | None, cat: Catalog, epoch: int,
         check: Callable[[ReferralRecord, ReferralRecord | None], str | None] | None = None) -> ReferralRecord | None:
    """The record the epoch pays (the period before its own), from ``verdicts`` the first time they state it, else as
    kept in ``state``; None without one, or if it was refused. ``check`` gives why the chain contradicts a record,
    given the last one accepted (None: not checked)."""
    pool = cat.scoring.referral_pool
    if pool is None:
        return None
    wanted = pool.period(epoch) - 1
    blocks = period_blocks(pool, cat.epoch_blocks, wanted)
    accepted = record_of(state.referral_record)
    refused = state.referral_refused or {}
    stated = record_of((verdicts.model_extra or {}).get("referral")) if verdicts is not None else None
    seen = (accepted is not None and pays(accepted, blocks)) or (
        refused.get("first_block"), refused.get("end_block")) == blocks
    if stated is not None and pays(stated, blocks) and not seen:
        why = stated.inconsistency() or parameters(stated, pool) or rewrites(accepted, stated)
        if why is None and check is not None:
            try:
                why = check(stated, accepted)
            except Exception as exc:  # chain trouble: unchecked, on the auditor's word
                log.warning("  referrals: the record of period %d could not be checked on chain: %s", wanted, exc)
        if why:
            log.error("  referrals: the auditor's record of period %d is refused (%s): no referral weights this "
                      "period", wanted, why)
            state.referral_refused = {"first_block": blocks[0], "end_block": blocks[1], "why": why}
        else:
            state.referral_record, state.referral_refused, accepted = stated.model_dump(mode="json"), None, stated
    return accepted if accepted is not None and pays(accepted, blocks) else None


def referral_weights(record: ReferralRecord | None, cat: Catalog, epoch: int, hotkey_of_uid: Mapping[int, str],
                     capacity: Mapping[int, float]) -> tuple[dict[int, float], dict[str, ReferrerStatus]]:
    """The referral pool's weight per UID in ``epoch`` from the record of the period before (see the module), and
    every referrer's status; ``capacity``: the epoch's score weights per UID. The weights sum to at most
    ``1 - miner_share``."""
    pool = cat.scoring.referral_pool
    if pool is None or record is None or not pays(record, period_blocks(pool, cat.epoch_blocks,
                                                                         pool.period(epoch) - 1)):
        return {}, {}
    room = round(1.0 - cat.scoring.miner_share_at(epoch), SHARE_DIGITS)
    scale = min(1.0, room / record.share) if record.share > 0 else 0.0
    excluded, uid_of = set(pool.excluded), {hotkey: uid for uid, hotkey in hotkey_of_uid.items()}
    codes: dict[str, dict[str, list[int]]] = {}
    for c in record.codes:
        codes.setdefault(c.hotkey, {})[c.code] = list(c.periods)
    weights: dict[int, float] = {}
    referrers: dict[str, ReferrerStatus] = {}
    for r in sorted(record.referrers, key=lambda r: r.hotkey):
        uid = uid_of.get(r.hotkey)
        unpaid = None
        if uid is None:
            unpaid = "holds no UID"
        elif r.hotkey == record.owner_hotkey or (r.coldkey is not None and r.coldkey == record.owner_coldkey):
            unpaid = "the subnet owner's"
        elif r.hotkey in excluded or r.coldkey in excluded:
            unpaid = "excluded"
        elif pool.serving_only and capacity.get(uid, 0.0) <= 0:
            unpaid = "not serving"
        elif r.share * scale > 0:
            weights[uid] = weights.get(uid, 0.0) + r.share * scale
        referrers[r.hotkey] = ReferrerStatus(codes=codes.get(r.hotkey, {}), referred_micros=r.referred_micros,
                                             share=r.share, unpaid=unpaid)
    return weights, referrers


def referral_period(record: ReferralRecord, cat: Catalog, epoch: int) -> ReferralPeriod:
    """What the network document states of the record the epoch pays."""
    room = round(1.0 - cat.scoring.miner_share_at(epoch), SHARE_DIGITS)
    caps = sum(r.cap for r in record.referrers)
    rate = record.payout_per_usd * (min(1.0, record.share / caps) if caps > 0 else 1.0) * (
        min(1.0, room / record.share) if record.share > 0 else 1.0)
    return ReferralPeriod(period=record.period, first_block=record.first_block, end_block=record.end_block,
                          window_periods=record.window_periods, payout_per_usd=record.payout_per_usd,
                          share=record.share, referred_micros=sum(r.referred_micros for r in record.referrers),
                          rate_per_usd=rate)
