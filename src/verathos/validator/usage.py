"""The usage pool on the validators' side (``catalog.UsagePool``, ``throughput.usage``).

Each epoch of period ``k + 1`` pays period ``k``'s pool: its record comes signed with the auditor's verdicts. A
validator takes the record once per period: it recomputes its figures and, unless it is the auditor, checks what its
node still holds against it; a record that fails either is refused for the whole period (no usage weights; the owner's
UID keeps the share). The record is kept with the scores, so an epoch whose verdicts do not come still pays it.

The epoch's usage share is the record's share, at most what the epoch's miner share and the referral pool's share
leave (``Scoring.usage_cap_at``). Each hotkey the record lists gets ``usage share x what users paid for its answers /
what they paid in all`` (its paid input tokens at the record's prices), on the UID it holds now: hotkeys the
catalog's ``excluded`` names, by hotkey or by the coldkey the record states, and hotkeys that hold no UID now get
nothing, and their part stays on the subnet owner's UID.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from verathos.catalog import Catalog
from verathos.throughput.feed import Verdicts
from verathos.throughput.usage import UsageRecord, period_blocks, record_of
from verathos.validator.scoring import ScoreState

log = logging.getLogger("verathos.validator")


def take(state: ScoreState, verdicts: Verdicts | None, cat: Catalog, epoch: int,
         check: Callable[[UsageRecord], str | None] | None = None) -> UsageRecord | None:
    """The record the epoch pays (the period before its own), from ``verdicts`` the first time they state it, else as
    kept in ``state``; None without one, or if it was refused. ``check`` gives why the chain contradicts a record
    (None: not checked)."""
    pool = cat.scoring.usage_pool
    if pool is None:
        return None
    wanted = pool.period(epoch) - 1
    blocks = period_blocks(pool, cat.epoch_blocks, wanted)
    kept = state.usage_record or {}
    stated = record_of((verdicts.model_extra or {}).get("usage")) if verdicts is not None else None
    if stated is not None and pays(stated, blocks) and (kept.get("first_block"), kept.get("end_block")) != blocks:
        why = stated.inconsistency()
        if why is None and check is not None:
            try:
                why = check(stated)
            except Exception as exc:  # chain trouble: unchecked, on the auditor's word
                log.warning("  usage pool: the record of period %d could not be checked on chain: %s", wanted, exc)
        if why:
            log.error("  usage pool: the auditor's record of period %d is refused (%s): no usage weights this period",
                      wanted, why)
        state.usage_record, state.usage_refused = stated.model_dump(mode="json"), why
        kept = state.usage_record
    if (kept.get("first_block"), kept.get("end_block")) != blocks or state.usage_refused:
        return None
    return UsageRecord.model_validate(kept)


def pays(record: UsageRecord, blocks: tuple[int, int]) -> bool:
    """Whether ``record`` is of the period of ``blocks`` (its first block, the next period's)."""
    return (record.first_block, record.end_block) == blocks


def usage_weights(record: UsageRecord | None, cat: Catalog, epoch: int,
                  hotkey_of_uid: dict[int, str]) -> dict[int, float]:
    """The usage pool's weight per UID in ``epoch`` from the record of the period before (see the module); they sum
    to at most ``Scoring.usage_cap_at``."""
    pool = cat.scoring.usage_pool
    if pool is None or record is None or not pays(record, period_blocks(pool, cat.epoch_blocks,
                                                                         pool.period(epoch) - 1)):
        return {}
    share = min(record.share, cat.scoring.usage_cap_at(epoch))
    paid = {s.hotkey: s.paid_usd(record.prices) for s in record.served}
    total = sum(paid.values())
    if share <= 0 or total <= 0:
        return {}
    excluded = set(pool.excluded)
    uid_of = {hotkey: uid for uid, hotkey in hotkey_of_uid.items()}
    out: dict[int, float] = {}
    for served in sorted(record.served, key=lambda s: s.hotkey):
        uid = uid_of.get(served.hotkey)
        if uid is None or paid[served.hotkey] <= 0 or served.hotkey in excluded or served.coldkey in excluded:
            continue
        out[uid] = out.get(uid, 0.0) + share * paid[served.hotkey] / total
    return out
