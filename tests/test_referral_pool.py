"""The referral pool: the auditor's record of what referred accounts topped up, the cap per referred USD, the
validators' checks of the record (its sums, the chain, and the record before), the referral weights and who gets
them, and the network document that lists referrers with the parts of their weight."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from tests.test_light import CAT as LIGHT_CAT
from tests.test_light import auditor_and_light
from tests.test_referral_gateway import USD, Ledger, move, registration
from tests.test_throughput import FakeChain, kp
from tests.test_usage_pool import SN96, STORAGE, Substrate
from tests.tlsfake import Presenting, pin
from verathos import catalog
from verathos.gateway.referrals import ReferralReporter
from verathos.protocol import signing, tls
from verathos.protocol.gateway import ReferralEntry, ReferralReport
from verathos.protocol.referral import SignedMove, SignedRegistration
from verathos.protocol.status import NetworkStatus
from verathos.throughput.auditor import Auditor
from verathos.throughput.feed import VERDICTS_DOMAIN, Verdicts
from verathos.throughput.policy import AuditPolicy
from verathos.throughput.referral import (
    CodeState,
    Move,
    close,
    contradiction,
    follow,
    owner_at,
    record_of,
    sized,
)
from verathos.validator import status as status_module
from verathos.validator.__main__ import weight_parts, weights_to_set
from verathos.validator.referral import referral_period, referral_weights, rewrites, take
from verathos.validator.scoring import ScoreState

POOL_SETTINGS = {"period_epochs": 2, "window_periods": 3, "term_days": 30, "account_cap_usd": 1000}
L = 2 * 180  # blocks per period
TAO_USD, ALPHA_TAO = 300.0, 0.004
EMISSION_USD = SN96.miner_alpha_per_block() * L * TAO_USD * ALPHA_TAO  # a period's miner emission: 177.12 USD
OWNER_HOTKEY, OWNER_COLDKEY = kp().ss58_address, kp().ss58_address
COLD_A, COLD_B = kp().ss58_address, kp().ss58_address


def pooled(settings: dict | None = None, miner_share=0.59, base: catalog.Catalog | None = None) -> catalog.Catalog:
    document = (base or catalog.load()).document()
    scoring = {**document["scoring"], "miner_share": miner_share,
               "referral_pool": {**POOL_SETTINGS, **(settings or {})}}
    return catalog.Catalog.parse({**document, "scoring": scoring})


CAT = pooled()
POOL = CAT.scoring.referral_pool
HOTKEYS = {"alice": "hkA", "al2": "hkA", "bob": "hkB", "carol": "hkC", "dave": "hkD", "self": "hkS"}


def rec(period: int, entries, *, cat: catalog.Catalog = CAT, hotkeys=None, coldkeys=None,
        owner=(OWNER_HOTKEY, OWNER_COLDKEY), prices=(TAO_USD, ALPHA_TAO), since_block: int = 0):
    """The record of ``period`` of ``entries`` (code, block, USD), as the auditor closes it."""
    record = close(period, pool=cat.scoring.referral_pool, epoch_blocks=cat.epoch_blocks,
                   entries=[(code, block, round(usd * USD)) for code, block, usd in entries],
                   hotkeys=hotkeys or HOTKEYS, coldkeys=coldkeys or {}, owner=owner, emission=SN96,
                   tao_usd=prices[0], alpha_tao=prices[1], since_block=since_block)
    assert record.inconsistency() is None
    return record


def cap(usd: float) -> float:
    """The share of a period's miner emission that pays payout_per_usd / window_periods of ``usd``."""
    return 0.3 * usd / (3 * EMISSION_USD)


# ---- the record ----

def test_each_referrer_is_paid_its_cap_below_the_pool_and_a_pro_rata_part_of_it_at_the_pool() -> None:
    entries = [("alice", 2 * L + 10, 100), ("alice", 3 * L + 10, 20), ("bob", 4 * L + 10, 50),
               ("carol", 2 * L - 1, 999), ("bob", 5 * L, 70)]  # carol's is before the window, bob's 70 after it
    r = rec(4, entries)
    assert (r.first_block, r.end_block, r.window_periods) == (4 * L, 5 * L, 3)
    assert [(c.code, c.hotkey, c.periods) for c in r.codes] == [("alice", "hkA", [100 * USD, 20 * USD, 0]),
                                                               ("bob", "hkB", [0, 0, 50 * USD])]
    assert {x.hotkey: x.referred_micros for x in r.referrers} == {"hkA": 120 * USD, "hkB": 50 * USD}
    assert {x.hotkey: x.cap for x in r.referrers} == pytest.approx({"hkA": cap(120), "hkB": cap(50)})
    assert r.share == pytest.approx(cap(170)) and r.share < 0.1  # below the pool: each its cap
    assert {x.hotkey: x.share for x in r.referrers} == pytest.approx({"hkA": cap(120), "hkB": cap(50)})
    assert (r.owner_hotkey, r.owner_coldkey, r.tao_usd, r.alpha_tao) == (OWNER_HOTKEY, OWNER_COLDKEY, 300.0, 0.004)

    full = rec(4, [*entries, ("dave", 4 * L + 20, 200)])
    assert full.share == 0.1 and sum(cap(u) for u in (120, 50, 200)) > 0.1  # the pool's share, pro rata
    assert {x.hotkey: x.share for x in full.referrers} == pytest.approx(
        {h: cap(u) * 0.1 / sum(cap(v) for v in (120, 50, 200)) for h, u in (("hkA", 120), ("hkB", 50), ("hkD", 200))})
    design = sized({"A": 6000 * USD, "B": 3000 * USD, "C": 1000 * USD}, payout_per_usd=0.3, window_periods=30,
                   max_share=0.1, tao_usd=274.92, alpha_tao=0.004223, period_emission=2952.0)  # the design's example
    assert {h: round(c, 4) for h, c in design[0].items()} == {"A": 0.0175, "B": 0.0088, "C": 0.0029}
    assert design[1] == pytest.approx(sum(design[0].values()))
    nothing = rec(4, [("carol", 10, 999)], prices=(None, ALPHA_TAO))
    assert (nothing.codes, nothing.referrers, nothing.share) == ([], [], 0.0)  # nothing referred: no TAO price needed


def test_a_refund_counts_negative_and_a_referrers_sum_is_never_below_zero() -> None:
    r = rec(4, [("alice", 2 * L + 1, 100), ("alice", 3 * L + 1, -100), ("al2", 4 * L + 1, 50)])
    assert [(c.code, c.periods) for c in r.codes] == [("al2", [0, 0, 50 * USD]), ("alice", [100 * USD, -100 * USD, 0])]
    assert [(x.hotkey, x.referred_micros) for x in r.referrers] == [("hkA", 50 * USD)]
    refunded = rec(4, [("bob", 2 * L + 1, 10), ("bob", 4 * L + 1, -30), ("alice", 3 * L + 1, 5)])
    assert [(x.hotkey, x.referred_micros, x.cap) for x in refunded.referrers] == [
        ("hkA", 5 * USD, pytest.approx(cap(5))), ("hkB", 0, 0.0)]  # what a refund returned beyond is not taken back
    netted = rec(4, [("bob", 3 * L + 1, 10), ("bob", 3 * L + 2, -10)])
    assert netted.codes == [] and netted.share == 0.0


def test_a_record_whose_figures_do_not_follow_from_its_inputs_is_inconsistent() -> None:
    r = rec(4, [("alice", 2 * L + 10, 100), ("bob", 4 * L + 10, 50)], coldkeys={"hkA": COLD_A, "hkB": COLD_B})
    alice, bob = r.codes
    first = r.referrers[0]
    for change, why in (
            ({"share": r.share * 2}, "share"),
            ({"referrers": [first.model_copy(update={"cap": first.cap * 2}), r.referrers[1]]}, "caps"),
            ({"referrers": [first.model_copy(update={"share": 0.09}), r.referrers[1]]}, "caps"),
            ({"referrers": [first.model_copy(update={"referred_micros": 1}), r.referrers[1]]}, "referrers"),
            ({"referrers": [first]}, "referrers"),
            ({"referrers": [first.model_copy(update={"coldkey": COLD_B}), r.referrers[1]]}, "referrers"),
            ({"codes": [alice, alice.model_copy(update={"code": "al2", "coldkey": COLD_B}), bob]}, "two coldkeys"),
            ({"codes": [alice, alice, bob]}, "twice"),
            ({"codes": [alice.model_copy(update={"periods": [1, 2]}), bob]}, "window"),
            ({"tao_usd": None}, "prices"), ({"alpha_tao": 0.0}, "prices"),
            ({"period_emission": 1.0}, "emission"), ({"first_block": r.end_block}, "ends before")):
        assert why in r.model_copy(update=change).inconsistency(), why
    with pytest.raises(ValueError, match="payout_per_usd"):
        type(r).model_validate({**r.model_dump(), "payout_per_usd": 0.95})  # never above 0.9, whatever it says
    assert record_of({"period": 1}) is None and record_of("x") is None and record_of(r.model_dump()) == r


# ---- what a referred dollar earns ----

def paid_usd(records: dict[int, object], cat: catalog.Catalog, uid: int, hotkeys: dict[int, str]) -> float:
    """USD of emission ``uid`` is paid over the epochs that pay ``records`` (period -> record)."""
    pool = cat.scoring.referral_pool
    total = 0.0
    for period, record in records.items():
        for epoch in range((period + 1) * pool.period_epochs, (period + 2) * pool.period_epochs):
            weights, _ = referral_weights(record, cat, epoch, hotkeys, {})
            total += weights.get(uid, 0.0) * SN96.miner_alpha_per_block() * cat.epoch_blocks * TAO_USD * ALPHA_TAO
    return total


@pytest.mark.parametrize("own, rivals, miner_share", [(10, 0, 0.59), (1, 0, 0.59), (59, 0, 0.59), (500, 0, 0.59),
                                                       (50, 150, 0.59), (50, 0, 0.98), (5000, 1, 0.15)])
def test_self_referral_returns_at_most_payout_per_usd_per_dollar_whatever_the_timing_or_the_pool(
        own, rivals, miner_share) -> None:
    """A referrer topping up its own account with its own code: whether nobody else refers (an empty pool), others do,
    the top-up fills the pool on its own, or the miner share leaves little room, a dollar returns at most 0.3 of
    emission over its window; alone and below the pool, exactly that."""
    cat = pooled(miner_share=miner_share)
    hotkeys = {0: OWNER_HOTKEY, 1: "hkS", 2: "hkB"}
    for arrival in (2 * L, 3 * L - 1):  # at the start or the end of its period
        entries = [("self", arrival, own), ("bob", arrival, rivals)] if rivals else [("self", arrival, own)]
        records = {k: rec(k, entries, cat=cat) for k in range(1, 8)}  # before, through and after its window
        earned = paid_usd(records, cat, 1, hotkeys)
        assert earned <= 0.3 * own * (1 + 1e-12)
        if not rivals and cap(own) <= min(0.1, 1 - miner_share):
            assert earned == pytest.approx(0.3 * own, rel=1e-9)
    split = {k: rec(k, [("self", 2 * L, own / 2), ("al2", 2 * L + 9, own / 2)], cat=cat,
                    hotkeys={**HOTKEYS, "al2": "hkS"}) for k in range(1, 8)}
    assert paid_usd(split, cat, 1, hotkeys) <= 0.3 * own * (1 + 1e-12)  # two codes or accounts: no more


# ---- the validators ----

def verdicts_of(record) -> Verdicts:
    return Verdicts(netuid=405, auditor="a", epoch=0, credits=[], invalid=[], referral=record.model_dump(mode="json"))


def test_a_validator_takes_a_record_once_per_period_and_refuses_one_that_rewrites_an_earlier_period(caplog) -> None:
    caplog.set_level(logging.ERROR, logger="verathos.validator")
    history = [("alice", 1 * L + 5, 40), ("bob", 2 * L + 5, 30), ("alice", 3 * L + 5, 10)]
    r2, r3, r4 = (rec(k, history) for k in (2, 3, 4))
    state, checked = ScoreState(), []

    def check(record, previous):
        checked.append(record.period)
        return None
    assert take(state, verdicts_of(r2), CAT, 6, check) == r2  # epochs 6 and 7 pay period 2
    assert take(state, None, CAT, 7, check) == r2 and take(state, verdicts_of(r2), CAT, 7, check) == r2
    assert checked == [2]  # taken once
    rewritten = rec(3, [*history, ("bob", 2 * L + 9, 5)])  # period 2 holds 5 USD more than it did
    assert take(state, verdicts_of(rewritten), CAT, 8, check) is None
    assert "rewrites the sum of code bob in the period ending at block 1080" in state.referral_refused["why"]
    assert "is refused" in caplog.text and checked == [2]
    assert take(state, verdicts_of(r3), CAT, 9) is None  # refused for the whole period
    kept = rec(4, [*history, ("bob", 2 * L + 9, 5)])
    assert take(state, verdicts_of(kept), CAT, 10) is None  # the rewrite kept: still against the record accepted
    assert take(ScoreState(), verdicts_of(kept), CAT, 10) == kept  # a validator without that record cannot tell
    assert take(state, verdicts_of(r4), CAT, 10) is None  # period 4's was refused already this period
    later = rec(5, history)
    assert take(state, verdicts_of(later), CAT, 12) == later  # the honest history goes on
    assert take(state, verdicts_of(r3), CAT, 14) is None  # a record of another period pays nothing
    assert take(ScoreState(), verdicts_of(r2), pooled({"window_periods": 4}), 6) is None  # not the catalog's window
    assert "other parameters" in take_why(r2, pooled({"payout_per_usd": 0.2}))
    assert take_why(r2, CAT, lambda r, previous: "the chain has no such owner") == "the chain has no such owner"
    tampered = verdicts_of(r2).model_copy(update={"referral": {**r2.model_dump(mode="json"), "share": 0.3}})
    assert take(ScoreState(), tampered, CAT, 6) is None

    def unreachable(record, previous):
        raise ConnectionError("node down")
    assert take(ScoreState(), verdicts_of(r2), CAT, 6, unreachable) == r2  # unchecked: the auditor's word
    assert take(ScoreState(), verdicts_of(r2), catalog.load(), 6) is None  # no pool: nothing


def take_why(record, cat, check=None) -> str | None:
    state = ScoreState()
    take(state, verdicts_of(record), cat, (record.period + 1) * cat.scoring.referral_pool.period_epochs, check)
    return (state.referral_refused or {}).get("why")


def test_a_code_moves_once_only_to_a_hotkey_of_its_coldkey_and_stays_there(caplog) -> None:
    """The auditor decides each move once, in the order things arrived, by the coldkey a code belongs to: the coldkey
    of the hotkey that registered it, followed through the chain's coldkey swaps. A hotkey swapped away later, or a
    code registered after a move, changes nothing decided."""
    codes = [CodeState("alice", "hkA", 1, 10), CodeState("bob", "hkB", 2, 10), CodeState("carol", "hkC", 3, 10)]
    moves = [Move(4, "hkA", "hkB"), Move(5, "hkB", "hkX"), Move(6, "hkC", "hkZ"), Move(7, "hkQ", "hkA")]
    states, outcomes = follow(codes, moves, {"hkA": COLD_A, "hkB": COLD_A, "hkC": COLD_B, "hkX": COLD_B}, 2 * L)
    assert states == {"alice": ("hkB", COLD_A), "bob": ("hkB", COLD_A), "carol": ("hkC", COLD_B)}
    assert outcomes == {4: "moved", 5: "refused", 6: "refused", 7: "empty"}  # hkZ belongs to no coldkey
    late = [CodeState("alice", "hkA", 1, 10, "hkB", COLD_A), CodeState("zed", "hkA", 9, 4 * L + 1)]
    swapped = follow(late, [], {"hkB": COLD_A}, 4 * L)[0]  # zed arrived after the period: not yet
    assert swapped == {"alice": ("hkB", COLD_A)}
    after_move = follow([CodeState("alice", "hkA", 1, 10), CodeState("zed", "hkA", 3, 10)], [Move(2, "hkA", "hkB")],
                        {"hkA": COLD_A, "hkB": COLD_A}, 2 * L)[0]
    assert after_move == {"alice": ("hkB", COLD_A), "zed": ("hkA", COLD_A)}  # registered after the move: stays

    # a hotkey swap: the old hotkey has no owner any more, the new one has the code's coldkey
    rotated = follow([CodeState("alice", "hkA", 1, 10, "hkA", COLD_A)], [Move(2, "hkA", "hkN")],
                     {"hkA": None, "hkN": COLD_A}, 3 * L)
    assert rotated == ({"alice": ("hkN", COLD_A)}, {2: "moved"})
    # a coldkey swap, then a move to a hotkey of the new coldkey in the same period
    swap = follow([CodeState("alice", "hkA", 1, 10, "hkA", COLD_A)], [Move(2, "hkA", "hkB")],
                  {"hkA": COLD_B, "hkB": COLD_B}, 3 * L)
    assert swap == ({"alice": ("hkB", COLD_B)}, {2: "moved"})
    # once moved, a later swap of the hotkey it left changes nothing
    assert follow([CodeState("alice", "hkA", 1, 10, "hkB", COLD_A)], [], {"hkB": COLD_A, "hkA": None},
                  4 * L)[0] == {"alice": ("hkB", COLD_A)}


def test_a_full_validator_refuses_a_record_that_moves_a_code_to_another_coldkey(caplog) -> None:
    """Against the last record it accepted and its node at the record's end block: a code now on another hotkey is
    refused when its previous hotkey belongs to another coldkey than its new one there; a move within a coldkey, after
    a coldkey swap, or from a hotkey swapped away (no owner) is taken."""
    history = [("alice", 2 * L + 5, 40)]
    before = rec(2, history, coldkeys={"hkA": COLD_A})

    def node(owners: dict[str, str | None]) -> SimpleNamespace:
        storage = {**STORAGE, "SubnetOwnerHotkey": OWNER_HOTKEY, "SubnetOwner": OWNER_COLDKEY,
                   "Owner": lambda params: owners.get(params[0])}
        return SimpleNamespace(substrate=Substrate({}, storage))
    stolen = rec(3, history, hotkeys={"alice": "hkX"}, coldkeys={"hkX": COLD_B})
    why = contradiction(stolen, node({"hkA": COLD_A, "hkX": COLD_B}), 405, before)
    assert why.startswith("it moves code alice from hkA to hkX, a hotkey of another coldkey")
    state = ScoreState()
    take(state, verdicts_of(before), CAT, 6)
    assert take(state, verdicts_of(stolen), CAT, 8, lambda r, previous: contradiction(
        r, node({"hkA": COLD_A, "hkX": COLD_B}), 405, previous)) is None
    within = rec(3, history, hotkeys={"alice": "hkB"}, coldkeys={"hkB": COLD_A})
    assert contradiction(within, node({"hkA": COLD_A, "hkB": COLD_A}), 405, before) is None
    swapped = rec(3, history, hotkeys={"alice": "hkB"}, coldkeys={"hkB": COLD_B})  # hkA's coldkey swapped to COLD_B
    assert contradiction(swapped, node({"hkA": COLD_B, "hkB": COLD_B}), 405, before) is None
    rotated = rec(3, history, hotkeys={"alice": "hkN"}, coldkeys={"hkN": COLD_A})
    assert contradiction(rotated, node({"hkA": None, "hkN": COLD_A}), 405, before) is None
    assert contradiction(stolen, node({"hkA": COLD_A, "hkX": COLD_B}), 405, rec(0, history)) is None  # no overlap
    assert rewrites(before, stolen) is None  # the sums are as they were: the chain tells a move apart


def test_referral_weights_pay_registered_referrers_within_what_the_miner_share_leaves() -> None:
    fleet = kp().ss58_address
    cat = pooled({"excluded": [fleet, COLD_B]})
    hotkeys = {"alice": "hkA", "bob": "hkB", "carol": "hkC", "dave": OWNER_HOTKEY, "erin": "hkE", "fred": fleet,
               "gina": "hkG"}
    r = rec(2, [(code, 2 * L + 1, 30) for code in hotkeys], cat=cat, hotkeys=hotkeys,
            coldkeys={"hkC": COLD_B, "hkE": OWNER_COLDKEY})
    uids = {0: OWNER_HOTKEY, 1: "hkA", 2: "hkB", 3: "hkC", 4: "hkE", 5: fleet}  # hkG holds no UID
    weights, referrers = referral_weights(r, cat, 6, uids, capacity={2: 0.5})
    share = r.referrers[0].share
    assert weights == {1: pytest.approx(share), 2: pytest.approx(share)}  # serving or not, any registered hotkey
    assert {h: s.unpaid for h, s in referrers.items()} == {
        "hkA": None, "hkB": None, "hkC": "excluded", OWNER_HOTKEY: "the subnet owner's", "hkE": "the subnet owner's",
        fleet: "excluded", "hkG": "holds no UID"}
    assert referrers["hkA"].codes == {"alice": [0, 0, 30 * USD]} and referrers["hkA"].referred_micros == 30 * USD
    serving = pooled({"serving_only": True})
    assert referral_weights(rec(2, [("alice", 2 * L, 30), ("bob", 2 * L, 30)], cat=serving), serving, 6,
                            uids, {2: 0.5})[0] == {2: pytest.approx(cap(30))}
    assert referral_weights(r, cat, 8, uids, {})[0] == {}  # the next period needs the next record

    w, parts = weight_parts({2: 1.0}, uids, OWNER_HOTKEY, 0.59, None, weights)
    assert w == pytest.approx({0: 0.41 - 2 * share, 1: share, 2: 0.59 + share}) and sum(w.values()) == pytest.approx(1)
    assert parts == {"serving": {2: 0.59}, "usage": {}, "referral": weights}
    assert weights_to_set({2: 1.0}, uids, OWNER_HOTKEY, 0.59, None, weights) == w
    big = rec(2, [("alice", 2 * L, 100_000)], cat=pooled(miner_share=0.95))
    tight = referral_weights(big, pooled(miner_share=0.95), 6, uids, {})[0]
    assert big.share == 0.1 and tight == {1: pytest.approx(0.05)}  # cut to what the miner share leaves
    period = referral_period(big, pooled(miner_share=0.95), 6)
    assert period.rate_per_usd == pytest.approx(0.3 * 0.1 / big.referrers[0].cap * 0.5)
    assert referral_period(r, cat, 6).rate_per_usd == pytest.approx(0.3 * 0.1 / sum(x.cap for x in r.referrers))
    small = rec(2, [("alice", 2 * L, 30)], cat=cat)
    assert referral_period(small, cat, 6).rate_per_usd == pytest.approx(0.3)  # below the pool: the full rate


def test_a_validator_checks_the_reference_block_its_node_still_holds() -> None:
    storage = {**STORAGE, "SubnetOwnerHotkey": OWNER_HOTKEY, "SubnetOwner": OWNER_COLDKEY,
               "Owner": lambda params: {"hkA": COLD_A}.get(params[0])}
    st = SimpleNamespace(substrate=Substrate({}, storage))
    assert owner_at(st, 5 * L, 405) == (OWNER_HOTKEY, OWNER_COLDKEY)
    good = rec(4, [("alice", 3 * L, 10)], coldkeys={"hkA": COLD_A})
    assert contradiction(good, st, 405) is None
    for change, why in (({"alpha_tao": 0.008}, "price"), ({"owner_coldkey": COLD_B}, "subnet owner"),
                        ({"referrers": [good.referrers[0].model_copy(update={"coldkey": COLD_B})]}, "coldkey"),
                        ({"emission": SN96.model_copy(update={"owner_cut": 0})}, "emission")):
        assert why in contradiction(good.model_copy(update=change), st, 405), why
    pruned = SimpleNamespace(substrate=Substrate({}, storage, pruned={5 * L}))
    assert contradiction(good.model_copy(update={"owner_coldkey": COLD_B}), pruned, 405) is None  # not held: unchecked


# ---- the auditor ----

class ReferralChain(FakeChain):
    """The chain as the auditor's referral pool reads it."""

    def __init__(self, head: int, coldkeys: dict[str, str] | None = None) -> None:
        super().__init__(head)
        self.coldkey_of = coldkeys or {}

    def emission(self, block: int, netuid: int, mechanism: int):
        return SN96.model_copy(update={"mechanism": mechanism})

    def alpha_price(self, block: int, netuid: int) -> float:
        return ALPHA_TAO

    def coldkeys(self, hotkeys, block: int) -> dict[str, str | None]:
        return {hotkey: self.coldkey_of.get(hotkey) for hotkey in hotkeys}

    def owner(self, block: int, netuid: int) -> tuple[str, str]:
        return OWNER_HOTKEY, OWNER_COLDKEY


def referral_auditor(path, cat: catalog.Catalog, chain: ReferralChain) -> Auditor:
    path.mkdir(parents=True, exist_ok=True)
    return Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                   catalog=lambda: cat, references={}, db_path=path / "t.sqlite", tao_usd=lambda: TAO_USD)


def at(auditor: Auditor, block: int) -> None:
    """The chain at ``block``, final; the auditor's loops step once."""
    auditor.chain.head_block = auditor.chain.finalized = block
    auditor.follower.poll()
    auditor.settle()
    auditor.step_referrals()


def signed_code(keypair, code: str) -> SignedRegistration:
    return SignedRegistration.model_validate(registration(keypair, code))


def entry(code: str, usd: float, n: int) -> ReferralEntry:
    return ReferralEntry(id=f"{n:064x}", code=code, amount_micros=round(usd * USD), credited_ms=1)


def test_the_auditor_takes_each_reported_item_once_and_records_each_period_once_its_end_is_final(tmp_path) -> None:
    alice_kp, bob_kp, new_kp = kp(), kp(), kp()
    chain = ReferralChain(2 * L + 1, {alice_kp.ss58_address: COLD_A, new_kp.ss58_address: COLD_A,
                                      bob_kp.ss58_address: COLD_B})
    auditor = referral_auditor(tmp_path, CAT, chain)
    at(auditor, 2 * L + 1)
    report = ReferralReport(codes=[signed_code(alice_kp, "alice"), signed_code(bob_kp, "bobby")],
                            entries=[entry("alice", 40, 1), entry("bobby", 10, 2), entry("nobody", 5, 3)])
    assert auditor.accept_referrals("gw", report) == {"codes": 2, "moves": 0, "entries": 2, "refused": 0,
                                                      "unknown_codes": ["nobody"]}
    assert auditor.accept_referrals("gw", report)["entries"] == 0  # sent again: taken once
    forged = signed_code(alice_kp, "carol").model_copy(update={"signature": signed_code(bob_kp, "carol").signature})
    other_netuid = SignedRegistration.create(signed_code(bob_kp, "dan1").registration.model_copy(
        update={"netuid": 96}), bob_kp)
    moved = SignedMove.model_validate(move(new_kp, alice_kp.ss58_address, new_kp.ss58_address))
    across = SignedMove.model_validate(move(bob_kp, new_kp.ss58_address, bob_kp.ss58_address))
    taken = auditor.accept_referrals("gw", ReferralReport(codes=[forged, other_netuid], moves=[moved, across],
                                                          entries=[entry("bobby", -4, 4)]))
    assert taken == {"codes": 0, "moves": 2, "entries": 1, "refused": 2, "unknown_codes": []}
    at(auditor, 2 * L + 200)
    assert auditor.store.referral_record(2 * L, 3 * L) is None  # period 2 is not over
    at(auditor, 3 * L + 3)  # period 3 begins: period 2 closes once its end block is final
    signed = auditor.verdicts(6)
    assert signing.verify(auditor.hotkey, VERDICTS_DOMAIN, signed.verdicts.model_dump(mode="json"), signed.signature)
    r = record_of(signed.verdicts.model_extra["referral"])
    assert (r.period, r.first_block, r.end_block, r.payout_per_usd, r.max_share) == (2, 2 * L, 3 * L, 0.3, 0.1)
    assert [(c.code, c.hotkey, c.coldkey, c.periods) for c in r.codes] == [
        ("alice", new_kp.ss58_address, COLD_A, [0, 0, 40 * USD]),  # moved within its coldkey
        ("bobby", bob_kp.ss58_address, COLD_B, [0, 0, 6 * USD])]  # the move to bob's coldkey is refused
    assert (r.tao_usd, r.alpha_tao, r.owner_hotkey) == (TAO_USD, ALPHA_TAO, OWNER_HOTKEY) and r.inconsistency() is None
    at(auditor, 7 * 180 + 1)
    assert record_of(auditor.verdicts(7).verdicts.model_extra["referral"]) == r  # every epoch of period 3 states it

    late = auditor.accept_referrals("gw", ReferralReport(entries=[entry("bobby", 1, 5)]))
    assert late["entries"] == 1  # after the close: it counts in period 3, never in the closed period 2
    for block in range(3 * L + 103, 4 * L, 100):
        at(auditor, block)
    at(auditor, 4 * L + 2)
    r3 = record_of(auditor.verdicts(8).verdicts.model_extra["referral"])
    assert [c.periods for c in r3.codes] == [[0, 40 * USD, 0], [0, 6 * USD, 1 * USD]]
    assert rewrites(r, r3) is None and take(ScoreState(), Verdicts.model_validate(
        auditor.verdicts(8).verdicts.model_dump()), CAT, 8) == r3


def test_the_first_verdicts_of_a_period_wait_briefly_for_its_record(tmp_path) -> None:
    chain = ReferralChain(2 * L + 5)
    auditor = referral_auditor(tmp_path, CAT, chain)
    auditor.tao_usd = lambda: (_ for _ in ()).throw(ConnectionError("no price"))
    assert auditor.accept_referrals("gw", ReferralReport(codes=[signed_code(kp(), "bobby")])) is None  # no head yet
    auditor.follower.poll()
    auditor.accept_referrals("gw", ReferralReport(codes=[signed_code(kp(), "alice")], entries=[entry("alice", 1, 1)]))
    chain.head_block = chain.finalized = 3 * L + 2
    auditor.follower.poll()
    auditor.step_referrals()  # period 2 would close, but TAO's price cannot be read: tried again
    assert auditor.store.referral_record(2 * L, 3 * L) is None and auditor.verdicts(6) is None
    chain.head_block = 3 * L + 25
    auditor.follower.poll()
    assert auditor.verdicts(6).verdicts.model_extra.get("referral") is None  # then the verdicts come without it


def test_a_new_period_length_or_window_starts_the_windows_afresh_so_no_dollar_counts_twice(tmp_path) -> None:
    catalogs = [CAT]
    chain = ReferralChain(2 * L + 1, {})
    auditor = Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                      catalog=lambda: catalogs[-1], references={}, db_path=tmp_path / "t.sqlite",
                      tao_usd=lambda: TAO_USD)
    at(auditor, 2 * L + 1)
    auditor.accept_referrals("gw", ReferralReport(codes=[signed_code(kp(), "alice")], entries=[entry("alice", 9, 1)]))
    at(auditor, 3 * L + 1)
    first = record_of(auditor.verdicts(6).verdicts.model_extra["referral"])
    assert first.codes[0].periods == [0, 0, 9 * USD]
    catalogs.append(pooled({"window_periods": 4}))  # a longer window would pay the 9 USD a fourth time
    for block in range(3 * L + 101, 4 * L + 2, 100):
        at(auditor, block)
    at(auditor, 4 * L + 2)
    assert record_of(auditor.verdicts(8).verdicts.model_extra["referral"]).codes == []
    auditor.accept_referrals("gw", ReferralReport(entries=[entry("alice", 2, 2)]))
    for block in range(4 * L + 101, 5 * L + 2, 100):
        at(auditor, block)
    at(auditor, 5 * L + 2)
    assert record_of(auditor.verdicts(10).verdicts.model_extra["referral"]).codes[0].periods == [0, 0, 0, 2 * USD]


def test_what_the_gateway_reports_reaches_the_auditor_once_and_its_record_pays_the_referrer(tmp_path) -> None:
    """End to end: a code registered and an account referred at the gateway, top-ups and a refund decided and posted
    (one post again after its answer was lost), the auditor's record, and the weights a validator sets from it."""
    from verathos.validator.service import add_gateway_routes

    ledger = Ledger(tmp_path / "gateway")
    gateway_kp = kp()
    chain = ReferralChain(2 * L + 1, {ledger.miner.ss58_address: COLD_A})
    auditor = referral_auditor(tmp_path / "auditor", CAT, chain)
    at(auditor, 2 * L + 1)
    app = FastAPI()
    add_gateway_routes(app, auditor, probation=list, gateways=lambda: [gateway_kp.ss58_address])
    lost = []

    class LosingAnswers(httpx.AsyncBaseTransport):
        """Passes posts on, losing the answer to the first."""

        def __init__(self) -> None:
            self.inner = Presenting({"auditor": app})

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            response = await self.inner.handle_async_request(request)
            if not lost:
                lost.append(await response.aread())
                raise httpx.ReadTimeout("the answer was lost")
            return response

    async def auditor_service():
        return tls.Service("https://auditor", pin("auditor"))

    client = httpx.AsyncClient(transport=LosingAnswers())
    reporter = ReferralReporter(ledger.referrals, gateway_kp, client, auditor=auditor_service,
                                auditor_hotkey=auditor.hotkey, catalog=lambda: CAT)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    pay = ledger.credit("u_1", 60, at_s=10)
    ledger.credit("u_1", 1500, at_s=20)  # 940 count: the account's cap
    ledger.credit("u_1", -60, at_s=30, kind="refund", refund_of=pay)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(reporter.step())
    assert b'"entries":3' in lost[0]  # the auditor took them; the gateway did not hear it
    asyncio.run(reporter.step())  # sent again as it was: taken once
    asyncio.run(reporter.step())
    at(auditor, 3 * L + 1)
    r = record_of(auditor.verdicts(6).verdicts.model_extra["referral"])
    assert [(c.code, c.hotkey, c.periods) for c in r.codes] == [
        ("alice", ledger.miner.ss58_address, [0, 0, 940 * USD])]
    hotkeys = {0: OWNER_HOTKEY, 7: ledger.miner.ss58_address}
    weights, _ = referral_weights(take(ScoreState(), auditor.verdicts(6).verdicts, CAT, 6), CAT, 6, hotkeys, {})
    assert cap(940) > 0.1 and weights == {7: pytest.approx(0.1)}  # alone beyond the pool: the pool's share
    asyncio.run(client.aclose())


def test_the_network_document_lists_referrers_with_the_parts_of_their_weight_and_light_validators_set_it(
        tmp_path, monkeypatch) -> None:
    """A referrer without endpoints is listed with its codes' sums and its referral weight; the auditor publishes the
    weights with them, and a light validator sets exactly those."""
    from verathos.validator import chain as validator_chain

    def commit(subtensor, wallet, netuid, mechid, uids, weights, **kwargs):  # the subnet has commit-reveal on
        subtensor.set.append({"netuid": netuid, "uids": uids, "weights": weights, "mechid": mechid,
                              "version_key": kwargs["version_key"]})
        return SimpleNamespace(success=True, message="")
    monkeypatch.setattr(validator_chain, "commit_timelocked_weights_extrinsic", commit)
    cat = pooled(base=LIGHT_CAT)
    record = close(49, pool=cat.scoring.referral_pool, epoch_blocks=cat.epoch_blocks,  # epoch 100 pays period 49
                   entries=[("alice", 49 * L + 5, 300 * USD), ("bob", 48 * L, 100 * USD), ("zed", 49 * L, 50 * USD)],
                   hotkeys={"alice": "hkR", "bob": "hkA", "zed": "hkZ"}, coldkeys={}, owner=(None, None),
                   emission=SN96, tao_usd=TAO_USD, alpha_tao=ALPHA_TAO)

    def verdicts(auditor: str, epoch: int) -> Verdicts:
        return Verdicts(netuid=405, auditor=auditor, epoch=epoch, credits=[], invalid=[],
                        referral=record.model_dump(mode="json"))
    chain, follower, board, auditor_epoch = auditor_and_light(tmp_path, cat, verdicts)
    owner = chain.owner
    chain.metagraph = lambda netuid, lite=True: SimpleNamespace(
        hotkeys=[owner, "hkA", "hkB", "hkR"], axons=[SimpleNamespace(ip="0.0.0.0", port=0)] * 4,
        S=[0.0] * 4, validator_permit=[True, False, False, False])
    state = ScoreState(ema={"hkA/gpu0": 0.6, "hkB/gpu0": 0.2}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    published = auditor_epoch(state, 100)
    network = NetworkStatus.model_validate_json(board.network())
    shares = {r.hotkey: r.share for r in record.referrers}
    assert network.weights == dict(zip(published["uids"], published["weights"]))
    assert network.weights == pytest.approx({0: 0.41 - shares["hkA"] - shares["hkR"], 1: 0.75 * 0.59 + shares["hkA"],
                                             2: 0.25 * 0.59, 3: shares["hkR"]})
    assert network.referral_share == round(shares["hkA"] + shares["hkR"], 6)  # hkZ holds no UID
    assert network.referral.period == 49 and network.referral.referred_micros == 450 * USD
    listed = {h.hotkey: h for h in network.hotkeys}
    assert set(listed) == {"hkA", "hkR", "hkZ"}  # referrers only: the scores list hotkeys by their endpoints
    referrer = listed["hkR"]
    assert (referrer.uid, referrer.endpoints, referrer.serving_weight) == (3, [], 0.0)
    assert referrer.referral_weight == round(shares["hkR"], 6) == referrer.weight
    assert referrer.referral.codes == {"alice": [0, 0, 300 * USD]} and referrer.referral.unpaid is None
    assert listed["hkA"].serving_weight == round(0.75 * 0.59, 6) and listed["hkA"].referral_weight > 0
    assert (listed["hkZ"].uid, listed["hkZ"].referral.unpaid) == (None, "holds no UID")
    follower.step()
    assert chain.set[-1] == published


def test_a_hotkeys_status_shows_its_referral_and_the_parts_of_its_weight() -> None:
    from verathos.protocol.status import ReferrerStatus

    referrer = ReferrerStatus(codes={"alice": [0, 5, 0]}, referred_micros=5, share=0.01)
    by_hotkey = status_module.build(epoch=7, epoch_blocks=180, endpoints=[], results={}, scores={}, failed=set(),
                                    invalid_hotkeys=set(), credited={}, scored_models=set(), state=ScoreState(),
                                    weights={3: 0.01, 0: 0.99}, parts={"serving": {}, "referral": {3: 0.01}},
                                    referral_share=0.01, referrers={"hkR": referrer}, hotkey_of_uid={3: "hkR"})
    status = by_hotkey["hkR"]
    assert (status.uid, status.weight, status.referral_weight, status.referral_share) == (3, 0.01, 0.01, 0.01)
    assert status.referral == referrer and status.epoch == 7


def test_a_move_decided_once_stands_when_the_hotkey_it_left_is_swapped_away(tmp_path) -> None:
    """A code moved within its coldkey stays moved, and the next record is taken, after the hotkey it left loses its
    owner (a hotkey swap reads as the zero account, which counts as none)."""
    a, b = kp(), kp()
    chain = ReferralChain(2 * L + 1, {a.ss58_address: COLD_A, b.ss58_address: COLD_A})
    auditor = referral_auditor(tmp_path, CAT, chain)
    at(auditor, 2 * L + 1)
    moved = SignedMove.model_validate(move(b, a.ss58_address, b.ss58_address))
    auditor.accept_referrals("gw", ReferralReport(codes=[signed_code(a, "alice")], moves=[moved],
                                                  entries=[entry("alice", 40, 1)]))
    at(auditor, 3 * L + 1)
    r2 = record_of(auditor.verdicts(6).verdicts.model_extra["referral"])
    assert [(c.hotkey, c.coldkey) for c in r2.codes] == [(b.ss58_address, COLD_A)]
    chain.coldkey_of[a.ss58_address] = None
    auditor.accept_referrals("gw", ReferralReport(entries=[entry("alice", 5, 2)]))
    for block in range(3 * L + 101, 4 * L, 100):
        at(auditor, block)
    at(auditor, 4 * L + 2)
    r3 = record_of(auditor.verdicts(8).verdicts.model_extra["referral"])
    assert [(c.hotkey, c.periods) for c in r3.codes] == [(b.ss58_address, [0, 40 * USD, 5 * USD])]
    storage = {**STORAGE, "SubnetOwnerHotkey": OWNER_HOTKEY, "SubnetOwner": OWNER_COLDKEY,
               "Owner": lambda params: {b.ss58_address: COLD_A}.get(params[0], "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1"
                                                                                 "eSuyUpnhM")}  # the zero account
    st = SimpleNamespace(substrate=Substrate({}, storage))
    state = ScoreState()
    assert take(state, verdicts_of(r2), CAT, 6) == r2
    assert take(state, verdicts_of(r3), CAT, 8, lambda r, previous: contradiction(r, st, 405, previous)) == r3


def test_what_arrives_while_a_period_is_closed_counts_in_the_next(tmp_path) -> None:
    """Once a period's close has begun, a report counts in a later period, even taken with a chain head the auditor
    has not seen pass the period's end."""
    chain = ReferralChain(2 * L + 5)
    auditor = referral_auditor(tmp_path, CAT, chain)
    auditor.follower.poll()
    auditor.accept_referrals("gw", ReferralReport(codes=[signed_code(kp(), "alice")], entries=[entry("alice", 1, 1)]))
    chain.finalized = 3 * L  # the end is final; the auditor's head is behind
    auditor.tao_usd = lambda: (_ for _ in ()).throw(ConnectionError("no price"))
    auditor.step_referrals()  # begins the close, fails: tried again
    auditor.accept_referrals("gw", ReferralReport(entries=[entry("alice", 2, 2)]))
    auditor.tao_usd = lambda: TAO_USD
    auditor.step_referrals()
    r2 = record_of(json.loads(auditor.store.referral_record(2 * L, 3 * L)))
    assert [c.periods for c in r2.codes] == [[0, 0, 1 * USD]]  # the 2 USD count in period 3


def test_with_the_owner_hotkey_holding_no_uid_the_pools_are_never_scaled_up(caplog) -> None:
    unowned = {1: "hkA", 2: "hkB"}
    w, parts = weight_parts({1: 0.75, 2: 0.25}, unowned, "owner", 0.59, None, {2: 0.05})
    assert w == pytest.approx({1: 0.75 * 0.95, 2: 0.25 * 0.95 + 0.05}) and parts["referral"] == {2: 0.05}
    assert weight_parts({}, unowned, "owner", 0.59, None, {2: 0.05})[0] == {}  # nothing to put the rest on
    assert "no weights set" in caplog.text
