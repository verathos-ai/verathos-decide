"""The usage pool: weight on top of the scores for the hotkeys that served paying users, sized each period from what the
users paid, recorded and signed by the auditor, and set alike by every validator."""

from __future__ import annotations

import dataclasses
import logging
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.test_throughput import FakeChain, kp
from verathos import catalog
from verathos.protocol import signing
from verathos.protocol.gateway import ServedCount
from verathos.throughput.auditor import Auditor, Slot
from verathos.throughput.feed import VERDICTS_DOMAIN, Verdicts
from verathos.throughput.policy import AuditPolicy
from verathos.throughput.usage import Emission, UsageRecord, close, contradiction, record_of, sized_share
from verathos.validator.__main__ import weights_to_set
from verathos.validator.scoring import ScoreState, weights
from verathos.validator.usage import take, usage_weights

OTHER = kp().ss58_address
OWNER_COLDKEY = kp().ss58_address
SN96 = Emission(alpha_out=10**9, owner_cut=11796, mechanism=0, mechanisms=1)  # 1 alpha per block, 18% owner cut
TESTNET = Emission(alpha_out=10**9, owner_cut=11796, mechanism=1, mechanisms=2, split=[58982, 6553])
MILLION = 1_000_000


def with_pool(pool: dict | None, miner_share=0.15, epoch_blocks: int = 180, referral: dict | None = None
              ) -> catalog.Catalog:
    document = catalog.load().document()
    scoring = {**document["scoring"], "miner_share": miner_share}
    if pool is not None:
        scoring["usage_pool"] = pool
    if referral is not None:
        scoring["referral_pool"] = referral
    return catalog.Catalog.parse({**document, "epoch_blocks": epoch_blocks, "scoring": scoring})


POOL = {"fraction": 0.6, "period_epochs": 2}


# ---- settings ----

def test_the_pool_is_off_unless_the_catalog_sets_it_and_its_settings_are_checked() -> None:
    assert catalog.load().scoring.usage_pool is None  # the shipped catalog
    assert catalog.load().scoring.miner_share_at(10**6) == 0.15  # a flat miner share
    pool = with_pool({}).scoring.usage_pool
    assert (pool.fraction, pool.period_epochs, pool.excluded) == (0.6, 40, [])  # the designed values: a day of epochs
    assert [with_pool(POOL).scoring.usage_pool.period(e) for e in (0, 1, 2, 5)] == [0, 0, 1, 2]
    for invalid, why in (({**POOL, "excluded": ["5Nope"]}, "ss58"), ({**POOL, "fraction": 0}, "fraction"),
                         ({**POOL, "fraction": 0.95}, "fraction"), ({**POOL, "period_epochs": 0}, "period_epochs")):
        with pytest.raises(ValidationError, match=why):
            with_pool(invalid)
    assert with_pool({**POOL, "mode": "burned"}).unknown_settings() == ["scoring.usage_pool.mode"]


def test_the_pool_runs_with_the_referral_pool_and_a_referred_usd_never_earns_a_usd() -> None:
    both = with_pool(POOL, referral={})  # usage 0.6 of the paid usage, referrals 0.3 per referred USD: 0.9 together
    assert both.scoring.usage_pool is not None and both.scoring.referral_pool is not None
    assert both.scoring.usage_cap_at(7) == 0.75  # what the miner share (0.15) and the referral share (0.10) leave
    assert with_pool(POOL).scoring.usage_cap_at(7) == 0.85
    assert with_pool(POOL, miner_share=1.0, referral={}).scoring.usage_cap_at(7) == 0.0
    with pytest.raises(ValidationError, match="add up to at most 0.9"):
        with_pool({**POOL, "fraction": 0.7}, referral={})  # 0.7 + 0.3: referring and serving oneself would pay
    assert with_pool({**POOL, "fraction": 0.8}, referral={"payout_per_usd": 0.1}).scoring.usage_pool.fraction == 0.8


# ---- sizing ----

def test_the_miners_emission_per_block_follows_the_chain() -> None:
    assert SN96.miner_alpha_per_block() == pytest.approx(0.41, abs=1e-4)  # 41% of 1 alpha per block
    assert TESTNET.miner_alpha_per_block() == pytest.approx(0.41 * 6553 / 65535, abs=1e-5)  # mechanism 1's tenth
    even = TESTNET.model_copy(update={"split": None})
    assert even.miner_alpha_per_block() == pytest.approx(0.41 / 2, abs=1e-4)
    assert SN96.model_copy(update={"owner_cut": 0}).miner_alpha_per_block() == 0.5
    assert SN96.model_copy(update={"mechanism": 1}).miner_alpha_per_block() == 0.0  # no such mechanism


def paid(tokens: dict[str, int], model: str = "kev-4b") -> dict[tuple[str, str, str], int]:
    return {(hotkey, "gpu0", model): n for hotkey, n in tokens.items()}


def test_a_period_is_sized_from_its_paid_usage_at_the_models_prices_and_the_chains_alpha_price() -> None:
    cat = with_pool(POOL)
    pool = cat.scoring.usage_pool
    tokens = {("hkA", "gpu0", "kev-4b"): 500 * MILLION, ("hkA", "gpu1", "kev-9b"): 20 * MILLION,
              ("hkB", "gpu1", "kev-9b"): 100 * MILLION}
    record = close(2, pool=pool, cat=cat, epoch_blocks=180, paid=tokens, coldkeys={"hkA": OTHER}, emission=SN96,
                   tao_usd=300.0, alpha_tao=0.004)
    assert (record.first_block, record.end_block) == (720, 1080)  # period 2 of 2 epochs of 180 blocks
    assert record.revenue_usd == pytest.approx(500 * 0.04 + 120 * 0.09)  # USD per million input tokens
    assert record.basis == pytest.approx(30.8 / 300.0 / 0.004)  # 25.67 alpha
    assert record.period_emission == pytest.approx(0.41 * 360, abs=0.05)
    assert record.share == round(0.6 * record.basis / record.period_emission, 6)
    assert record.paid_input_tokens == {"kev-4b": 500 * MILLION, "kev-9b": 120 * MILLION}
    assert [(s.hotkey, s.coldkey, s.input_tokens) for s in record.served] == [
        ("hkA", OTHER, {"kev-4b": 500 * MILLION, "kev-9b": 20 * MILLION}), ("hkB", None, {"kev-9b": 100 * MILLION})]
    assert record.inconsistency() is None
    assert close(2, pool=pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96, tao_usd=300.0,
                 alpha_tao=0.004).share == 0.0  # no paid usage: nothing
    huge = close(2, pool=pool, cat=cat, epoch_blocks=180, paid=paid({"hkA": 10**15}), coldkeys={}, emission=SN96,
                 tao_usd=300.0, alpha_tao=0.004)
    assert huge.share == 1.0  # never more than the mechanism's whole miner emission
    assert sized_share(0.6, 5.0, 0.0) == 0.0


def test_a_record_whose_figures_do_not_follow_from_its_inputs_is_inconsistent() -> None:
    cat = with_pool(POOL)
    record = close(2, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid=paid({"hkA": 10**9}),
                   coldkeys={}, emission=SN96, tao_usd=300.0, alpha_tao=0.004)
    assert record.inconsistency() is None
    twice = [*record.served, record.served[0]]
    for change, why in (({"share": record.share * 2}, "share"), ({"basis": 99.0}, "basis"),
                        ({"period_emission": 1.0}, "emission"), ({"paid_input_tokens": {"kev-4b": 11}}, "add up"),
                        ({"revenue_usd": 1000.0}, "revenue"), ({"prices": {"kev-4b": 0.5}}, "revenue"),
                        ({"served": twice, "paid_input_tokens": {"kev-4b": 2 * 10**9}}, "twice")):
        assert why in record.model_copy(update=change).inconsistency()
    with pytest.raises(ValidationError):
        UsageRecord.model_validate({**record.model_dump(mode="json"), "tao_usd": None})


# ---- weights ----

def record(share: float, served: dict[str, dict[str, int]], period: int = 2, coldkeys: dict | None = None,
           cat: catalog.Catalog | None = None) -> UsageRecord:
    """A record of ``period`` whose paid usage is ``served`` (per hotkey, input tokens per model), its revenue in alpha
    scaled to size ``share``, as consistent as the auditor's."""
    cat = cat or with_pool(POOL)
    tokens = {(hotkey, "gpu0", model): n for hotkey, models in served.items() for model, n in models.items()}
    probe = close(period, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid=tokens,
                  coldkeys=coldkeys or {}, emission=SN96, tao_usd=300.0, alpha_tao=1.0)
    alpha_tao = probe.basis * cat.scoring.usage_pool.fraction / (share * probe.period_emission) if share else 1.0
    built = close(period, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid=tokens,
                  coldkeys=coldkeys or {}, emission=SN96, tao_usd=300.0, alpha_tao=alpha_tao)
    assert built.share == pytest.approx(share if tokens else 0.0, abs=1e-6) and built.inconsistency() is None
    return built


HOTKEYS = {0: "owner", 1: "hkA", 2: "hkB", 3: "hkC"}


def test_usage_weights_split_the_share_by_what_users_paid_on_top_of_the_capacity_weights() -> None:
    cat = with_pool(POOL)
    # hkA answered 300 million paid kev-4b tokens (12 USD), hkB 100 million kev-9b tokens (9 USD)
    r = record(0.1, {"hkA": {"kev-4b": 300 * MILLION}, "hkB": {"kev-9b": 100 * MILLION}})
    usage = usage_weights(r, cat, 6, HOTKEYS)  # epoch 6 is in period 3: it pays period 2
    assert usage == pytest.approx({1: 0.1 * 12 / 21, 2: 0.1 * 9 / 21})  # by what was paid, not by tokens
    capacity = {2: 0.5, 3: 0.5}
    w = weights_to_set(capacity, HOTKEYS, "owner", 0.15, usage)
    assert w == pytest.approx({0: 1 - 0.15 - 0.1, 1: 0.1 * 12 / 21, 2: 0.15 * 0.5 + 0.1 * 9 / 21, 3: 0.15 * 0.5})
    assert sum(w.values()) == pytest.approx(1.0)
    assert usage_weights(r, cat, 7, HOTKEYS) == usage  # every epoch of the period pays it alike
    assert usage_weights(r, cat, 8, HOTKEYS) == {} and usage_weights(r, cat, 5, HOTKEYS) == {}  # other periods: none


def test_the_usage_share_is_at_most_what_the_miner_and_referral_shares_leave_and_the_rest_is_burned() -> None:
    big = record(0.9, {"hkA": {"kev-4b": MILLION}})
    assert usage_weights(big, with_pool(POOL), 6, HOTKEYS) == {1: pytest.approx(0.85)}  # 1 - 0.15
    both = with_pool(POOL, referral={})
    usage = usage_weights(big, both, 6, HOTKEYS)
    assert usage == {1: pytest.approx(0.75)}  # 1 - 0.15 - 0.10: the referrals' slice stays theirs
    w = weights_to_set({2: 1.0}, HOTKEYS, "owner", 0.15, usage, {3: 0.10})
    assert w == pytest.approx({1: 0.75, 2: 0.15, 3: 0.10}) and 0 not in w  # nothing left for the owner's UID
    small = weights_to_set({2: 1.0}, HOTKEYS, "owner", 0.15, usage_weights(record(0.2, {"hkA": {"kev-4b": 1}}),
                                                                            both, 6, HOTKEYS), {3: 0.02})
    assert small == pytest.approx({0: 0.63, 1: 0.2, 2: 0.15, 3: 0.02})  # what the slices do not pay is burned
    assert usage_weights(big, with_pool(POOL, miner_share=1.0), 6, HOTKEYS) == {}  # the miner share leaves nothing


def test_excluded_and_departed_hotkeys_get_no_usage_pay_and_their_part_stays_on_the_owners_uid() -> None:
    fleet = kp().ss58_address
    cat = with_pool({**POOL, "excluded": [fleet, OWNER_COLDKEY]})
    hotkeys = {0: "owner", 1: "hkA", 2: "hkB", 3: fleet}
    each = {"kev-4b": 100 * MILLION}
    r = record(0.2, {"hkA": each, "hkB": each, fleet: each, "hkGone": each}, coldkeys={"hkB": OWNER_COLDKEY},
               cat=cat)
    usage = usage_weights(r, cat, 6, hotkeys)
    assert usage == {1: pytest.approx(0.05)}  # hkB by its coldkey, the fleet hotkey by name, hkGone holds no UID
    assert weights_to_set({1: 1.0}, hotkeys, "owner", 0.15, usage) == pytest.approx({0: 0.80, 1: 0.20})


def test_without_the_pool_or_without_paid_usage_the_weights_are_the_scores_alone() -> None:
    capacity = {1: 0.75, 2: 0.25}
    alone = weights_to_set(capacity, HOTKEYS, "owner", 0.15)
    assert usage_weights(record(0.1, {"hkA": {"kev-4b": 1}}), with_pool(None), 6, HOTKEYS) == {}
    assert usage_weights(None, with_pool(POOL), 6, HOTKEYS) == {}
    assert usage_weights(record(0.0, {}), with_pool(POOL), 6, HOTKEYS) == {}  # nobody paid
    assert weights_to_set(capacity, HOTKEYS, "owner", 0.15, {}) == alone
    assert weights_to_set({}, HOTKEYS, "owner", 0.15, {}) == {0: 1.0}
    assert weights_to_set(capacity, HOTKEYS, "owner", 1.0, {}) == capacity
    unowned = {1: "hkA", 2: "hkB"}  # the owner hotkey holds no UID: the miners get it all, in proportion
    assert weights_to_set(capacity, unowned, "owner", 0.15, {1: 0.85}) == pytest.approx(
        {1: 0.15 * 0.75 + 0.85, 2: 0.15 * 0.25})


def test_every_validator_sets_the_same_weights_from_the_same_record_and_averages() -> None:
    """Two validators whose averages were folded in another order, taking the record from the same verdicts, set the
    same weights to the last digit."""
    cat = with_pool(POOL)
    r = record(0.123457, {"hkA": {"kev-4b": 7}, "hkB": {"kev-9b": 11}, "hkC": {"kev-4b": 13, "clef-flash": 3}})
    verdicts = Verdicts(netuid=405, auditor="a", epoch=6, credits=[], invalid=[], usage=r.model_dump(mode="json"))
    ema = {"hkA/gpu0": 0.3, "hkB/gpu0": 0.1, "hkC/gpu1": 0.2, "hkC/gpu0": 0.05}
    out = []
    for order in (sorted(ema), sorted(ema, reverse=True)):
        state = ScoreState(ema={k: ema[k] for k in order}, uid_of={k: {"hkA": 1, "hkB": 2, "hkC": 3}[k[:3]]
                                                                  for k in order})
        taken = take(state, verdicts, cat, 6)
        out.append(weights_to_set(weights(state, HOTKEYS), HOTKEYS, "owner", cat.scoring.miner_share_at(6),
                                  usage_weights(taken, cat, 6, HOTKEYS)))
    assert out[0] == out[1] and sum(out[0].values()) == pytest.approx(1.0)


def test_a_validator_takes_a_record_once_per_period_keeps_it_and_refuses_one_that_does_not_hold(caplog) -> None:
    cat = with_pool(POOL)
    r = record(0.1, {"hkA": {"kev-4b": 1}})
    stated = Verdicts(netuid=405, auditor="a", epoch=6, credits=[], invalid=[], usage=r.model_dump(mode="json"))
    state, checked = ScoreState(), []

    def check(rec):
        checked.append(rec.period)
        return None
    assert take(state, stated, cat, 6, check) == r and checked == [2]
    assert take(state, None, cat, 7, check) == r and checked == [2]  # no verdicts: the record kept pays the epoch
    assert take(state, stated, cat, 7, check) == r and checked == [2]  # taken once
    assert take(state, None, cat, 8) is None  # the next period needs the next record
    assert take(ScoreState(), stated, cat, 8) is None  # a record of another period pays nothing
    assert take(state, stated, with_pool(None), 6) is None

    contradicted = ScoreState()
    with caplog.at_level(logging.ERROR, logger="verathos.validator"):
        assert take(contradicted, stated, cat, 6, lambda rec: "another alpha price") is None
    assert "refused" in caplog.text and contradicted.usage_refused == "another alpha price"
    assert take(contradicted, stated, cat, 7) is None  # refused for the whole period
    tampered = Verdicts(netuid=405, auditor="a", epoch=6, credits=[], invalid=[],
                        usage={**r.model_dump(mode="json"), "share": 0.3})
    assert take(ScoreState(), tampered, cat, 6) is None

    def unreachable(rec):
        raise ConnectionError("node down")
    assert take(ScoreState(), stated, cat, 6, unreachable) == r  # unchecked: the auditor's word
    assert record_of({"period": 1}) is None and record_of("x") is None


# ---- the chain ----

class Substrate:
    """Storage as the SDK decodes it; blocks in ``pruned`` are beyond the node's state."""

    def __init__(self, events: dict[int, list[dict]], storage: dict[str, object], pruned: set[int] = frozenset()):
        self.events, self.storage, self.pruned = events, storage, pruned

    def get_block_hash(self, block: int) -> str:
        return f"0x{block:064x}"

    def _block(self, block_hash: str) -> int:
        block = int(block_hash, 16)
        if block in self.pruned:
            raise RuntimeError(f"State already discarded for {block_hash}")
        return block

    def get_events(self, block_hash: str) -> list[dict]:
        return self.events.get(self._block(block_hash), [])

    def query(self, module: str, name: str, params: list, block_hash: str):
        self._block(block_hash)
        value = self.storage[name]
        return SimpleNamespace(value=value(params) if callable(value) else value)

    def get_metadata_storage_function(self, module: str, name: str, block_hash: str):
        return name in self.storage


STORAGE = {"SubnetAlphaOutEmission": 10**9, "SubnetOwnerCut": 11796, "OwnerCutEnabled": True,
           "MechanismCountCurrent": 1, "MechanismEmissionSplit": None, "SubnetMovingPrice": {"bits": 17179869},
           "Owner": lambda params: {"hkA": OTHER}.get(params[0])}


def test_a_validator_checks_the_reference_block_its_node_still_holds() -> None:
    from verathos.throughput.usage import alpha_price_at, coldkeys_at, emission_at

    st = SimpleNamespace(substrate=Substrate({}, STORAGE))
    assert emission_at(st, 1080, 405, 0) == SN96
    assert alpha_price_at(st, 1080, 405) == pytest.approx(0.004, rel=1e-6)
    assert coldkeys_at(st, ["hkA", "hkB"], 1080) == {"hkA": OTHER, "hkB": None}
    cat = with_pool(POOL)
    good = close(2, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96,
                 tao_usd=300.0, alpha_tao=alpha_price_at(st, 1080, 405))
    assert contradiction(good, st, 405) is None
    assert "emission" in contradiction(good.model_copy(update={"emission": TESTNET}), st, 405)
    assert "price" in contradiction(good.model_copy(update={"alpha_tao": 0.008}), st, 405)
    pruned = SimpleNamespace(substrate=Substrate({}, STORAGE, pruned={1080}))
    assert contradiction(good.model_copy(update={"alpha_tao": 0.008}), pruned, 405) is None  # nothing it holds


class Behind(Substrate):
    """A node that does not hold the blocks after ``head`` yet: no hash for them, and a read without a hash is of its
    head, where the price has moved since."""

    def __init__(self, head: int) -> None:
        super().__init__({}, {**STORAGE, "SubnetMovingPrice": {"bits": 34359738}})
        self.head = head

    def get_block_hash(self, block: int) -> str | None:
        return super().get_block_hash(block) if block <= self.head else None

    def query(self, module: str, name: str, params: list, block_hash: str | None):
        return super().query(module, name, params, block_hash or super().get_block_hash(self.head))

    def get_metadata_storage_function(self, module: str, name: str, block_hash: str | None):
        return name in self.storage


def test_a_node_without_the_reference_block_checks_nothing_rather_than_its_head() -> None:
    """A record's reference block that the node does not hold yet (a lagging node behind a load balancer) is not read
    at the node's head: that would take the head's price for the block's and refuse a sound record for a whole
    period. The reads raise, and what cannot be read is not checked."""
    from verathos.throughput.usage import alpha_price_at, block_hash

    st = SimpleNamespace(substrate=Behind(head=1079))
    with pytest.raises(LookupError, match="holds no block 1080"):
        block_hash(st, 1080)
    with pytest.raises(LookupError):
        alpha_price_at(st, 1080, 405)
    cat = with_pool(POOL)
    record = close(2, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96,
                   tao_usd=300.0, alpha_tao=0.004)
    assert record.end_block == 1080 and contradiction(record, st, 405) is None


# ---- the auditor ----

class UsageChain(FakeChain):
    """The chain as the auditor's usage pool reads it."""

    def emission(self, block: int, netuid: int, mechanism: int) -> Emission:
        return SN96.model_copy(update={"mechanism": mechanism})

    def alpha_price(self, block: int, netuid: int) -> float:
        return 0.004

    def coldkeys(self, hotkeys, block: int) -> dict[str, str | None]:
        return {hk: {"hkA": OTHER}.get(hk) for hk in hotkeys}


def usage_auditor(tmp_path, cat: catalog.Catalog, chain: UsageChain, tao_usd=lambda: 300.0) -> Auditor:
    auditor = Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                      catalog=lambda: cat, references={}, db_path=tmp_path / "t.sqlite", tao_usd=tao_usd)
    auditor.update_slots([Slot("hkA", "gpu0", "https://a/e/gpu0", "kev-4b", 50.0, "ab" * 32),
                          Slot("hkB", "gpu0", "https://b/e/gpu0", "kev-9b", 50.0, "cd" * 32)])
    return auditor


def at(auditor: Auditor, block: int) -> None:
    """The chain at ``block``, final; the auditor's loops step once."""
    auditor.chain.head_block = auditor.chain.finalized = block
    auditor.follower.poll()
    auditor.settle()
    auditor.step_usage()


def serve(auditor: Auditor, hotkey: str, paid_tokens: int, unpaid: int = 0) -> None:
    auditor.accept_served([ServedCount(hotkey=hotkey, endpoint_id="gpu0", second=int(time.time()), answered=1000,
                                       input_tokens=paid_tokens + unpaid, paid_input_tokens=paid_tokens)])


def test_the_auditor_records_a_periods_paid_usage_at_the_prices_of_its_end_and_signs_it(tmp_path) -> None:
    cat = with_pool(POOL)
    chain = UsageChain(4 * 180 + 1)
    auditor = usage_auditor(tmp_path, cat, chain)
    at(auditor, 4 * 180 + 1)  # period 2 (epochs 4 and 5) begins
    serve(auditor, "hkA", 2 * MILLION, unpaid=500)  # kev-4b at 0.04 USD per million: 0.08 USD
    serve(auditor, "hkB", MILLION, unpaid=5 * MILLION)  # kev-9b at 0.09: 0.09 USD
    for block in range(4 * 180 + 101, 6 * 180, 100):  # blocks 821 (epoch 4), 921 and 1021 (epoch 5)
        at(auditor, block)
        if block == 921:
            serve(auditor, "hkA", MILLION)  # 0.04 USD more
    before = record_of(auditor.verdicts(5).verdicts.model_extra["usage"])  # the period before, closed on arrival
    assert (before.period, before.served, before.share) == (1, [], 0.0)  # nobody paid in it, as far as known
    at(auditor, 6 * 180 + 3)  # period 3 begins: period 2 closes once its end block is final
    signed = auditor.verdicts(6)
    v = signed.verdicts
    assert signing.verify(auditor.hotkey, VERDICTS_DOMAIN, v.model_dump(mode="json"), signed.signature)
    r = record_of(v.model_extra["usage"])
    assert (r.period, r.first_block, r.end_block) == (2, 720, 1080)
    assert [(s.hotkey, s.coldkey, s.input_tokens) for s in r.served] == [
        ("hkA", OTHER, {"kev-4b": 3 * MILLION}), ("hkB", None, {"kev-9b": MILLION})]
    assert r.revenue_usd == pytest.approx(0.21) and (r.tao_usd, r.alpha_tao) == (300.0, 0.004)
    assert r.basis == pytest.approx(0.21 / 300.0 / 0.004) and r.inconsistency() is None
    assert r.share == round(0.6 * r.basis / r.period_emission, 6)
    at(auditor, 7 * 180 + 1)
    assert record_of(auditor.verdicts(7).verdicts.model_extra["usage"]) == r  # every epoch of period 3 states it
    usage = usage_weights(take(ScoreState(), v, cat, 6), cat, 6, HOTKEYS)
    assert usage == pytest.approx({1: r.share * 0.12 / 0.21, 2: r.share * 0.09 / 0.21})


def test_the_first_verdicts_of_a_period_wait_briefly_for_its_record_then_come_without_it(tmp_path) -> None:
    cat = with_pool(POOL)
    prices = []

    def tao_usd() -> float:
        if not prices:
            raise ValueError("no TAO price: every ticker down")
        return prices[0]
    auditor = usage_auditor(tmp_path, cat, UsageChain(4 * 180 + 1), tao_usd)
    at(auditor, 4 * 180 + 1)
    serve(auditor, "hkA", MILLION)
    at(auditor, 5 * 180 + 1)
    at(auditor, 6 * 180 + 2)  # period 2 ends, but no TAO price: its record cannot be made yet, the verdicts wait ...
    assert auditor.verdicts(6) is None
    at(auditor, 6 * 180 + 30)  # ... briefly, then come without it
    assert auditor.verdicts(6).verdicts.model_extra.get("usage") is None
    prices.append(300.0)
    at(auditor, 6 * 180 + 40)  # the price is back: the record is made, and the next verdicts state it
    at(auditor, 7 * 180 + 1)
    assert record_of(auditor.verdicts(7).verdicts.model_extra["usage"]).revenue_usd == pytest.approx(0.04)


def test_a_new_period_length_starts_periods_of_other_blocks_and_nothing_is_paid_twice(tmp_path) -> None:
    """The auditor gives no record to a period overlapping one recorded before; a validator pays a record only for the
    blocks of the period its catalog makes."""
    catalogs = [with_pool(POOL)]  # periods of 2 epochs: 360 blocks
    chain = UsageChain(4 * 180 + 1)
    auditor = Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                      catalog=lambda: catalogs[-1], references={}, db_path=tmp_path / "t.sqlite",
                      tao_usd=lambda: 300.0)
    auditor.update_slots([Slot("hkA", "gpu0", "https://a/e/gpu0", "kev-4b", 50.0, "ab" * 32)])
    for block in range(4 * 180 + 1, 6 * 180 + 2, 100):
        at(auditor, block)
        if block == 4 * 180 + 1:
            serve(auditor, "hkA", MILLION)
    at(auditor, 6 * 180 + 2)
    first = record_of(auditor.verdicts(6).verdicts.model_extra["usage"])
    assert (first.first_block, first.end_block, first.paid_input_tokens) == (720, 1080, {"kev-4b": MILLION})
    catalogs.append(with_pool({**POOL, "period_epochs": 4}))  # periods of 720 blocks: [720, 1440) is paid already
    for block in range(6 * 180 + 102, 8 * 180 + 2, 100):
        at(auditor, block)
    at(auditor, 8 * 180 + 2)
    assert auditor.verdicts(8) is None  # waiting briefly for a record that does not come
    at(auditor, 8 * 180 + 25)
    assert auditor.verdicts(8).verdicts.model_extra.get("usage") is None
    for block in range(8 * 180 + 102, 12 * 180 + 2, 100):
        at(auditor, block)
        if block == 8 * 180 + 102:
            serve(auditor, "hkA", 2 * MILLION)
    at(auditor, 12 * 180 + 2)
    later = record_of(auditor.verdicts(12).verdicts.model_extra["usage"])
    assert (later.first_block, later.end_block, later.paid_input_tokens) == (1440, 2160, {"kev-4b": 2 * MILLION})

    stated = Verdicts(netuid=405, auditor="a", epoch=6, credits=[], invalid=[], usage=first.model_dump(mode="json"))
    longer = with_pool({**POOL, "period_epochs": 3})  # epoch 6 is the first of period 2, of blocks [1080, 1620)
    assert take(ScoreState(), stated, longer, 6) is None and usage_weights(first, longer, 6, HOTKEYS) == {}


def test_the_tao_price_falls_back_to_the_next_ticker_and_takes_only_a_price_within_the_bounds() -> None:
    from verathos.throughput import usage as usage_module

    coingecko, kraken, coinbase = (url for _, url, _ in usage_module.TAO_USD_SOURCES)
    answers = {coingecko: ConnectionError("blocked"),
               kraken: {"error": [], "result": {"TAOUSD": {"c": ["123456.0", "1.0"]}}},  # out of bounds
               coinbase: {"data": {"base": "TAO", "currency": "USD", "amount": "301.25"}}}

    def fetch(url, **_):
        answer = answers[url]
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert usage_module.tao_usd(fetch) == 301.25
    answers[kraken] = {"error": [], "result": {"TAOUSD": {"c": ["299.5", "1.0"]}}}
    assert usage_module.tao_usd(fetch) == 299.5  # the first good one
    answers[coingecko] = {"bittensor": {"usd": 300.0}}
    assert usage_module.tao_usd(fetch) == 300.0
    for url in answers:
        answers[url] = ValueError("down")
    with pytest.raises(ValueError, match="no TAO price: CoinGecko: ValueError: down; Kraken"):
        usage_module.tao_usd(fetch)
