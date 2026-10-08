"""The usage pool: weight on top of the scores for the hotkeys that served paying users, sized each period from the
buyback's burns or the users' payments, recorded and signed by the auditor."""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.test_throughput import FakeChain, kp
from verathos import catalog
from verathos.protocol import signing
from verathos.protocol.gateway import ServedCount
from verathos.throughput.auditor import SCAN_REACH_BLOCKS, Auditor, Slot
from verathos.throughput.feed import VERDICTS_DOMAIN
from verathos.throughput.policy import AuditPolicy
from verathos.throughput.usage import (
    Burn,
    Emission,
    burns_in,
    close,
    contradiction,
    record_of,
    sized_share,
)

BUYBACK = kp().ss58_address
OTHER = kp().ss58_address
OWNER_COLDKEY = kp().ss58_address
SN96 = Emission(alpha_out=10**9, owner_cut=11796, mechanism=0, mechanisms=1)  # 1 alpha per block, 18% owner cut
TESTNET = Emission(alpha_out=10**9, owner_cut=11796, mechanism=1, mechanisms=2, split=[58982, 6553])


def with_pool(pool: dict | None, miner_share=0.59, epoch_blocks: int = 180) -> catalog.Catalog:
    document = catalog.load().document()
    scoring = {**document["scoring"], "miner_share": miner_share}
    if pool is not None:
        scoring["usage_pool"] = pool
    return catalog.Catalog.parse({**document, "epoch_blocks": epoch_blocks, "scoring": scoring})


BURNED = {"mode": "burned", "fraction": 1.0, "period_epochs": 2, "buyback_coldkey": BUYBACK}
REVENUE = {"mode": "revenue", "fraction": 0.9, "period_epochs": 2}


# ---- settings ----

def test_the_pool_is_off_unless_the_catalog_sets_it_and_its_settings_are_checked() -> None:
    assert catalog.load().scoring.usage_pool is None  # the shipped catalog: today's weights
    pool = with_pool(BURNED).scoring.usage_pool
    assert (pool.mode, pool.fraction, pool.period_epochs, pool.excluded) == ("burned", 1.0, 2, [])
    assert [pool.period(e) for e in (0, 1, 2, 5)] == [0, 0, 1, 2]
    assert with_pool({"mode": "revenue", "fraction": 0.9}).scoring.usage_pool.period_epochs == 40  # a day of epochs
    for invalid, why in (({"mode": "burned", "fraction": 1.0}, "buyback_coldkey"),
                         ({**BURNED, "buyback_coldkey": "not-an-address"}, "ss58"),
                         ({**REVENUE, "excluded": ["5Nope"]}, "ss58"),
                         ({**REVENUE, "fraction": 0}, "fraction"), ({**REVENUE, "fraction": 1.1}, "fraction"),
                         ({**REVENUE, "period_epochs": 0}, "period_epochs"), ({"fraction": 0.9}, "mode"),
                         ({**REVENUE, "mode": "minted"}, "mode")):
        with pytest.raises(ValidationError, match=why):
            with_pool(invalid)
    assert with_pool({**REVENUE, "period": 3}).unknown_settings() == ["scoring.usage_pool.period"]


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


def test_burned_mode_pays_the_alpha_the_buyback_burned_in_the_period_once_at_most_the_whole_emission() -> None:
    cat = with_pool(BURNED)
    pool = cat.scoring.usage_pool
    first, end = 2 * 360, 3 * 360  # period 2 of 2 epochs of 180 blocks
    burns = [Burn(block=first + 10, event_index=3, coldkey=BUYBACK, hotkey="hkX", amount=20 * 10**9),
             Burn(block=end - 1, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=4 * 10**9),
             Burn(block=first + 11, event_index=1, coldkey=OTHER, hotkey="hkX", amount=99 * 10**9),  # not the buyback's
             Burn(block=end, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=7 * 10**9)]  # the next period's
    record = close(2, pool=pool, cat=cat, epoch_blocks=180, paid=paid({"hkA": 300, "hkB": 100}),
                   coldkeys={"hkA": OTHER}, emission=SN96, burns=burns, unscanned=[(first + 100, first + 120)])
    assert record.basis == 24.0 and [b.amount for b in record.burns] == [20 * 10**9, 4 * 10**9]
    assert record.period_emission == pytest.approx(0.41 * 360, abs=0.05)
    assert record.share == round(24.0 / record.period_emission, 6)  # the slice, per epoch of the next period
    assert record.unscanned == [(first + 100, first + 120)] and record.inconsistency() is None
    assert [(s.hotkey, s.coldkey, s.input_tokens) for s in record.served] == [("hkA", OTHER, 300), ("hkB", None, 100)]
    huge = close(2, pool=pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96,
                 burns=[Burn(block=first, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=10**6 * 10**9)])
    assert huge.share == 1.0  # never more than the mechanism's whole miner emission
    assert sized_share(1.0, 5.0, 0.0) == 0.0


def test_revenue_mode_converts_the_periods_paid_usage_at_the_models_prices_and_the_chains_alpha_price() -> None:
    cat = with_pool(REVENUE)
    pool = cat.scoring.usage_pool
    tokens = {("hkA", "gpu0", "kev-4b"): 500_000_000, ("hkB", "gpu1", "kev-9b"): 100_000_000}
    record = close(2, pool=pool, cat=cat, epoch_blocks=180, paid=tokens, coldkeys={}, emission=SN96,
                   tao_usd=300.0, alpha_tao=0.004)
    assert record.revenue_usd == pytest.approx(500 * 0.04 + 100 * 0.09)  # USD per million input tokens
    assert record.basis == pytest.approx(29.0 / 300.0 / 0.004)  # 24.17 alpha
    assert record.share == round(0.9 * record.basis / record.period_emission, 6)
    assert record.paid_input_tokens == {"kev-4b": 500_000_000, "kev-9b": 100_000_000}
    assert record.inconsistency() is None
    assert close(2, pool=pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96, tao_usd=300.0,
                 alpha_tao=0.004).share == 0.0  # no paid usage: nothing


def test_a_record_whose_figures_do_not_follow_from_its_inputs_is_inconsistent() -> None:
    cat = with_pool(BURNED)
    burn = Burn(block=800, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=10 * 10**9)
    record = close(2, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid=paid({"hkA": 10}), coldkeys={},
                   emission=SN96, burns=[burn])
    assert record.inconsistency() is None
    for change, why in (({"share": record.share * 2}, "share"), ({"basis": 99.0}, "basis"),
                        ({"burns": [burn, burn]}, "twice"),
                        ({"burns": [burn.model_copy(update={"coldkey": OTHER})]}, "coldkey"),
                        ({"burns": [burn.model_copy(update={"block": 10})]}, "outside"),
                        ({"period_emission": 1.0}, "emission"),
                        ({"paid_input_tokens": {"kev-4b": 11}}, "add up")):
        assert why in record.model_copy(update=change).inconsistency()
    revenue = close(2, pool=with_pool(REVENUE).scoring.usage_pool, cat=cat, epoch_blocks=180,
                    paid=paid({"hkA": 10**9}), coldkeys={}, emission=SN96, tao_usd=300.0, alpha_tao=0.004)
    assert "revenue" in revenue.model_copy(update={"revenue_usd": 1000.0}).inconsistency()
    assert "lacks" in revenue.model_copy(update={"tao_usd": None}).inconsistency()


# ---- the chain ----

class Substrate:
    """Blocks' events and storage as the SDK decodes them; blocks in ``pruned`` are beyond the node's state."""

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


def alpha_burned(coldkey: str, amount: int, netuid: int = 405) -> dict:
    return {"phase": {"ApplyExtrinsic": 2}, "extrinsic_idx": 2, "module_id": "SubtensorModule",
            "event_id": "AlphaBurned", "attributes": (coldkey, "hkX", amount, netuid), "topics": []}


STORAGE = {"SubnetAlphaOutEmission": 10**9, "SubnetOwnerCut": 11796, "OwnerCutEnabled": True,
           "MechanismCountCurrent": 1, "MechanismEmissionSplit": None, "SubnetMovingPrice": {"bits": 17179869},
           "Owner": lambda params: {"hkA": OTHER}.get(params[0])}


def test_burns_are_read_from_the_blocks_events() -> None:
    weights_set = {"module_id": "SubtensorModule", "event_id": "WeightsSet", "attributes": (405, 3)}
    events = [weights_set, alpha_burned(BUYBACK, 5), alpha_burned(OTHER, 7, netuid=96),
              {**alpha_burned(BUYBACK, 9), "attributes": {"coldkey": BUYBACK, "hotkey": "hkX", "amount": 9,
                                                         "netuid": 405}}]
    assert burns_in(events, 800, 405) == [Burn(block=800, event_index=1, coldkey=BUYBACK, hotkey="hkX", amount=5),
                                          Burn(block=800, event_index=3, coldkey=BUYBACK, hotkey="hkX", amount=9)]


def test_a_validator_checks_the_burns_and_the_reference_block_its_node_still_holds() -> None:
    from verathos.throughput.usage import alpha_price_at, coldkeys_at, emission_at

    st = SimpleNamespace(substrate=Substrate({1075: [alpha_burned(BUYBACK, 10**9)]}, STORAGE, pruned={900}))
    assert emission_at(st, 1080, 405, 0) == SN96
    assert alpha_price_at(st, 1080, 405) == pytest.approx(0.004, rel=1e-6)
    assert coldkeys_at(st, ["hkA", "hkB"], 1080) == {"hkA": OTHER, "hkB": None}
    cat = with_pool(BURNED)
    good = close(2, pool=cat.scoring.usage_pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={}, emission=SN96,
                 burns=[Burn(block=1075, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=10**9),
                        Burn(block=900, event_index=4, coldkey=BUYBACK, hotkey="hkX", amount=10**9)])
    assert contradiction(good, st, 405) is None  # block 900 is beyond the node: the auditor's word
    invented = good.model_copy(update={"burns": [*good.burns, Burn(block=1076, event_index=0, coldkey=BUYBACK,
                                                                    hotkey="hkX", amount=10**9)]})
    assert "no burn" in contradiction(invented, st, 405)
    assert "emission" in contradiction(good.model_copy(update={"emission": TESTNET}), st, 405)
    revenue = close(2, pool=with_pool(REVENUE).scoring.usage_pool, cat=cat, epoch_blocks=180, paid={}, coldkeys={},
                    emission=SN96, tao_usd=300.0, alpha_tao=0.004)
    assert contradiction(revenue, st, 405) is None
    assert "price" in contradiction(revenue.model_copy(update={"alpha_tao": 0.008}), st, 405)
    pruned = SimpleNamespace(substrate=Substrate({}, STORAGE, pruned={1080, 1076, 1075, 900}))
    assert contradiction(invented, pruned, 405) is None  # nothing it holds says otherwise


# ---- the auditor ----

class UsageChain(FakeChain):
    """The chain as the auditor's usage pool reads it."""

    def __init__(self, head: int, burns: dict[int, list[Burn]] | None = None) -> None:
        super().__init__(head)
        self.burned, self.read, self.unreadable = burns or {}, [], set()

    def burns(self, block: int, netuid: int) -> list[Burn]:
        if block in self.unreadable:
            raise RuntimeError("State already discarded")
        self.read.append(block)
        return list(self.burned.get(block, []))

    def emission(self, block: int, netuid: int, mechanism: int) -> Emission:
        return SN96.model_copy(update={"mechanism": mechanism})

    def alpha_price(self, block: int, netuid: int) -> float:
        return 0.004

    def coldkeys(self, hotkeys, block: int) -> dict[str, str | None]:
        return {hk: {"hkA": OTHER}.get(hk) for hk in hotkeys}


def usage_auditor(tmp_path, cat: catalog.Catalog, chain: UsageChain) -> Auditor:
    auditor = Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                      catalog=lambda: cat, references={}, db_path=tmp_path / "t.sqlite", tao_usd=lambda: 300.0)
    auditor.update_slots([Slot("hkA", "gpu0", "http://a/e/gpu0", "kev-4b", 50.0),
                          Slot("hkB", "gpu0", "http://b/e/gpu0", "kev-9b", 50.0)])
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


def test_the_auditor_records_the_buybacks_burns_and_the_paid_usage_of_a_period_and_signs_them(tmp_path) -> None:
    burns = {900: [Burn(block=900, event_index=2, coldkey=BUYBACK, hotkey="hkX", amount=30 * 10**9),
                   Burn(block=900, event_index=5, coldkey=OTHER, hotkey="hkX", amount=10**9)],
             1079: [Burn(block=1079, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=6 * 10**9)],
             1080: [Burn(block=1080, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=10**9)]}  # period 3's
    cat = with_pool(BURNED)
    chain = UsageChain(4 * 180 + 1, burns)
    auditor = usage_auditor(tmp_path, cat, chain)
    at(auditor, 4 * 180 + 1)  # period 2 (epochs 4 and 5) begins: the burns are read from its first block
    assert chain.read == [720, 721]
    serve(auditor, "hkA", 3000, unpaid=500)
    serve(auditor, "hkB", 1000)
    for block in range(4 * 180 + 101, 6 * 180, 100):  # blocks 821 (epoch 4), 921 and 1021 (epoch 5)
        at(auditor, block)
        if block == 921:
            serve(auditor, "hkA", 1000)
    assert auditor.verdicts(5).verdicts.model_extra.get("usage") is None  # the pool has no period before
    at(auditor, 6 * 180 + 3)  # period 3 begins: period 2 closes once its end block is final
    signed = auditor.verdicts(6)
    v = signed.verdicts
    assert signing.verify(auditor.hotkey, VERDICTS_DOMAIN, v.model_dump(mode="json"), signed.signature)
    r = record_of(v.model_extra["usage"])
    assert (r.period, r.first_block, r.end_block, r.mode) == (2, 720, 1080, "burned")
    assert [(b.block, b.amount) for b in r.burns] == [(900, 30 * 10**9), (1079, 6 * 10**9)] and r.basis == 36.0
    assert [(s.hotkey, s.coldkey, s.input_tokens) for s in r.served] == [("hkA", OTHER, 4000), ("hkB", None, 1000)]
    assert r.unscanned == [] and r.inconsistency() is None
    assert r.share == round(36.0 / (SN96.miner_alpha_per_block() * 360), 6)
    at(auditor, 7 * 180 + 1)
    assert record_of(auditor.verdicts(7).verdicts.model_extra["usage"]) == r  # every epoch of period 3 states it


def test_the_first_verdicts_of_a_period_wait_briefly_for_the_record_and_blocks_out_of_reach_do_not_count(
        tmp_path) -> None:
    cat = with_pool(BURNED)
    burns = {730: [Burn(block=730, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=8 * 10**9)],
             1000: [Burn(block=1000, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=2 * 10**9)]}
    chain = UsageChain(4 * 180 + 1, burns)
    auditor = usage_auditor(tmp_path, cat, chain)
    at(auditor, 4 * 180 + 1)
    chain.unreadable = set(range(722, 1080))
    at(auditor, 4 * 180 + 50)  # a block it cannot read yet: tried again
    chain.unreadable = set()
    at(auditor, 6 * 180 + 1)  # the auditor was away: the blocks beyond the node's reach are not read
    r = record_of(auditor.verdicts(6).verdicts.model_extra["usage"])
    assert r.unscanned == [(722, 6 * 180 + 1 - SCAN_REACH_BLOCKS)] and [b.block for b in r.burns] == [1000]
    assert r.basis == 2.0

    late = UsageChain(6 * 180 + 1)
    (tmp_path / "late").mkdir()
    other = usage_auditor(tmp_path / "late", cat, late)
    other.chain.head_block = 6 * 180 + 2
    other.follower.poll()  # the record of period 2 cannot be made (its burns were never read): the verdicts wait ...
    assert other.verdicts(6) is None
    other.chain.head_block = 6 * 180 + 30
    other.follower.poll()  # ... briefly, then come without it
    assert other.verdicts(6).verdicts.model_extra.get("usage") is None


def test_in_revenue_mode_the_auditor_records_the_paid_usage_at_the_prices_of_the_periods_end(tmp_path) -> None:
    cat = with_pool(REVENUE)
    chain = UsageChain(4 * 180 + 1)
    auditor = usage_auditor(tmp_path, cat, chain)
    at(auditor, 4 * 180 + 1)
    serve(auditor, "hkA", 2_000_000)  # kev-4b at 0.04 USD per million: 0.08 USD
    serve(auditor, "hkB", 1_000_000, unpaid=5_000_000)  # kev-9b at 0.09: 0.09 USD
    at(auditor, 5 * 180 + 1)
    at(auditor, 6 * 180 + 1)
    r = record_of(auditor.verdicts(6).verdicts.model_extra["usage"])
    assert chain.read == []  # revenue mode reads no burns
    assert r.revenue_usd == pytest.approx(0.17) and (r.tao_usd, r.alpha_tao) == (300.0, 0.004)
    assert r.basis == pytest.approx(0.17 / 300.0 / 0.004) and r.inconsistency() is None
    assert r.share == round(0.9 * r.basis / r.period_emission, 6)


def test_a_new_period_length_starts_periods_of_other_blocks_and_nothing_is_paid_twice(tmp_path) -> None:
    """The auditor gives no record to a period overlapping one recorded before."""
    burns = {b: [Burn(block=b, event_index=0, coldkey=BUYBACK, hotkey="hkX", amount=10**9)] for b in (900, 1200, 1500)}
    catalogs = [with_pool(BURNED)]  # periods of 2 epochs: 360 blocks
    chain = UsageChain(4 * 180 + 1, burns)
    auditor = Auditor(netuid=405, keypair=kp(), chain=chain, policy=dataclasses.replace(AuditPolicy()),
                      catalog=lambda: catalogs[-1], references={}, db_path=tmp_path / "t.sqlite")
    for block in range(4 * 180 + 1, 6 * 180 + 2, 100):
        at(auditor, block)
    at(auditor, 6 * 180 + 2)
    first = record_of(auditor.verdicts(6).verdicts.model_extra["usage"])
    assert (first.first_block, first.end_block, first.basis) == (720, 1080, 1.0)
    catalogs.append(with_pool({**BURNED, "period_epochs": 4}))  # periods of 720 blocks: [720, 1440) is paid already
    for block in range(6 * 180 + 102, 8 * 180 + 2, 100):
        at(auditor, block)
    at(auditor, 8 * 180 + 2)
    assert auditor.verdicts(8) is None  # waiting briefly for a record that does not come
    at(auditor, 8 * 180 + 25)
    assert auditor.verdicts(8).verdicts.model_extra.get("usage") is None
    for block in range(8 * 180 + 102, 12 * 180 + 2, 100):
        at(auditor, block)
    at(auditor, 12 * 180 + 2)
    later = record_of(auditor.verdicts(12).verdicts.model_extra["usage"])
    assert (later.first_block, later.end_block, later.basis) == (1440, 2160, 1.0)  # the burn at 1500 only
