"""Referrals at the gateway: the catalog's referral pool, codes bound to hotkeys by their signature, attribution fixed
by the first top-up, refunds, what counts within the term and the account cap, the referrer's figures, and the
reports to the auditor."""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import re
import time

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import select, update

from tests.mock_engine import MockEngine
from tests.test_gateway import ACCOUNTS_TOKEN, ADMIN_TOKEN, Setup, admin_call
from tests.test_throughput import kp
from verathos import catalog
from verathos.gateway import db
from verathos.gateway.accounts import Accounts
from verathos.gateway.referrals import MAX_CODES, Referrals, reserved
from verathos.protocol.gateway import ReferralReport
from verathos.protocol.referral import (
    CodeMove,
    CodeRegistration,
    ReferralView,
    SignedMove,
    SignedRegistration,
    fold,
    random_code,
    read_view_token,
    view_token,
    well_formed,
)

pytestmark = pytest.mark.usefixtures("lab_network")
SITE = "https://referrals.example"
USD = 1_000_000


def with_referrals(pool: dict | None = None, usage: dict | None = None) -> catalog.Catalog:
    document = catalog.load().document()
    scoring = {**document["scoring"], "miner_share": 0.59}
    if pool is not None:
        scoring["referral_pool"] = pool
    if usage is not None:
        scoring["usage_pool"] = usage
    return catalog.Catalog.parse({**document, "scoring": scoring})


def now_ms() -> int:
    return int(time.time() * 1000)


def registration(keypair, code: str, **fields) -> dict:
    message = {"netuid": 405, "hotkey": keypair.ss58_address, "code": code, "site": SITE, "issued_ms": now_ms(),
               **fields}
    return SignedRegistration.create(CodeRegistration(**message), keypair).model_dump(mode="json")


def move(signer, from_hotkey: str, to_hotkey: str, **fields) -> dict:
    message = {"netuid": 405, "from_hotkey": from_hotkey, "to_hotkey": to_hotkey, "site": SITE,
               "issued_ms": now_ms(), **fields}
    return SignedMove.create(CodeMove(**message), signer).model_dump(mode="json")


# ---- the catalog ----

def test_the_referral_pool_is_off_in_the_shipped_catalog_and_its_settings_are_checked() -> None:
    assert catalog.load().scoring.referral_pool is None
    pool = with_referrals({}).scoring.referral_pool
    assert (pool.share, pool.payout_per_usd, pool.window_periods, pool.period_epochs, pool.term_days,
            pool.account_cap_usd, pool.serving_only, pool.excluded) == (0.10, 0.3, 30, 40, 365, 10_000.0, False, [])
    assert [pool.period(e) for e in (0, 39, 40, 85)] == [0, 0, 1, 2]
    assert pool.window_s(180, 12.0) == 30 * 40 * 180 * 12.0  # about 30 days
    lifetime = with_referrals({"term_days": None, "account_cap_usd": None}).scoring.referral_pool
    assert lifetime.term_days is None and lifetime.account_cap_usd is None
    for invalid, why in (({"payout_per_usd": 0.91}, "payout_per_usd"), ({"payout_per_usd": 0}, "payout_per_usd"),
                         ({"share": 0}, "share"), ({"share": 0.6}, "share"), ({"window_periods": 0}, "window"),
                         ({"window_periods": 91}, "window"), ({"term_days": 29}, "term_days"),
                         ({"account_cap_usd": 999}, "account_cap_usd"), ({"excluded": ["5Nope"]}, "ss58"),
                         ({"period_epochs": 0}, "period_epochs")):
        with pytest.raises(ValidationError, match=why):
            with_referrals(invalid)
    both = with_referrals({}, usage={})  # 0.3 per referred USD and 0.6 of the paid usage: 0.9 together
    assert both.scoring.usage_pool.fraction + both.scoring.referral_pool.payout_per_usd == pytest.approx(0.9)
    with pytest.raises(ValidationError, match="add up to at most 0.9"):
        with_referrals({}, usage={"fraction": 0.9})
    assert with_referrals({"rate": 1}).unknown_settings() == ["scoring.referral_pool.rate"]


def test_the_subnet_admin_names_the_referral_pool() -> None:
    from verathos.subnet.admin import referral_pool_line

    assert referral_pool_line(catalog.load(), 100) == "referrals: off"
    line = referral_pool_line(with_referrals({"period_epochs": 2}), 7)
    assert "0.3 USD of emission per referred USD over 30 periods of 2 epochs" in line
    assert "at most 0.1 of the miner emission" in line and "epoch 7 is in period 3, which pays period 2's" in line


# ---- codes ----

def test_codes_are_random_or_readable_and_unique_after_folding() -> None:
    for good in ("alice", "a1ice", "bob-the-miner", "x7k2m9pq", "7k2m9pqx", "abcd", "a" * 24):
        assert well_formed(good), good
    for bad in ("abc", "a" * 25, "Alice", "-alice", "alice-", "al--ice", "4lice", "al_ice", "aliçe", "7k2m9pq0", ""):
        assert not well_formed(bad), bad
    assert fold("A1ice") == fold("alice") == fold("al-ice") and fold("5able") == "sable" and fold("b0b") == "bob"
    for _ in range(20):
        code = random_code()
        assert well_formed(code) and len(code) == 8 and not set(code) & set("01oil")
    assert reserved("verathos-fan") == "verathos" and reserved("decideai") == "decide"
    for look_alike, name in (("v3rathos", "verathos"), ("dec1de", "decide"), ("b1ttensor", "bittensor"),
                             ("adm1n", "adm1n"), ("off1cial", "off1cial"), ("off1c1a1", "off1c1a1"),
                             ("t3am-x", "t3am"), ("h3lp", "h3lp"), ("deciide", None)):
        assert reserved(look_alike) == name, look_alike  # digits read as the letters they look like
    assert reserved("11um", ["lium"]) == "lium" and reserved("he1ium", ["lium"]) == "lium"
    assert reserved("team") == "team" and reserved("the-team") == "team" and reserved("te-am") == "te-am"
    assert reserved("steamboat") is None and reserved("helpful") is None  # words only whole or as a part
    assert reserved("kev4bfan", ["kev-4b"]) == "kev-4b" and reserved("chutesfan", ["chutes"]) == "chutes"
    assert reserved("alice") is None and reserved("supp0rt") == "supp0rt"


def admin(tmp_path, **config):
    from verathos.gateway.admin import ByPort, admin_app

    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")},
              **({"referral_site": SITE, "referral_key_file": tmp_path / "referral.key"} | config))
    s.app = ByPort(s.app, 9999, admin_app(s.gateway, token=ADMIN_TOKEN, accounts_token=ACCOUNTS_TOKEN))
    return s


def call(s, method: str, path: str, body=None, token: str = ACCOUNTS_TOKEN) -> httpx.Response:
    return asyncio.run(admin_call(s, method, path, json_=body, token=token))


def test_a_hotkey_registers_codes_it_signed_for_this_site_within_the_rules(tmp_path) -> None:
    s = admin(tmp_path)
    miner, other = kp(), kp()
    first = call(s, "POST", "/admin/v1/referral-codes", registration(miner, "alice"))
    assert first.status_code == 201
    assert first.json() | {"created_at": None} == {"code": "alice", "hotkey": miner.ss58_address, "created_at": None,
                                                   "revoked_at": None}
    assert call(s, "POST", "/admin/v1/referral-codes", registration(miner, "alice")).status_code == 200  # again
    refusals = {
        "another hotkey's": (registration(other, "alice"), 409, "taken"),
        "a look-alike": (registration(other, "a1ice"), 409, "reads as the code 'alice'"),
        "reserved": (registration(other, "verathos-team"), 400, "reserved"),
        "a model id": (registration(other, "kev-4b-fast"), 400, "reserved"),
        "malformed": (registration(other, "Bob"), 400, "not a code"),
        "stale": (registration(other, "bobby", issued_ms=now_ms() - 301_000), 400, "5 minutes"),
        "ahead": (registration(other, "bobby", issued_ms=now_ms() + 120_000), 400, "ahead"),
        "another site": (registration(other, "bobby", site="https://elsewhere.example"), 400, "site"),
        "another netuid": (registration(other, "bobby", netuid=96), 400, "netuid"),
        "signed by another key": ({**registration(other, "bobby"), "signature": registration(miner, "bobby")[
            "signature"]}, 400, "signature"),
        "an extra field": ({**registration(other, "bobby"), "note": "x"}, 400, "note"),
    }
    for what, (body, status, why) in refusals.items():
        r = call(s, "POST", "/admin/v1/referral-codes", body)
        assert (r.status_code, why in r.json()["error"]["message"]) == (status, True), what
    for code in ("x7k2m9pq", "bob-the-miner"):
        assert call(s, "POST", "/admin/v1/referral-codes", registration(miner, code)).status_code == 201
    fourth = call(s, "POST", "/admin/v1/referral-codes", registration(miner, "carol"))
    assert fourth.status_code == 409 and f"{MAX_CODES} codes" in fourth.json()["error"]["message"]

    found = call(s, "GET", "/admin/v1/referral-codes/A1ICE").json()  # typed in any case, read as it folds
    assert (found["code"], found["hotkey"]) == ("alice", miner.ss58_address)
    assert call(s, "GET", "/admin/v1/referral-codes/nobody").status_code == 404
    assert call(s, "GET", "/admin/v1/referral-codes/n%20b").status_code == 404
    assert call(s, "DELETE", "/admin/v1/referral-codes/alice").status_code == 403  # the operator's only
    revoked = call(s, "DELETE", "/admin/v1/referral-codes/alice", token=ADMIN_TOKEN).json()
    assert revoked["revoked_at"] is not None
    assert call(s, "POST", "/admin/v1/referral-codes", registration(miner, "carol")).status_code == 201  # a slot
    assert call(s, "POST", "/admin/v1/referral-codes", registration(other, "alice")).status_code == 409  # never reused

    off = admin(tmp_path / "off", referral_site=None)
    assert call(off, "POST", "/admin/v1/referral-codes", registration(miner, "dave")).status_code == 503


def test_codes_move_to_another_hotkey_signed_by_it_never_by_the_old_one(tmp_path) -> None:
    s = admin(tmp_path)
    old, new = kp(), kp()
    for code in ("alice", "x7k2m9pq"):
        assert call(s, "POST", "/admin/v1/referral-codes", registration(old, code)).status_code == 201
    by_old = call(s, "POST", "/admin/v1/referral-codes/move", move(old, old.ss58_address, new.ss58_address))
    assert by_old.status_code == 400 and "signature" in by_old.json()["error"]["message"]
    signed = move(new, old.ss58_address, new.ss58_address)
    moved = call(s, "POST", "/admin/v1/referral-codes/move", signed)
    assert moved.status_code == 200 and moved.json() == {"from_hotkey": old.ss58_address,
                                                         "to_hotkey": new.ss58_address,
                                                         "codes": ["alice", "x7k2m9pq"]}
    assert call(s, "POST", "/admin/v1/referral-codes/move", signed).json() == moved.json()  # once
    assert call(s, "GET", "/admin/v1/referral-codes/alice").json()["hotkey"] == new.ss58_address
    nothing = call(s, "POST", "/admin/v1/referral-codes/move", move(new, old.ss58_address, new.ss58_address))
    assert nothing.status_code == 404
    same = call(s, "POST", "/admin/v1/referral-codes/move", move(new, new.ss58_address, new.ss58_address))
    assert same.status_code == 400


def test_a_move_logs_the_codes_it_moved_not_the_ones_the_hotkey_held_before(tmp_path, caplog) -> None:
    s = admin(tmp_path)
    old, new = kp(), kp()
    for keypair, code in ((old, "alice"), (new, "bobby")):
        assert call(s, "POST", "/admin/v1/referral-codes", registration(keypair, code)).status_code == 201
    caplog.set_level(logging.INFO, logger="verathos.gateway.admin")
    signed = move(new, old.ss58_address, new.ss58_address)
    assert call(s, "POST", "/admin/v1/referral-codes/move", signed).json()["codes"] == ["alice", "bobby"]
    assert call(s, "POST", "/admin/v1/referral-codes/move", signed).status_code == 200  # again: nothing moves
    logged = [r.getMessage() for r in caplog.records if " moved from " in r.getMessage()]
    assert logged == [f"referral codes alice moved from {old.ss58_address} to {new.ss58_address} by the accounts "
                      "token from 127.0.0.1"]


def test_an_accounts_referrer_is_set_once_and_fixed_by_its_first_top_up(tmp_path) -> None:
    s = admin(tmp_path)
    miner = kp()
    for code in ("alice", "bobby"):
        assert call(s, "POST", "/admin/v1/referral-codes", registration(miner, code)).status_code == 201
    for name in ("u_1", "u_2", "u_3"):
        assert call(s, "POST", "/admin/v1/accounts", {"name": name, "rate_rps": 1, "burst": 1,
                                                      "prepaid": True}).status_code == 201
    assert call(s, "GET", "/admin/v1/accounts/u_1/referral").json() == {"referral": None, "topped_up": False}
    set_ = call(s, "PUT", "/admin/v1/accounts/u_1/referral", {"code": "A1ICE"})
    assert set_.status_code == 201 and (set_.json()["code"], set_.json()["hotkey"]) == ("alice", miner.ss58_address)
    assert call(s, "PUT", "/admin/v1/accounts/u_1/referral", {"code": "alice"}).status_code == 200  # the same, again
    shopping = call(s, "PUT", "/admin/v1/accounts/u_1/referral", {"code": "bobby"})
    assert shopping.status_code == 409 and "fixed" in shopping.json()["error"]["message"]

    topup = {"amount": 5 * USD, "kind": "topup", "reference": "pay-u2"}
    assert call(s, "POST", "/admin/v1/accounts/u_2/credits", topup).status_code == 201
    late = call(s, "PUT", "/admin/v1/accounts/u_2/referral", {"code": "alice"})
    assert late.status_code == 409 and "topped up" in late.json()["error"]["message"]
    assert call(s, "GET", "/admin/v1/accounts/u_2/referral").json() == {"referral": None, "topped_up": True}
    grant = {"amount": USD, "kind": "grant", "reference": "signup:u_3"}
    assert call(s, "POST", "/admin/v1/accounts/u_3/credits", grant).status_code == 201
    assert call(s, "PUT", "/admin/v1/accounts/u_3/referral", {"code": "bobby"}).status_code == 201  # grants: still open
    assert call(s, "PUT", "/admin/v1/accounts/u_9/referral", {"code": "alice"}).status_code == 404
    assert call(s, "PUT", "/admin/v1/accounts/acme/referral", {"code": "alice"}).status_code == 403  # not its account
    assert call(s, "PUT", "/admin/v1/accounts/u_3/referral", {"code": "nobody"}).status_code == 404
    call(s, "DELETE", "/admin/v1/referral-codes/bobby", token=ADMIN_TOKEN)
    assert call(s, "POST", "/admin/v1/accounts", {"name": "u_4", "rate_rps": 1, "burst": 1,
                                                  "prepaid": True}).status_code == 201
    assert call(s, "PUT", "/admin/v1/accounts/u_4/referral", {"code": "bobby"}).status_code == 404  # revoked
    assert call(s, "GET", "/admin/v1/accounts/u_3/referral").json()["referral"]["code"] == "bobby"  # kept


# ---- refunds ----

def test_a_refund_returns_at_most_what_is_left_of_a_top_up_of_the_account(tmp_path) -> None:
    s = admin(tmp_path)
    for name in ("u_1", "u_2"):
        call(s, "POST", "/admin/v1/accounts", {"name": name, "rate_rps": 1, "burst": 1, "prepaid": True})
    call(s, "POST", "/admin/v1/accounts/u_1/credits", {"amount": 10 * USD, "kind": "topup", "reference": "pay-1"})
    call(s, "POST", "/admin/v1/accounts/u_1/credits", {"amount": USD, "kind": "grant", "reference": "signup:u_1"})
    call(s, "POST", "/admin/v1/accounts/u_2/credits", {"amount": USD, "kind": "topup", "reference": "pay-2"})

    def refund(amount: int, of: str | None, reference: str, account: str = "u_1", token: str = ACCOUNTS_TOKEN):
        body = {"amount": amount, "kind": "refund", "reference": reference, "refund_of": of}
        return call(s, "POST", f"/admin/v1/accounts/{account}/credits", body, token=token)
    assert refund(-4 * USD, "pay-1", "refund-1").status_code == 201
    assert refund(-4 * USD, "pay-1", "refund-1").status_code == 200  # recorded once
    for amount, of, why in ((-7 * USD, "pay-1", "left to refund"), (-USD, "signup:u_1", "names no top-up"),
                            (-USD, "pay-2", "names no top-up"), (USD, "pay-1", "negative"),
                            (-USD, None, "refund_of")):
        r = refund(amount, of, f"r{amount}{of}")
        assert r.status_code == 400 and why in r.json()["error"]["message"], why
    topup_naming = call(s, "POST", "/admin/v1/accounts/u_1/credits",
                        {"amount": USD, "kind": "topup", "reference": "x", "refund_of": "pay-1"})
    assert topup_naming.status_code == 400
    assert refund(-6 * USD, "pay-1", "refund-2").status_code == 201  # the rest of it
    assert refund(-1, "pay-1", "refund-3").status_code == 400
    assert call(s, "GET", "/admin/v1/accounts/u_1/credits").json()["balance"] == USD
    takes_back = {"amount": -USD // 2, "kind": "adjust", "reference": "bonus-refund:pay-1"}
    assert call(s, "POST", "/admin/v1/accounts/u_1/credits", takes_back).status_code == 201
    adds = {"amount": USD, "kind": "adjust", "reference": "gift"}
    assert call(s, "POST", "/admin/v1/accounts/u_1/credits", adds).status_code == 403  # the operator's only
    ledger = call(s, "GET", "/admin/v1/accounts/u_1/credits").json()["credits"]
    assert [(c["kind"], c["refund_of"]) for c in ledger][:2] == [("adjust", None), ("refund", "pay-1")]
    totals = call(s, "GET", "/admin/v1/usage/totals").json()
    assert totals["topups_micros"]["all_time"] == USD  # pay-1 returned in full: only pay-2 stays bought


def test_paid_usage_counts_the_credit_bought_less_its_refunds(tmp_path) -> None:
    s = Setup(tmp_path, {"10.0.0.1": MockEngine("kev-4b")})
    asyncio.run(s.accounts.set_account("acme", prepaid=True))
    principal = asyncio.run(s.accounts.authenticate(s.key))
    asyncio.run(s.accounts.add_credit("acme", 2 * USD, kind="grant", reference="g"))
    asyncio.run(s.accounts.add_credit("acme", 5 * USD, kind="topup", reference="t"))
    asyncio.run(s.accounts.add_credit("acme", -4 * USD, kind="refund", reference="r", refund_of="t"))
    principal = asyncio.run(s.accounts.authenticate(s.key))
    asyncio.run(s.meter.read_balance(principal.account_id))
    assert not s.meter.paid(principal)  # 3 USD left of which 2 are the grant's: bought credit is not spent yet
    asyncio.run(s.accounts.add_credit("acme", -2 * USD, kind="adjust", reference="a"))
    asyncio.run(s.meter.read_balance(principal.account_id))
    assert s.meter.paid(principal)  # 1 USD left, no more than the 1 USD bought


# ---- what counts ----

POOL = with_referrals({"period_epochs": 2, "window_periods": 3, "term_days": 30,
                       "account_cap_usd": 1000}).scoring.referral_pool
WINDOW_S = POOL.window_s(180, 12.0)  # 3 periods of 2 epochs: 3.6 hours


REFERRAL_KEY = bytes(range(32))  # the gateway's referral key (``referral_key_file``)


class Ledger:
    """The referral ledger on its own database: accounts, codes, referrals and credits at chosen times."""

    def __init__(self, tmp_path, key: bytes = REFERRAL_KEY) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        url = f"sqlite+aiosqlite:///{tmp_path}/gateway.db"
        db.upgrade(url)
        self.engine = db.create_engine(url)
        self.accounts = Accounts(self.engine)
        self.referrals = Referrals(self.engine, netuid=405, site=SITE, catalog=catalog.load, key=key)
        self.miner = kp()
        self.t0 = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)

    def run(self, coro):
        return asyncio.run(coro)

    def code(self, code: str, keypair=None) -> None:
        signed = SignedRegistration.model_validate(registration(keypair or self.miner, code))
        self.run(self.referrals.register(signed))

    def account(self, name: str, code: str | None = None, at_s: float = 0.0) -> None:
        self.run(self.accounts.create_account(name, rate_rps=1, burst=1, prepaid=True))
        if code is not None:
            self.run(self.referrals.refer(name, code))
            account = select(db.accounts.c.id).where(db.accounts.c.name == name).scalar_subquery()
            self.run(self._at(db.referrals, db.referrals.c.account_id == account,
                              referred_at=self.t0 + dt.timedelta(seconds=at_s)))

    def credit(self, account: str, usd: float, at_s: float, kind: str = "topup", refund_of: str | None = None,
               reference: str | None = None) -> str:
        reference = reference or f"{account}:{kind}:{at_s}:{usd}"
        self.run(self.accounts.add_credit(account, round(usd * USD), kind=kind, reference=reference,
                                          refund_of=refund_of))
        self.run(self._at(db.credits, db.credits.c.reference == reference,
                          created_at=self.t0 + dt.timedelta(seconds=at_s)))
        return reference

    async def _at(self, table, where, **values) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(update(table).where(where).values(**values))

    def decide(self, pool=POOL) -> list[float]:
        return [micros / USD for micros in self.run(self.referrals.decide(pool, 180))]

    def entry_ids(self) -> dict[str, str]:
        """Each decided credit's reference -> the id its entry goes to the auditor under."""
        async def read():
            async with self.engine.connect() as conn:
                return dict((await conn.execute(select(db.credits.c.reference, db.referral_reports.c.entry_id).join(
                    db.referral_reports, db.referral_reports.c.credit_id == db.credits.c.id))).all())
        return self.run(read())


def test_top_ups_count_from_the_referral_within_the_term_and_until_the_code_is_revoked(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice", at_s=100)
    ledger.account("u_2")  # not referred: nothing of it is decided
    ledger.credit("u_1", 5, at_s=50)  # before the referral (a code entered with the money already moving)
    ledger.credit("u_1", 7, at_s=200)
    ledger.credit("u_2", 9, at_s=200)
    ledger.credit("u_1", 1, at_s=300, kind="grant")
    ledger.credit("u_1", 11, at_s=100 + 30 * 86400)  # the term ends 30 days after the referral
    assert ledger.decide() == [0, 7, 0]
    assert ledger.decide() == []  # decided once
    ledger.credit("u_1", 3, at_s=400)
    ledger.run(ledger.referrals.revoke("alice"))
    ledger.credit("u_1", 4, at_s=dt.datetime.now(dt.UTC).timestamp() - ledger.t0.timestamp() + 60)
    assert ledger.decide() == [3, 0]  # credited before the revocation it counts, after it nothing
    ledger.code("bobby")
    ledger.account("u_3", "bobby", at_s=0)
    ledger.credit("u_3", 8, at_s=10)
    assert ledger.decide(pool=None) == [0]  # no referral pool in the catalog: nothing counts, then or later
    assert ledger.decide() == []


def test_the_account_cap_holds_within_any_window_whatever_the_order_top_ups_are_decided_in(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    ledger.credit("u_1", 600, at_s=10)
    ledger.credit("u_1", 600, at_s=20)
    assert ledger.decide() == [600, 400]  # 1,000 USD per window at most
    ledger.credit("u_1", 50, at_s=WINDOW_S - 100)
    assert ledger.decide() == [0]
    ledger.credit("u_1", 900, at_s=WINDOW_S + 15)  # a window's length after the first: the second's 400 remain
    assert ledger.decide() == [600]
    ledger.credit("u_1", 500, at_s=WINDOW_S + 25)  # after the second too: the 600 just counted remain
    assert ledger.decide() == [400]
    # a top-up credited earlier but decided later still leaves every window within the cap
    late = Ledger(tmp_path / "late")
    late.code("alice")
    late.account("u_1", "alice")
    late.credit("u_1", 900, at_s=1000)
    assert late.decide() == [900]
    late.credit("u_1", 900, at_s=500)
    assert late.decide() == [100]
    freed = Ledger(tmp_path / "freed")
    freed.code("alice")
    freed.account("u_1", "alice")
    pay = freed.credit("u_1", 1000, at_s=10)
    freed.credit("u_1", -1000, at_s=20, kind="refund", refund_of=pay)
    freed.credit("u_1", 500, at_s=30)
    assert freed.decide() == [1000, -1000, 0]  # a refund frees none of the cap
    uncapped = Ledger(tmp_path / "uncapped")
    uncapped.code("alice")
    uncapped.account("u_1", "alice")
    uncapped.credit("u_1", 5000, at_s=10)
    assert uncapped.decide(POOL.model_copy(update={"account_cap_usd": None})) == [5000]


def test_a_refund_reverses_what_its_top_up_counted_and_no_more(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    pay = ledger.credit("u_1", 1500, at_s=10)  # counts 1,000 (the cap)
    small = ledger.credit("u_1", 20, at_s=WINDOW_S + 100)
    assert ledger.decide() == [1000, 20]
    ledger.credit("u_1", -600, at_s=200, kind="refund", refund_of=pay)
    ledger.credit("u_1", -600, at_s=300, kind="refund", refund_of=pay)
    ledger.credit("u_1", -20, at_s=WINDOW_S + 200, kind="refund", refund_of=small)
    assert ledger.decide() == [-600, -400, -20]  # 1,000 counted, 1,000 reversed
    ledger.account("u_2")
    other = ledger.credit("u_2", 5, at_s=10)
    ledger.credit("u_2", -5, at_s=20, kind="refund", refund_of=other)
    assert ledger.decide() == []  # not referred: neither counts
    ledger.code("bobby")
    ledger.account("u_3", "bobby", at_s=100)
    early = ledger.credit("u_3", 5, at_s=50)
    ledger.credit("u_3", -5, at_s=150, kind="refund", refund_of=early)
    assert ledger.decide() == [0, 0]  # its top-up counted nothing: nor does its refund
    after = ledger.credit("u_3", 9, at_s=200)
    assert ledger.decide(pool=None) == [0]
    ledger.credit("u_3", -9, at_s=300, kind="refund", refund_of=after)
    assert ledger.decide() == [0]


def test_a_referrers_figures_are_counts_and_sums_per_day_and_code_never_accounts(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.code("x7k2m9pq")
    ledger.code("other", kp())
    today = dt.datetime.now(dt.UTC).replace(hour=12, minute=0, second=0, microsecond=0)
    ledger.t0 = today - dt.timedelta(days=2)  # two days ago, at noon
    ledger.account("u_1", "alice", at_s=0)
    ledger.account("u_2", "alice", at_s=3600)
    ledger.account("u_3", "x7k2m9pq", at_s=2 * 86400)  # today
    ledger.account("u_4", "other", at_s=0)
    pay = ledger.credit("u_1", 10, at_s=60)
    ledger.credit("u_1", 5, at_s=120)
    ledger.credit("u_2", 1500, at_s=4000)
    ledger.credit("u_3", 2, at_s=2 * 86400 + 10)
    ledger.credit("u_4", 99, at_s=60)
    ledger.credit("u_1", -10, at_s=86400, kind="refund", refund_of=pay)
    ledger.decide()
    figures = ledger.run(ledger.referrals.referrer(ledger.miner.ss58_address, days=3))
    assert [c["code"] for c in figures["codes"]] == ["alice", "x7k2m9pq"]
    assert [c["referred"] for c in figures["codes"]] == [2, 1]
    assert [b["start"] for b in figures["buckets"]] == [(today.date() - dt.timedelta(days=d)).isoformat()
                                                         for d in (2, 1, 0)]
    first, second, third = (b["codes"] for b in figures["buckets"])
    assert first["alice"] == {"referred": 2, "paying": 2, "topups_micros": 1515 * USD, "refunds_micros": 0,
                              "counted_micros": 1015 * USD}  # u_2's 1,500 counts the account cap
    assert second["alice"] == {"referred": 0, "paying": 0, "topups_micros": 0, "refunds_micros": 10 * USD,
                               "counted_micros": -10 * USD}
    assert third["x7k2m9pq"]["topups_micros"] == 2 * USD and third["alice"]["referred"] == 0
    assert figures["totals"] == {"referred": 3, "paying": 3, "topups_micros": 1517 * USD, "refunds_micros": 10 * USD,
                                 "counted_micros": 1007 * USD}
    assert "u_1" not in str(figures) and pay not in str(figures)
    assert ledger.run(ledger.referrals.referrer(ledger.miner.ss58_address, days=1))["totals"]["referred"] == 1
    with pytest.raises(ValueError):
        ledger.run(ledger.referrals.referrer(ledger.miner.ss58_address, days=0))
    assert ledger.run(ledger.referrals.referrer(kp().ss58_address))["codes"] == []


def test_the_dashboard_figures_reach_the_account_service(tmp_path) -> None:
    s = admin(tmp_path)
    miner = kp()
    call(s, "POST", "/admin/v1/referral-codes", registration(miner, "alice"))
    figures = call(s, "GET", f"/admin/v1/referrers/{miner.ss58_address}?days=7").json()
    assert figures["days"] == 7 and len(figures["buckets"]) == 7 and figures["codes"][0]["code"] == "alice"
    assert call(s, "GET", f"/admin/v1/referrers/{miner.ss58_address}?days=0").status_code == 400
    assert call(s, "GET", f"/admin/v1/referrers/{miner.ss58_address}?days=x").status_code == 400


# ---- reports ----

def test_what_counts_is_reported_once_with_the_codes_registrations_and_moves(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    pay = ledger.credit("u_1", 5, at_s=10)
    ledger.credit("u_1", 2000, at_s=20)  # 995 of it counts
    ledger.credit("u_1", 1, at_s=30, kind="grant")
    ledger.account("u_2", "alice")
    ledger.credit("u_2", 3, at_s=10 + 31 * 86400)  # after the term: counts nothing, never reported
    new = kp()
    ledger.run(ledger.referrals.move(SignedMove.model_validate(move(new, ledger.miner.ss58_address,
                                                                     new.ss58_address))))
    ledger.decide()
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport, answer=None):
        posted.append(report)
        return answer or {"codes": len(report.codes), "moves": len(report.moves), "entries": len(report.entries)}

    async def failing(report: ReferralReport):
        posted.append(report)
        raise httpx.ConnectError("the auditor is down")

    with pytest.raises(httpx.ConnectError):
        ledger.run(ledger.referrals.report(failing))
    assert ledger.run(ledger.referrals.report(post)) == 4
    assert posted[0] == posted[1]  # what failed goes again as it was
    report = posted[1]
    assert [c.registration.code for c in report.codes] == ["alice"] and report.codes[0].verify()
    assert report.moves[0].move.to_hotkey == new.ss58_address and report.moves[0].verify()
    assert [(e.code, e.amount_micros) for e in report.entries] == [("alice", 5 * USD), ("alice", 995 * USD)]
    credited = int((ledger.t0 + dt.timedelta(seconds=10)).timestamp() * 1000)
    assert report.entries[0].credited_ms == credited - credited % 3_600_000  # the start of its hour
    assert ledger.run(ledger.referrals.report(post)) == 0  # taken: nothing more
    ledger.credit("u_1", -5, at_s=40, kind="refund", refund_of=pay)
    ledger.decide()
    assert ledger.run(ledger.referrals.report(post)) == 1
    assert [(e.code, e.amount_micros) for e in posted[-1].entries] == [("alice", -5 * USD)]

    # an auditor that started over no longer knows the code: the registry goes again, with the entry
    ledger.credit("u_1", 7, at_s=WINDOW_S + 50)
    ledger.decide()
    assert ledger.run(ledger.referrals.report(lambda r: post(r, {"unknown_codes": ["alice"]}))) == 1
    assert ledger.run(ledger.referrals.report(post)) == 3
    assert ([c.registration.code for c in posted[-1].codes], len(posted[-1].moves),
            [e.amount_micros for e in posted[-1].entries]) == (["alice"], 1, [7 * USD])


def test_an_entry_is_named_by_the_keyed_hash_of_its_credits_reference_in_every_post_and_after_a_restore(
        tmp_path) -> None:
    """An entry's id is the HMAC of its credit's reference with the gateway's referral key: every post of the entry
    names it alike, after a failed post or an auditor that started over; a gateway whose database was restored from a
    backup, which records the payment again under its reference, names it alike too, so the auditor, which took the
    id before, counts it once; under another key the id is another, and without the key nothing links it to the
    credit."""
    from verathos.gateway.referrals import entry_id

    def named(path, key: bytes = REFERRAL_KEY) -> str:
        ledger = Ledger(path, key)
        ledger.code("alice")
        ledger.account("u_1", "alice")
        reference = ledger.credit("u_1", 5, at_s=10, reference="tao:5120344:7")
        ledger.decide()
        kept = ledger.entry_ids()[reference]
        posted: list[ReferralReport] = []

        async def post(report: ReferralReport, answer=None, fail: bool = False):
            posted.append(report)
            if fail:
                raise httpx.ConnectError("the auditor is down")
            return answer or {}
        with pytest.raises(httpx.ConnectError):
            ledger.run(ledger.referrals.report(lambda r: post(r, fail=True)))
        ledger.run(ledger.referrals.report(lambda r: post(r, {"unknown_codes": ["alice"]})))
        assert ledger.run(ledger.referrals.report(post)) == 2  # the registration again, with the entry
        assert [[e.id for e in r.entries] for r in posted] == [[kept]] * 3
        assert ledger.entry_ids() == {reference: kept} and ledger.run(ledger.referrals.report(post)) == 0
        return kept
    first, restored = named(tmp_path / "a"), named(tmp_path / "restored")  # a fresh database, the same key
    assert first == restored == entry_id(REFERRAL_KEY, "tao:5120344:7") and re.fullmatch(r"[0-9a-f]{64}", first)
    assert named(tmp_path / "other", bytes(32)) != first


def hashes(text: str) -> set[str]:
    """``text`` hashed with every algorithm hashlib guarantees, in hex (the extendable ones to 32 and 64 bytes)."""
    found = set()
    for name in hashlib.algorithms_guaranteed:
        digest = hashlib.new(name, text.encode())
        found |= {digest.hexdigest(n) for n in (32, 64)} if name.startswith("shake") else {digest.hexdigest()}
    return found


def test_nothing_reported_to_the_auditor_is_or_hashes_to_a_credits_reference(tmp_path) -> None:
    """A top-up's reference names its payment (a TAO top-up's, the transfer on chain): no field of a post to the
    auditor is a reference or a hash of one, nor holds one."""
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    tao = ledger.credit("u_1", 5, at_s=10, reference="tao:5120344:7")
    references = [tao, ledger.credit("u_1", 3, at_s=20, reference=f"x402:base:0x{'ab' * 20}:0x{'cd' * 32}"),
                  ledger.credit("u_1", -2, at_s=30, kind="refund", refund_of=tao, reference=f"refund:{tao}")]
    new = kp()
    ledger.run(ledger.referrals.move(SignedMove.model_validate(move(new, ledger.miner.ss58_address,
                                                                     new.ss58_address))))
    ledger.decide()
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport):
        posted.append(report)
        return {}
    while ledger.run(ledger.referrals.report(post)):
        pass
    assert [len(r.entries) for r in posted] == [3] and posted[0].codes and posted[0].moves

    def fields(value) -> list[str]:
        if isinstance(value, dict):
            return [f for v in value.values() for f in fields(v)]
        if isinstance(value, list):
            return [f for v in value for f in fields(v)]
        return [str(value)]
    linked = set(references).union(*map(hashes, references))
    bodies = [r.model_dump_json() for r in posted]
    assert not linked & {f for body in bodies for f in fields(json.loads(body))}
    assert not [x for x in linked for body in bodies if x in body]
    # the credit time goes to the hour: with the amount, a finer one could find the payment
    assert {e.credited_ms % 3_600_000 for r in posted for e in r.entries} == {0}


def test_the_reports_kept_get_random_ids_when_the_database_is_upgraded(tmp_path) -> None:
    """A database from before the ids (0005) is upgraded in place: every report kept gets a random id of its own,
    those the auditor has not taken go to it under theirs, and an id is required and unique."""
    from alembic import command
    from sqlalchemy import insert, text
    from sqlalchemy.exc import IntegrityError

    url = f"sqlite+aiosqlite:///{tmp_path}/old.db"
    command.upgrade(db.alembic_config(url), "0005")
    miner, references = kp(), ["tao:5120344:7", "tao:5120399:2", "tao:5120402:0"]

    async def old_rows():
        engine = db.create_engine(url)
        async with engine.begin() as conn:
            await conn.execute(text("INSERT INTO accounts (name, rate_rps, burst) VALUES ('u_1', 1, 1)"))
            for reference in references:
                await conn.execute(text("INSERT INTO credits (account_id, amount, kind, reference) VALUES "
                                        "(1, 5000000, 'topup', :reference)"), {"reference": reference})
            await conn.execute(text("INSERT INTO referral_codes (code, folded, hotkey, registration, reported_at) "
                                    "VALUES ('alice', 'alice', :hotkey, :registration, CURRENT_TIMESTAMP)"),
                               {"hotkey": miner.ss58_address, "registration": json.dumps(registration(miner, "alice"))})
            await conn.execute(text("INSERT INTO referrals (account_id, code) VALUES (1, 'alice')"))
            await conn.execute(text("INSERT INTO referral_reports (credit_id, code, amount, reported_at) VALUES "
                                    "(1, 'alice', 5000000, CURRENT_TIMESTAMP), (2, 'alice', 5000000, NULL), "
                                    "(3, 'alice', 0, NULL)"))  # taken, not taken yet, counting nothing
        await engine.dispose()

    asyncio.run(old_rows())
    db.upgrade(url)
    engine = db.create_engine(url)

    async def kept() -> dict[int, str]:
        async with engine.connect() as conn:
            return dict((await conn.execute(select(db.referral_reports.c.credit_id,
                                                   db.referral_reports.c.entry_id))).all())
    ids = asyncio.run(kept())
    assert sorted(ids) == [1, 2, 3] and len(set(ids.values())) == 3
    assert all(re.fullmatch(r"[0-9a-f]{64}", id_) for id_ in ids.values())
    assert not set(ids.values()) & set().union(*map(hashes, references))
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport):
        posted.append(report)
        return {}
    assert asyncio.run(Referrals(engine, netuid=405, site=SITE).report(post)) == 1
    assert [e.id for e in posted[0].entries] == [ids[2]]

    async def report_with(entry_id: str | None) -> None:
        async with engine.begin() as conn:
            await conn.execute(text("INSERT INTO credits (account_id, amount, kind, reference) VALUES "
                                    "(1, 1000000, 'topup', 'tao:5120500:1')"))
            await conn.execute(insert(db.referral_reports).values(credit_id=4, code="alice", amount=USD,
                                                                  entry_id=entry_id))
    for taken in (ids[1], None):
        with pytest.raises(IntegrityError):
            asyncio.run(report_with(taken))
    asyncio.run(engine.dispose())


# ---- the view token ----

def test_a_view_token_is_valid_for_its_hotkey_site_and_netuid_until_it_expires() -> None:
    miner, now = kp(), now_ms()
    view = ReferralView(netuid=405, hotkey=miner.ss58_address, site=SITE, expires_ms=now + 3600_000)
    token = view_token(view, miner)
    assert "=" not in token and read_view_token(token, netuid=405, site=SITE, now_ms=now) == view
    for why, kwargs in (("expired", {"now_ms": now + 3600_001}), ("netuid or site", {"netuid": 96}),
                        ("netuid or site", {"site": "https://elsewhere.example"})):
        with pytest.raises(ValueError, match=why):
            read_view_token(token, **({"netuid": 405, "site": SITE, "now_ms": now} | kwargs))
    long = view_token(view.model_copy(update={"expires_ms": now + 25 * 3600_000}), miner)
    with pytest.raises(ValueError, match="longer than a day"):
        read_view_token(long, netuid=405, site=SITE, now_ms=now)
    forged = token.split(".")[0] + "." + view_token(view, kp()).split(".")[1]
    with pytest.raises(ValueError, match="signature"):
        read_view_token(forged, netuid=405, site=SITE, now_ms=now)
    with pytest.raises(ValueError, match="not a referral view token"):
        read_view_token("garbage", netuid=405, site=SITE, now_ms=now)


def test_the_operator_refunds_a_top_up_and_revokes_a_code_from_the_command_line(tmp_path, capsys) -> None:
    import json
    import sys

    from verathos.gateway.__main__ import main

    url = f"sqlite+aiosqlite:///{tmp_path}/admin.db"
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps({"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "database_url": url,
                                  "referral_site": SITE, "referral_key_file": str(tmp_path / "referral.key")}))

    def cli(*args: str):
        sys.argv = ["verathos-gateway", "--config", str(config), *args]
        main()
        out = capsys.readouterr().out
        return json.loads(out) if args[0] != "db" else out

    cli("db", "upgrade")
    cli("accounts", "create", "--name", "acme", "--rate", "5", "--burst", "10")
    cli("credits", "add", "--account", "acme", "--usd", "5", "--kind", "topup", "--reference", "pay-1")
    refund = cli("credits", "add", "--account", "acme", "--usd", "-2", "--kind", "refund", "--reference", "back-1",
                 "--refund-of", "pay-1")
    assert refund["created"] and refund["credit"]["refund_of"] == "pay-1"
    with pytest.raises(SystemExit):
        cli("credits", "add", "--account", "acme", "--usd", "-4", "--kind", "refund", "--reference", "back-2",
            "--refund-of", "pay-1")
    assert "left to refund" in capsys.readouterr().err
    engine = db.create_engine(url)
    referrals = Referrals(engine, netuid=405, site=SITE)
    asyncio.run(referrals.register(SignedRegistration.model_validate(registration(kp(), "alice"))))
    assert cli("referrals", "show", "--code", "a1ice")["revoked_at"] is None
    assert cli("referrals", "revoke", "--code", "alice")["revoked_at"] is not None
    with pytest.raises(SystemExit):
        cli("referrals", "show", "--code", "nobody")
    assert "no referral code" in capsys.readouterr().err


def test_entries_wait_until_every_registration_is_out(tmp_path, monkeypatch) -> None:
    """More registrations than one post holds go first, over several posts; the entries follow, so the auditor never
    hears of a code it does not know (and a registry sent again after its database started over gets through)."""
    import verathos.gateway.referrals as referrals_module

    monkeypatch.setattr(referrals_module, "MAX_REFERRAL_SIGNED", 1)
    ledger = Ledger(tmp_path)
    for code in ("alice", "bobby"):
        ledger.code(code)
        ledger.account(f"u_{code}", code)
        ledger.credit(f"u_{code}", 5, at_s=10)
    ledger.decide()
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport, answer=None):
        posted.append(report)
        return answer or {}
    for _ in range(3):
        ledger.run(ledger.referrals.report(post))
    assert [([c.registration.code for c in r.codes], len(r.entries)) for r in posted] == [
        (["alice"], 0), (["bobby"], 2), ([], 0)][:len(posted)]
    assert len(posted) == 2  # the third round had nothing to post
    ledger.credit("u_alice", 1, at_s=WINDOW_S + 99)
    ledger.decide()
    ledger.run(ledger.referrals.report(lambda r: post(r, {"unknown_codes": ["alice"]})))
    for _ in range(3):
        ledger.run(ledger.referrals.report(post))
    assert [([c.registration.code for c in r.codes], len(r.entries)) for r in posted[2:]] == [
        ([], 1), (["alice"], 0), (["bobby"], 1)]


def test_a_move_keeps_the_limit_of_codes_per_hotkey(tmp_path) -> None:
    s = admin(tmp_path)
    a, b = kp(), kp()
    for code in ("alice", "anna", "ally"):
        assert call(s, "POST", "/admin/v1/referral-codes", registration(a, code)).status_code == 201
    assert call(s, "POST", "/admin/v1/referral-codes", registration(b, "bobby")).status_code == 201
    over = call(s, "POST", "/admin/v1/referral-codes/move", move(b, a.ss58_address, b.ss58_address))
    assert over.status_code == 409 and "more than 3 codes" in over.json()["error"]["message"]
    call(s, "DELETE", "/admin/v1/referral-codes/bobby", token=ADMIN_TOKEN)  # revoked codes do not count
    assert call(s, "POST", "/admin/v1/referral-codes/move", move(b, a.ss58_address, b.ss58_address)).status_code == 200
    for code in ("amy1", "amy2", "amy3"):  # a holds none now: three more, which cannot join b's
        assert call(s, "POST", "/admin/v1/referral-codes", registration(a, code)).status_code == 201
    assert call(s, "POST", "/admin/v1/referral-codes/move", move(b, a.ss58_address, b.ss58_address)).status_code == 409


def test_one_entry_counts_at_most_what_a_report_carries(tmp_path) -> None:
    ledger = Ledger(tmp_path)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    ledger.credit("u_1", 2_000_000, at_s=10)  # a top-up of 2 million USD, no account cap
    assert ledger.decide(POOL.model_copy(update={"account_cap_usd": None})) == [1_000_000]
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport):
        posted.append(report)
        return {}
    assert ledger.run(ledger.referrals.report(post)) == 2 and posted[0].entries[0].amount_micros == 10**12


def test_registrations_and_moves_are_posted_in_the_order_they_happened(tmp_path) -> None:
    """The auditor applies a move to the codes it took before it: a code registered after a move goes in a later post
    than the move, so the move does not take it along."""
    ledger = Ledger(tmp_path)
    a, b = ledger.miner, kp()
    ledger.code("alice")
    ledger.run(ledger.referrals.move(SignedMove.model_validate(move(b, a.ss58_address, b.ss58_address))))
    ledger.code("anna")  # on a again, after the move
    later = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=5)
    ledger.run(ledger._at(db.referral_codes, db.referral_codes.c.code == "anna", created_at=later))
    posted: list[ReferralReport] = []

    async def post(report: ReferralReport):
        posted.append(report)
        return {}
    while ledger.run(ledger.referrals.report(post)):
        pass
    assert [([c.registration.code for c in r.codes], [m.move.to_hotkey for m in r.moves]) for r in posted] == [
        (["alice"], [b.ss58_address]), (["anna"], [])]



def test_without_a_referral_key_nothing_is_decided_and_a_site_needs_one(tmp_path, caplog) -> None:
    import logging

    from pydantic import ValidationError

    from verathos.gateway.config import GatewayConfig

    ledger = Ledger(tmp_path, key=None)
    ledger.code("alice")
    ledger.account("u_1", "alice")
    ledger.credit("u_1", 5, at_s=10)
    with caplog.at_level(logging.ERROR, logger="verathos.gateway.referrals"):
        assert ledger.decide() == [] and ledger.decide() == []
    assert caplog.messages == ["referrals: no referral key (referral_key_file): no top-up or refund is decided"]
    with pytest.raises(ValidationError, match="referral_site needs referral_key_file"):
        GatewayConfig(netuid=405, wallet_name="w", hotkey_name="h", database_url="sqlite+aiosqlite://",
                      referral_site=SITE)
