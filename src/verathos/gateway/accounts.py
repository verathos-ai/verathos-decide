"""Accounts and their API keys.

A key is ``vd_`` followed by 64 hex characters (32 random bytes). Only its SHA-256 is stored, with its first
``PREFIX_CHARS`` characters as a unique prefix to recognise it. A lookup is cached per key for ``CACHE_TTL_S`` (a
revoked key or a changed limit applies within that time), an unknown key for ``NEGATIVE_TTL_S`` in a cache of its
own, so unknown keys never push valid ones out; concurrent lookups of one key share one query. For an account with
a monthly quota, a lookup also has the usage meter read the account's month.

The hashes of the keys not revoked are kept in memory (``load_keys``, every ``KEYS_RELOAD_S`` and at each key this
process creates): a key not among them is looked up only within a budget shared by all clients
(``UNLISTED_LOOKUPS_PER_S``), for a key another process created since the last reload, so a flood of unknown keys
from many addresses never reaches the database. While the budget is spent, such a key is refused until the reload.

Credit is a ledger (``credits``): each entry names the payment or grant it records (``reference``), so recording one
twice adds it once. The account's balance is kept with it: the entries minus the cost of the usage written. A refund
(negative) names the account's top-up it returns (``refund_of``), and returns at most what is left of it; credit
bought is the top-ups less their refunds (``BOUGHT``).

Usage is paid when the account is prepaid, has topped up (``Principal.paid``), and its free credit is spent: its
balance is no more than the credit it bought (``UsageMeter.paid``). Free credit (a grant: a sign-up, the demo) is
counted as spent first, as ``paid_cost`` counts it, so an account that only ever had free credit is unpaid, and one
that topped up is paid once its grants are used. Only paid usage moves the models' emission shares
(``ServedCount.paid_input_tokens``).

The network's totals (``totals``) and their series per day or week (``series``) count every account's answered
requests. Their paid usage is stricter, money that came in: each account's usage cost beyond its free credit (counted
as spent first), up to what it bought (``paid_cost``). One account's own series (``account_series``) has its
answered and failed requests, input tokens and cost per model, over the same buckets.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Date, case, exists, func, insert, literal_column, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.gateway.db import BOUGHT, accounts, api_keys, credits, usage_daily, usage_hourly
from verathos.gateway.limits import RateLimiter
from verathos.gateway.usage import UsageMeter

log = logging.getLogger(__name__)

KEY_PREFIX = "vd_"
PREFIX_CHARS = 16
_KEY = re.compile(r"vd_[0-9a-f]{64}")
CACHE_TTL_S = 60.0
NEGATIVE_TTL_S = 10.0
MAX_CACHED = 100_000
KEYS_RELOAD_S = 60.0
UNLISTED_LOOKUPS_PER_S, UNLISTED_LOOKUPS_BURST = 20.0, 100  # keys not among the loaded hashes, all clients together
CREDIT_KINDS = ("grant", "topup", "adjust", "refund")  # free credit, a payment, an operator's correction (either
# sign), a payment returned (negative)
SERIES_BUCKETS = {"day": 1, "week": 7}  # days per bucket of the usage series
SERIES_MAX_BUCKETS = 400


class Conflict(Exception):
    """The name or reference is taken by something else."""


@dataclass(frozen=True)
class Principal:
    account_id: int
    key_id: int
    account: str
    rate_rps: float
    burst: int
    monthly_quota: int | None
    prepaid: bool = False
    paid: bool = False  # prepaid, and topped up at least once; paid usage once its grants are spent (see the module)


def new_key() -> str:
    return KEY_PREFIX + secrets.token_hex(32)


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def well_formed_key(key: str | None) -> bool:
    return key is not None and _KEY.fullmatch(key) is not None


def bucket_start(day: dt.date, bucket: str) -> dt.date:
    """The first day of the series bucket (``SERIES_BUCKETS``) holding ``day``: itself, or its week's Monday."""
    return day - dt.timedelta(days=day.weekday()) if bucket == "week" else day


def series_starts(days: int | None, bucket: str, today: dt.date, first: dt.date | None) -> list[dt.date]:
    """The first days of a usage series' buckets (``SERIES_BUCKETS``): those that hold the last ``days`` days, whole,
    up to today's; without ``days``, from the one holding ``first`` (today's without it), at most
    ``SERIES_MAX_BUCKETS``, the latest. Raises ValueError for an unknown bucket, under a day or more buckets than
    that."""
    if bucket not in SERIES_BUCKETS:
        raise ValueError(f"bucket is one of {tuple(SERIES_BUCKETS)}")
    step, last = SERIES_BUCKETS[bucket], bucket_start(today, bucket)
    if days is None:
        start = min(last, max(bucket_start(first or today, bucket),
                              last - dt.timedelta(days=step * (SERIES_MAX_BUCKETS - 1))))
    elif days < 1:
        raise ValueError("days is at least 1")
    else:
        start = bucket_start(today - dt.timedelta(days=days - 1), bucket)
    count = (last - start).days // step + 1
    if count > SERIES_MAX_BUCKETS:
        raise ValueError(f"at most {SERIES_MAX_BUCKETS} buckets")
    return [start + dt.timedelta(days=step * i) for i in range(count)]


def paid_cost(cost: int, grants: int, topups: int) -> int:
    """The part of an account's usage cost paid with bought credit: beyond its grants, up to its top-ups."""
    return max(0, min(topups, cost - grants))


class Accounts:
    def __init__(self, engine: AsyncEngine, meter: UsageMeter | None = None, *, ttl_s: float = CACHE_TTL_S,
                 negative_ttl_s: float = NEGATIVE_TTL_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.engine, self.meter, self.ttl_s, self.negative_ttl_s, self.clock = (engine, meter, ttl_s,
                                                                                negative_ttl_s, clock)
        self._known: dict[str, tuple[float, Principal]] = {}
        self._unknown: dict[str, float] = {}
        self._loading: dict[str, asyncio.Future[Principal | None]] = {}
        self._hashes: set[str] | None = None  # the keys not revoked, as loaded; None: not loaded, every key looked up
        self._unlisted = RateLimiter()

    # ---- the request path ----

    async def authenticate(self, key: str) -> Principal | None:
        """The key's principal, or None if the key is unknown, revoked, expired or its account suspended. Raises on
        database errors."""
        if not well_formed_key(key):
            return None
        h, now = key_hash(key), self.clock()
        known = self._known.get(h)
        if known is not None and known[0] > now:
            return known[1]
        if self._unknown.get(h, 0.0) > now:
            return None
        if self._hashes is not None and h not in self._hashes and h not in self._loading and self._unlisted.check(
                "unlisted", UNLISTED_LOOKUPS_PER_S, UNLISTED_LOOKUPS_BURST, now):
            return None
        loading = self._loading.get(h)
        if loading is None:  # a task of its own: a waiter giving up (its deadline) stops no one else's lookup
            loading = self._loading[h] = asyncio.ensure_future(self._load(h))
            loading.add_done_callback(lambda done: self._loaded(h, done))
        return await asyncio.shield(loading)

    def _loaded(self, h: str, done: asyncio.Future[Principal | None]) -> None:
        del self._loading[h]
        if done.cancelled() or done.exception() is not None:  # retrieved: the waiters get it raised
            return
        principal, now = done.result(), self.clock()
        if principal is None:
            if len(self._unknown) >= MAX_CACHED:
                self._unknown = {k: t for k, t in self._unknown.items() if t > now}
                if len(self._unknown) >= MAX_CACHED:
                    self._unknown.clear()
            self._unknown[h] = now + self.negative_ttl_s
        else:
            if len(self._known) >= MAX_CACHED:
                self._known = {k: v for k, v in self._known.items() if v[0] > now}
            self._known[h] = (now + self.ttl_s, principal)

    async def _load(self, h: str) -> Principal | None:
        now = dt.datetime.now(dt.UTC)
        topped_up = exists().where(credits.c.account_id == accounts.c.id, credits.c.kind == "topup")
        query = (select(api_keys.c.id, api_keys.c.expires_at, api_keys.c.revoked_at, accounts.c.id, accounts.c.name,
                        accounts.c.status, accounts.c.rate_rps, accounts.c.burst, accounts.c.monthly_quota,
                        accounts.c.prepaid, topped_up)
                 .join(accounts, api_keys.c.account_id == accounts.c.id).where(api_keys.c.key_hash == h))
        async with self.engine.connect() as conn:
            row = (await conn.execute(query)).first()
            if row is None:
                return None
            key_id, expires_at, revoked_at, account_id, name, status, rate, burst, quota, prepaid, topped = row
            if revoked_at is not None or status != "active" or (expires_at is not None and _utc(expires_at) <= now):
                return None
        if quota is not None and self.meter is not None:
            await self.meter.read_month(account_id)
        return Principal(account_id, key_id, name, rate, burst, quota, bool(prepaid), bool(prepaid) and bool(topped))

    async def load_keys(self) -> None:
        """Load the hashes of the keys not revoked."""
        async with self.engine.connect() as conn:
            self._hashes = set((await conn.execute(select(api_keys.c.key_hash).where(
                api_keys.c.revoked_at.is_(None)))).scalars())

    async def run(self, every_s: float = KEYS_RELOAD_S) -> None:
        while True:
            await asyncio.sleep(every_s)
            try:
                await self.load_keys()
            except Exception as exc:
                log.warning("keys not reloaded (keeping %s): %s: %s",
                            "none" if self._hashes is None else len(self._hashes), type(exc).__name__, exc)

    # ---- administration ----

    async def create_account(self, name: str, *, rate_rps: float, burst: int, monthly_quota: int | None = None,
                             prepaid: bool = False) -> int:
        """Raises Conflict if the name is taken."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(insert(accounts).values(name=name, rate_rps=rate_rps, burst=burst,
                                                                    monthly_quota=monthly_quota, prepaid=prepaid))
                return int(result.inserted_primary_key[0])
        except IntegrityError:
            raise Conflict(f"account {name!r} exists") from None

    async def account(self, name: str) -> dict[str, Any] | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(accounts).where(accounts.c.name == name))).first()
        return None if row is None else dict(row._mapping)

    async def set_account(self, name: str, **values: Any) -> bool:
        """Change an account's status (active | suspended), rate_rps, burst, monthly_quota or prepaid."""
        allowed = {"status", "rate_rps", "burst", "monthly_quota", "prepaid"}
        if set(values) - allowed:
            raise ValueError(f"unknown account settings: {sorted(set(values) - allowed)}")
        async with self.engine.begin() as conn:
            return (await conn.execute(update(accounts).where(accounts.c.name == name).values(**values))).rowcount == 1

    async def create_key(self, account: str, name: str, expires_at: dt.datetime | None = None) -> str:
        """A new key for ``account``; the key itself is returned only here."""
        key = new_key()
        async with self.engine.begin() as conn:
            account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account))).scalar()
            if account_id is None:
                raise LookupError(f"no account {account!r}")
            await conn.execute(insert(api_keys).values(account_id=account_id, key_hash=key_hash(key),
                                                       prefix=key[:PREFIX_CHARS], name=name, expires_at=expires_at))
        if self._hashes is not None:
            self._hashes.add(key_hash(key))
        return key

    async def revoke_key(self, account: str, prefix: str) -> int:
        """Revoke the account's key with this prefix; -> 1, or 0 if there is none (or it is revoked already)."""
        async with self.engine.begin() as conn:
            account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account))).scalar()
            return (await conn.execute(update(api_keys).where(
                api_keys.c.account_id == account_id, api_keys.c.prefix == prefix, api_keys.c.revoked_at.is_(None))
                .values(revoked_at=dt.datetime.now(dt.UTC)))).rowcount

    async def list_accounts(self) -> list[dict[str, Any]]:
        async with self.engine.connect() as conn:
            rows = await conn.execute(select(accounts).order_by(accounts.c.name))
            return [dict(r._mapping) for r in rows]

    async def list_keys(self, account: str) -> list[dict[str, Any]]:
        query = (select(api_keys.c.prefix, api_keys.c.name, api_keys.c.created_at, api_keys.c.expires_at,
                        api_keys.c.revoked_at).join(accounts, api_keys.c.account_id == accounts.c.id)
                 .where(accounts.c.name == account).order_by(api_keys.c.created_at))
        async with self.engine.connect() as conn:
            return [dict(r._mapping) for r in await conn.execute(query)]

    async def usage(self, account: str, since: dt.date) -> list[dict[str, Any]]:
        query = (select(usage_daily.c.day, usage_daily.c.model_id, func.sum(usage_daily.c.requests).label("requests"),
                        func.sum(usage_daily.c.failed).label("failed"),
                        func.sum(usage_daily.c.input_tokens).label("input_tokens"),
                        func.sum(usage_daily.c.cost).label("cost"))
                 .join(accounts, usage_daily.c.account_id == accounts.c.id)
                 .where(accounts.c.name == account, usage_daily.c.day >= since)
                 .group_by(usage_daily.c.day, usage_daily.c.model_id).order_by(usage_daily.c.day))
        async with self.engine.connect() as conn:
            return [dict(r._mapping) for r in await conn.execute(query)]

    async def account_series(self, account: str, days: int | None = None, bucket: str = "day",
                             now: dt.datetime | None = None) -> dict[str, Any] | None:
        """One account's usage over time, per UTC day or week (Monday to Sunday), over the buckets ``series`` gives
        (all time: from its first usage): per model, its answered and failed requests, input tokens and cost
        (micro-USD). None for an unknown account. Raises ValueError for an unknown bucket or more buckets than
        ``SERIES_MAX_BUCKETS`` (``series_starts``)."""
        now = now or dt.datetime.now(dt.UTC)
        usage = usage_daily.c
        kinds = ("requests", "failed", "input_tokens", "cost")
        async with self.engine.connect() as conn:
            account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account))).scalar()
            if account_id is None:
                return None
            first = (await conn.execute(select(func.min(usage.day)).where(usage.account_id == account_id))).scalar()
            first = None if first is None else _date(first)
            starts = series_starts(days, bucket, now.date(), first)
            rows = [(model, _date(day), [int(n or 0) for n in counts]) for model, day, *counts in await conn.execute(
                select(usage.model_id, usage.day, *(func.sum(getattr(usage, kind)) for kind in kinds))
                .where(usage.account_id == account_id, usage.day >= starts[0])
                .group_by(usage.model_id, usage.day))]
        models = sorted({model for model, _, _ in rows})
        buckets = {s: {"start": s.isoformat(), **{kind: dict.fromkeys(models, 0) for kind in kinds}} for s in starts}
        for model, day, counts in rows:
            if (b := buckets.get(bucket_start(day, bucket))) is not None:
                for kind, n in zip(kinds, counts):
                    b[kind][model] += n
        return {"as_of": now.isoformat(timespec="seconds"), "bucket": bucket, "since": starts[0].isoformat(),
                "first_day": first and first.isoformat(), "models": models, "buckets": list(buckets.values())}

    async def totals(self, now: dt.datetime | None = None) -> dict[str, Any]:
        """The network's totals: answered requests and input tokens of every account, in all and per model, all time,
        in the last 24 hours (the current UTC hour and the 23 before) and the last 30 days (today and the 29 UTC days
        before); paid usage, top-ups and the accounts that topped up, all time and in the last 30 days. In micro-USD;
        adjustments are left out."""
        now = now or dt.datetime.now(dt.UTC)
        day = now.replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=23)
        month = dt.datetime.combine(now.date() - dt.timedelta(days=29), dt.time(), dt.UTC)

        def since(start: Any, when: Any, column: Any) -> Any:
            return func.sum(case((when >= start, column), else_=0))
        hourly = usage_hourly.c
        answered = select(hourly.model_id, *(column for c in (hourly.requests, hourly.input_tokens) for column in (
            func.sum(c), since(day, hourly.hour, c), since(month, hourly.hour, c)))).group_by(hourly.model_id)
        topped = select(credits.c.account_id).where(credits.c.kind == "topup")
        cost = (select(usage_daily.c.account_id, func.sum(usage_daily.c.cost),
                       since(month.date(), usage_daily.c.day, usage_daily.c.cost))
                .where(usage_daily.c.account_id.in_(topped)).group_by(usage_daily.c.account_id))
        credit = (select(credits.c.account_id, credits.c.kind, func.sum(credits.c.amount),
                         since(month, credits.c.created_at, credits.c.amount))
                  .where(credits.c.account_id.in_(topped), credits.c.kind.in_(("grant", *BOUGHT)))
                  .group_by(credits.c.account_id, credits.c.kind))
        async with self.engine.connect() as conn:
            per_model = {model: [int(n or 0) for n in counts] for model, *counts in await conn.execute(answered)}
            costs = {a: (int(c or 0), int(c or 0) - int(recent or 0)) for a, c, recent in await conn.execute(cost)}
            ledger: dict[tuple[int, str], tuple[int, int]] = {}  # (account, grant | topup) -> (all, before 30 days)
            for a, kind, total, recent in await conn.execute(credit):
                key = (a, "grant" if kind == "grant" else "topup")
                so_far = ledger.get(key, (0, 0))
                ledger[key] = (so_far[0] + int(total or 0), so_far[1] + int(total or 0) - int(recent or 0))

        def windows(counts: list[int]) -> dict[str, int]:
            return dict(zip(("all_time", "last_24h", "last_30d"), counts))
        models = {m: {"requests": windows(c[:3]), "input_tokens": windows(c[3:])} for m, c in sorted(per_model.items())}
        money = {"paid_usage_micros": [0, 0], "topups_micros": [0, 0], "paying_accounts": [0, 0]}  # all, 30 days
        for account in {a for a, kind in ledger if kind == "topup"}:
            cost_all, cost_before = costs.get(account, (0, 0))
            grants, grants_before = ledger.get((account, "grant"), (0, 0))
            bought, bought_before = ledger[(account, "topup")]
            paid = paid_cost(cost_all, grants, bought)
            for what, all_time, recent in (
                    ("paid_usage_micros", paid, max(0, paid - paid_cost(cost_before, grants_before, bought_before))),
                    ("topups_micros", bought, bought - bought_before),
                    ("paying_accounts", 1, int(bought > bought_before))):
                money[what][0] += all_time
                money[what][1] += recent
        return {"as_of": now.isoformat(timespec="seconds"),
                "since": {"last_24h": day.isoformat(), "last_30d": month.isoformat()},
                **{what: {w: sum(m[what][w] for m in models.values()) for w in ("all_time", "last_24h", "last_30d")}
                   for what in ("requests", "input_tokens")},
                "models": models,
                **{what: {"all_time": pair[0], "last_30d": pair[1]} for what, pair in money.items()}}

    async def series(self, days: int | None = None, bucket: str = "day", now: dt.datetime | None = None
                     ) -> dict[str, Any]:
        """The network's usage over time, per UTC day or week (Monday to Sunday): every account's answered requests and
        input tokens per model, paid usage and top-ups (micro-USD). The buckets that hold the last ``days`` days,
        whole, up to the current one; without ``days``, from the first holding usage or a top-up, at most
        ``SERIES_MAX_BUCKETS`` (the latest). Raises ValueError for an unknown bucket or more buckets than that
        (``series_starts``).

        Paid usage is allocated to days per account, as the totals count it (``paid_cost``): of its usage cost in day
        order, the first part, as much as its grants, is free, and the next, as much as its top-ups, is paid on the
        days it was spent. The days add up to the totals' paid usage; a grant recorded after paid spending makes the
        earliest of it free."""
        now = now or dt.datetime.now(dt.UTC)
        utc_day = self._utc_day
        topups = credits.c.kind == "topup"
        async with self.engine.connect() as conn:
            first = min((_utc(t).astimezone(dt.UTC).date() for t in (
                (await conn.execute(select(func.min(usage_hourly.c.hour)))).scalar(),
                (await conn.execute(select(func.min(credits.c.created_at)).where(topups))).scalar())
                if t is not None), default=None)
            starts = series_starts(days, bucket, now.date(), first)
            start = starts[0]
            since = dt.datetime.combine(start, dt.time(), dt.UTC)
            answered_on = utc_day(usage_hourly.c.hour)
            answered = await conn.execute(
                select(usage_hourly.c.model_id, answered_on, func.sum(usage_hourly.c.requests),
                       func.sum(usage_hourly.c.input_tokens))
                .where(usage_hourly.c.hour >= since).group_by(usage_hourly.c.model_id, answered_on))
            per_model = [(model, _date(day), int(n or 0), int(t or 0)) for model, day, n, t in answered]
            credited = utc_day(credits.c.created_at)
            bought = [(_date(day), int(amount or 0)) for day, amount in await conn.execute(
                select(credited, func.sum(credits.c.amount))
                .where(credits.c.kind.in_(BOUGHT), credits.c.created_at >= since).group_by(credited))]
            topped = select(credits.c.account_id).where(topups)
            ledger: dict[int, dict[str, int]] = {}
            for account, kind, amount in await conn.execute(
                    select(credits.c.account_id, credits.c.kind, func.sum(credits.c.amount))
                    .where(credits.c.account_id.in_(topped), credits.c.kind.in_(("grant", *BOUGHT)))
                    .group_by(credits.c.account_id, credits.c.kind)):
                kinds = ledger.setdefault(account, {})
                kind = "grant" if kind == "grant" else "topup"
                kinds[kind] = kinds.get(kind, 0) + int(amount or 0)
            spent = {account: int(cost or 0) for account, cost in await conn.execute(
                select(usage_daily.c.account_id, func.sum(usage_daily.c.cost))
                .where(usage_daily.c.account_id.in_(topped), usage_daily.c.day < start)
                .group_by(usage_daily.c.account_id))}
            costs = await conn.execute(
                select(usage_daily.c.account_id, usage_daily.c.day, func.sum(usage_daily.c.cost))
                .where(usage_daily.c.account_id.in_(topped), usage_daily.c.day >= start)
                .group_by(usage_daily.c.account_id, usage_daily.c.day)
                .order_by(usage_daily.c.account_id, usage_daily.c.day))
            paid: list[tuple[dt.date, int]] = []
            for account, day, cost in costs:
                free = ledger.get(account, {}).get("grant", 0)
                limit = free + ledger.get(account, {}).get("topup", 0)
                before = spent.get(account, 0)
                spent[account] = after = before + int(cost or 0)
                paid.append((_date(day), max(0, min(after, limit) - max(before, free))))

        models = sorted({model for model, *_ in per_model})
        buckets = {s: {"start": s.isoformat(), "requests": dict.fromkeys(models, 0),
                       "input_tokens": dict.fromkeys(models, 0), "paid_usage_micros": 0, "topups_micros": 0}
                   for s in starts}

        def at(day: dt.date) -> dict[str, Any] | None:
            return buckets.get(bucket_start(day, bucket))
        for model, day, requests, tokens in per_model:
            if (b := at(day)) is not None:
                b["requests"][model] += requests
                b["input_tokens"][model] += tokens
        for what, rows in (("topups_micros", bought), ("paid_usage_micros", paid)):
            for day, amount in rows:
                if (b := at(day)) is not None:
                    b[what] += amount
        return {"as_of": now.isoformat(timespec="seconds"), "bucket": bucket, "since": start.isoformat(),
                "first_day": first and first.isoformat(), "models": models, "buckets": list(buckets.values())}

    def _utc_day(self, column: Any) -> Any:
        """The UTC day of a timestamp column, in SQL (PostgreSQL would take the session's time zone)."""
        if self.engine.dialect.name == "postgresql":
            return func.date(func.timezone(literal_column("'UTC'"), column), type_=Date)
        return func.date(column, type_=Date)

    async def add_credit(self, account: str, amount: int, *, kind: str, reference: str, note: str | None = None,
                         refund_of: str | None = None) -> tuple[dict[str, Any], bool]:
        """Record ``amount`` micro-USD for ``account`` once per ``reference``. -> (the entry, whether it is new).
        Recording the same entry again returns it unchanged; a reference used for another entry raises Conflict, an
        unknown account LookupError. A refund (negative) names the account's top-up it returns (``refund_of``) and
        returns at most what earlier refunds left of it; ValueError otherwise."""
        if kind not in CREDIT_KINDS:
            raise ValueError(f"kind is one of {CREDIT_KINDS}")
        if amount == 0 or (amount < 0 and kind in ("grant", "topup")) or (amount > 0 and kind == "refund"):
            raise ValueError("a grant or top-up is positive, a refund negative; an adjustment either")
        if (kind == "refund") != (refund_of is not None):
            raise ValueError("a refund, and only a refund, names the top-up it returns (refund_of)")
        try:
            async with self.engine.begin() as conn:
                account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account)
                                                 .with_for_update())).scalar()  # one refund of it at a time
                if account_id is None:
                    raise LookupError(f"no account {account!r}")
                if refund_of is not None and not (await conn.execute(select(exists().where(
                        credits.c.reference == reference)))).scalar():  # recorded already: returned as it is
                    await _refundable(conn, account_id, refund_of, -amount)
                await conn.execute(insert(credits).values(account_id=account_id, amount=amount, kind=kind,
                                                          reference=reference, note=note, refund_of=refund_of))
                await conn.execute(update(accounts).where(accounts.c.id == account_id)
                                   .values(balance=accounts.c.balance + amount))
            created = True
        except IntegrityError:
            created = False
        entry = await self.credit(reference)
        if entry is None:  # the insert failed for another reason than the reference
            raise Conflict(f"credit {reference!r} could not be recorded")
        if not created and (entry["account"], entry["amount"], entry["kind"], entry["refund_of"]) != (
                account, amount, kind, refund_of):
            raise Conflict(f"reference {reference!r} records another credit")
        if self.meter is not None:
            self.meter.forget_balance(entry["account_id"])
        return entry, created

    async def credit(self, reference: str) -> dict[str, Any] | None:
        """The credit entry recorded under ``reference``, with its account's name, if any."""
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(credits, accounts.c.name.label("account"))
                                      .join(accounts, credits.c.account_id == accounts.c.id)
                                      .where(credits.c.reference == reference))).first()
        return None if row is None else dict(row._mapping)

    async def credits(self, account: str, limit: int = 100) -> list[dict[str, Any]]:
        """The account's credit entries, newest first."""
        query = (select(credits.c.id, credits.c.amount, credits.c.kind, credits.c.reference, credits.c.note,
                        credits.c.created_at, credits.c.refund_of).join(accounts, credits.c.account_id == accounts.c.id)
                 .where(accounts.c.name == account).order_by(credits.c.id.desc()).limit(limit))
        async with self.engine.connect() as conn:
            return [dict(r._mapping) for r in await conn.execute(query)]


async def _refundable(conn: Any, account_id: int, topup: str, amount: int) -> None:
    """Raises ValueError unless ``topup`` is a top-up of the account of which earlier refunds left ``amount``."""
    paid = (await conn.execute(select(credits.c.amount).where(
        credits.c.reference == topup, credits.c.account_id == account_id, credits.c.kind == "topup"))).scalar()
    if paid is None:
        raise ValueError(f"refund_of {topup!r} names no top-up of the account")
    returned = -int((await conn.execute(select(func.coalesce(func.sum(credits.c.amount), 0)).where(
        credits.c.kind == "refund", credits.c.refund_of == topup))).scalar())
    if amount > paid - returned:
        raise ValueError(f"top-up {topup!r} has {paid - returned} micro-USD left to refund, not {amount}")


def _date(value: dt.date | str) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


def _utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; they are UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
