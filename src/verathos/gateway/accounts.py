"""Accounts and their API keys.

A key is ``vd_`` followed by 64 hex characters (32 random bytes). Only its SHA-256 is stored, with its first
characters as a prefix to recognise it. A lookup is cached per key for ``CACHE_TTL_S`` (a revoked key or a changed
limit applies within that time), an unknown key for ``NEGATIVE_TTL_S``; concurrent lookups of one key share one
query. For an account with a monthly quota, a lookup also has the usage meter read the account's month.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.gateway.db import accounts, api_keys, usage_daily
from verathos.gateway.usage import UsageMeter

KEY_PREFIX = "vd_"
_KEY = re.compile(r"vd_[0-9a-f]{64}")
CACHE_TTL_S = 60.0
NEGATIVE_TTL_S = 10.0
MAX_CACHED = 100_000


@dataclass(frozen=True)
class Principal:
    account_id: int
    key_id: int
    account: str
    rate_rps: float
    burst: int
    monthly_quota: int | None


def new_key() -> str:
    return KEY_PREFIX + secrets.token_hex(32)


def key_hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Accounts:
    def __init__(self, engine: AsyncEngine, meter: UsageMeter | None = None, *, ttl_s: float = CACHE_TTL_S,
                 negative_ttl_s: float = NEGATIVE_TTL_S, clock: Callable[[], float] = time.monotonic) -> None:
        self.engine, self.meter, self.ttl_s, self.negative_ttl_s, self.clock = (engine, meter, ttl_s,
                                                                                negative_ttl_s, clock)
        self._cache: dict[str, tuple[float, Principal | None]] = {}
        self._loading: dict[str, asyncio.Future[Principal | None]] = {}

    # ---- the request path ----

    async def authenticate(self, key: str) -> Principal | None:
        """The key's principal, or None if the key is unknown, revoked, expired or its account suspended. Raises on
        database errors."""
        if not _KEY.fullmatch(key):
            return None
        h, now = key_hash(key), self.clock()
        cached = self._cache.get(h)
        if cached is not None and cached[0] > now:
            return cached[1]
        pending = self._loading.get(h)
        if pending is not None:
            return await asyncio.shield(pending)
        future = asyncio.get_running_loop().create_future()
        self._loading[h] = future
        try:
            principal = await self._load(h)
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # retrieved: the waiters get it raised
            raise
        finally:
            del self._loading[h]
        if len(self._cache) >= MAX_CACHED:
            self._cache = {k: v for k, v in self._cache.items() if v[0] > now}
            if len(self._cache) >= MAX_CACHED:
                self._cache.clear()
        self._cache[h] = (now + (self.ttl_s if principal else self.negative_ttl_s), principal)
        future.set_result(principal)
        return principal

    async def _load(self, h: str) -> Principal | None:
        now = dt.datetime.now(dt.UTC)
        query = (select(api_keys.c.id, api_keys.c.expires_at, api_keys.c.revoked_at, accounts.c.id, accounts.c.name,
                        accounts.c.status, accounts.c.rate_rps, accounts.c.burst, accounts.c.monthly_quota)
                 .join(accounts, api_keys.c.account_id == accounts.c.id).where(api_keys.c.key_hash == h))
        async with self.engine.connect() as conn:
            row = (await conn.execute(query)).first()
            if row is None:
                return None
            key_id, expires_at, revoked_at, account_id, name, status, rate, burst, quota = row
            if revoked_at is not None or status != "active" or (expires_at is not None and _utc(expires_at) <= now):
                return None
        if quota is not None and self.meter is not None:
            await self.meter.read_month(account_id)
        return Principal(account_id, key_id, name, rate, burst, quota)

    # ---- administration ----

    async def create_account(self, name: str, *, rate_rps: float, burst: int, monthly_quota: int | None = None) -> int:
        async with self.engine.begin() as conn:
            result = await conn.execute(insert(accounts).values(name=name, rate_rps=rate_rps, burst=burst,
                                                                monthly_quota=monthly_quota))
            return int(result.inserted_primary_key[0])

    async def set_account(self, name: str, **values: Any) -> bool:
        """Change an account's status (active | suspended), rate_rps, burst or monthly_quota."""
        allowed = {"status", "rate_rps", "burst", "monthly_quota"}
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
                                                       prefix=key[:10], name=name, expires_at=expires_at))
        return key

    async def revoke_key(self, prefix: str) -> int:
        async with self.engine.begin() as conn:
            return (await conn.execute(update(api_keys).where(api_keys.c.prefix == prefix,
                                                              api_keys.c.revoked_at.is_(None))
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
                        func.sum(usage_daily.c.input_tokens).label("input_tokens"))
                 .join(accounts, usage_daily.c.account_id == accounts.c.id)
                 .where(accounts.c.name == account, usage_daily.c.day >= since)
                 .group_by(usage_daily.c.day, usage_daily.c.model_id).order_by(usage_daily.c.day))
        async with self.engine.connect() as conn:
            return [dict(r._mapping) for r in await conn.execute(query)]


def _utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; they are UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
