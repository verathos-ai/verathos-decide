"""Usage per account, key, model and UTC day: counted in memory, written every ``flush_s`` in one statement.

A failed write keeps the counts for the next one, up to ``MAX_PENDING`` rows; nothing is written per request. For
monthly quotas the meter also keeps each account's answered requests this month: as read from the database (which
includes other gateway processes' written counts), plus what this process wrote since and has not written yet. Reads
and writes take turns, so a count is never in both or neither.

An answer costs its input tokens times the model's price (USD per million tokens, so micro-USD per token). The cost
is written with the usage, rounded to whole micro-USD per row, and taken from the account's balance in the same
transaction. A prepaid account is served while its balance is positive: as read from the database at most
``BALANCE_TTL_S`` ago (so other processes' charges and new credit apply within that time), minus what this process
charged since. Answers in flight when it runs out are still charged, so a balance can end slightly below zero.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.gateway import metrics
from verathos.gateway.db import accounts, usage_daily

if TYPE_CHECKING:
    from verathos.gateway.accounts import Principal

log = logging.getLogger(__name__)
FLUSH_S = 5.0
MAX_PENDING = 100_000
BALANCE_TTL_S = 5.0
Key = tuple[int, int, str, dt.date]  # account, key, model, day


def month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


class UsageMeter:
    def __init__(self, engine: AsyncEngine, *, flush_s: float = FLUSH_S, balance_ttl_s: float = BALANCE_TTL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.engine, self.flush_s, self.balance_ttl_s, self.clock = engine, flush_s, balance_ttl_s, clock
        self._counts: dict[Key, list] = {}  # -> [requests, failed, input tokens, cost in micro-USD]
        self._month: dict[int, list] = {}  # account -> [month start, answered as read or written]
        self._unwritten: dict[int, int] = {}  # account -> answered, not written yet
        self._balance: dict[int, list] = {}  # prepaid account -> [balance as read or written, when read]
        self._uncharged: dict[int, float] = {}  # account -> cost not written yet
        self._lock = asyncio.Lock()

    def record(self, principal: Principal, model_id: str, *, answered: bool, input_tokens: int = 0,
               price: float = 0.0) -> None:
        """``price``: the model's, in USD per million input tokens."""
        key = (principal.account_id, principal.key_id, model_id, dt.datetime.now(dt.UTC).date())
        counts = self._counts.get(key)
        if counts is None:
            counts = self._counts[key] = [0, 0, 0, 0.0]
        counts[0 if answered else 1] += 1
        counts[2] += input_tokens
        if answered:
            cost = input_tokens * price
            counts[3] += cost
            self._unwritten[principal.account_id] = self._unwritten.get(principal.account_id, 0) + 1
            self._uncharged[principal.account_id] = self._uncharged.get(principal.account_id, 0.0) + cost

    def pending(self) -> int:
        """Usage rows not written yet."""
        return len(self._counts)

    def within_quota(self, principal: Principal) -> bool:
        if principal.monthly_quota is None:
            return True
        month = month_start(dt.datetime.now(dt.UTC).date())
        state = self._month.get(principal.account_id)
        if state is None or state[0] != month:
            return True  # not read yet this month: the next key lookup reads it
        return state[1] + self._unwritten.get(principal.account_id, 0) < principal.monthly_quota

    async def has_credit(self, principal: Principal) -> bool:
        """Whether the account may be served: always, unless it is prepaid and its balance is used up."""
        if not principal.prepaid:
            return True
        state = self._balance.get(principal.account_id)
        if state is None or self.clock() - state[1] >= self.balance_ttl_s:
            await self.read_balance(principal.account_id)
            state = self._balance[principal.account_id]
        return state[0] - self._uncharged.get(principal.account_id, 0.0) > 0

    async def read_balance(self, account_id: int) -> None:
        async with self._lock:
            state = self._balance.get(account_id)
            if state is not None and self.clock() - state[1] < self.balance_ttl_s:
                return  # read meanwhile by a concurrent request
            async with self.engine.connect() as conn:
                balance = (await conn.execute(select(accounts.c.balance).where(accounts.c.id == account_id))
                           ).scalar()
            self._balance[account_id] = [int(balance or 0), self.clock()]

    def forget_balance(self, account_id: int) -> None:
        """The account's balance changed in the database: read it again at its next request."""
        self._balance.pop(account_id, None)

    async def read_month(self, account_id: int) -> None:
        month = month_start(dt.datetime.now(dt.UTC).date())
        async with self._lock:
            async with self.engine.connect() as conn:
                used = (await conn.execute(select(func.coalesce(func.sum(usage_daily.c.requests), 0)).where(
                    usage_daily.c.account_id == account_id, usage_daily.c.day >= month))).scalar_one()
            self._month[account_id] = [month, int(used)]

    async def flush(self) -> None:
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        counts, self._counts = self._counts, {}
        unwritten, self._unwritten = self._unwritten, {}
        uncharged, self._uncharged = self._uncharged, {}
        if not counts:
            return
        rows = [{"account_id": a, "key_id": k, "model_id": m, "day": d, "requests": c[0], "failed": c[1],
                 "input_tokens": c[2], "cost": round(c[3])} for (a, k, m, d), c in counts.items()]
        charged: dict[int, int] = {}
        for row in rows:
            charged[row["account_id"]] = charged.get(row["account_id"], 0) + row["cost"]
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        statement = insert(usage_daily)
        statement = statement.on_conflict_do_update(
            index_elements=["account_id", "key_id", "model_id", "day"],
            set_={c: getattr(usage_daily.c, c) + getattr(statement.excluded, c)
                  for c in ("requests", "failed", "input_tokens", "cost")})
        try:
            async with self.engine.begin() as conn:
                await conn.execute(statement, rows)
                for account, cost in sorted(charged.items()):  # one order in every process: no deadlocks
                    if cost:
                        await conn.execute(update(accounts).where(accounts.c.id == account)
                                           .values(balance=accounts.c.balance - cost))
            for (account, _, _, day), c in counts.items():
                state = self._month.get(account)
                if state is not None and month_start(day) == state[0]:
                    state[1] += c[0]
            for account, cost in charged.items():
                state = self._balance.get(account)
                if state is not None:
                    state[0] -= cost
        except BaseException:  # kept for the next write, also when the write is cancelled at shutdown
            for key, c in counts.items():
                if len(self._counts) >= MAX_PENDING:
                    break
                mine = self._counts.setdefault(key, [0, 0, 0, 0.0])
                for i in range(4):
                    mine[i] += c[i]
            for account, n in unwritten.items():
                self._unwritten[account] = self._unwritten.get(account, 0) + n
            for account, cost in uncharged.items():
                self._uncharged[account] = self._uncharged.get(account, 0.0) + cost
            metrics.USAGE_FLUSH_FAILURES.inc()
            raise

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.flush_s)
            try:
                await self.flush()
            except Exception:
                log.exception("writing usage failed; kept for the next write")
