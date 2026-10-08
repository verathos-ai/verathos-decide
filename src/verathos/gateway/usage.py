"""Usage per account, key, model and UTC day: counted in memory, written every ``flush_s`` in one transaction, together
with the answered requests and input tokens per model and UTC hour of all accounts (``usage_hourly``: the network's
totals).

What was counted since the last write is taken as a batch with an id of its own, written in one transaction that also
records the id (``usage_flushes``). A batch whose write failed waits and is written again as it is, before anything
counted since: if its first write did reach the database (its confirmation was lost, or it was cancelled at shutdown),
the id shows it and it is not counted twice. A batch with rows the database refuses (a key or account deleted by hand)
is written row by row (its hours apart), and a row refused on its own is dropped with an error in the log. Counts and
batches waiting are bounded by ``MAX_PENDING`` rows; counts beyond are dropped and logged. Nothing is written per
request.

For monthly quotas the meter also keeps each account's answered requests this month: as read from the database (which
includes other gateway processes' written counts), plus what this process counted since, waiting or being written. A
read that overlaps a write is read again, so a count is never in both or neither; reads do not wait for each other.

An answer costs its input tokens times the model's price (USD per million tokens, so micro-USD per token). The cost
is written with the usage in whole micro-USD per row, the fraction carried to the row's next write (so answers cost
what their tokens do, however they fall into writes), and taken from the account's balance in the same transaction.
A prepaid account is served while its balance is positive: as read from the database at most ``BALANCE_TTL_S`` ago
(so other processes' charges and new credit apply within that time), minus what this process charged since and what
its requests in flight may cost at most (``reserve``, settled when they end). A balance can therefore end below zero
by at most one request's cost per process.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.gateway import metrics
from verathos.gateway.db import accounts, usage_daily, usage_flushes, usage_hourly

if TYPE_CHECKING:
    from verathos.gateway.accounts import Principal

log = logging.getLogger(__name__)
FLUSH_S = 5.0
MAX_PENDING = 100_000
BALANCE_TTL_S = 5.0
READ_RETRY_S = 0.01  # a read waits this long for a write in progress, then tries again
FLUSH_IDS_KEPT = dt.timedelta(days=7)  # batch ids older than this are deleted; a batch is retried within minutes
Key = tuple[int, int, str, dt.date]  # account, key, model, day
Hour = tuple[str, dt.datetime]  # model, the hour's start


def month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


@dataclass
class Batch:
    """Usage rows written together (cost in whole micro-USD) with the hours' totals, and what they add per account."""

    rows: list[dict[str, Any]]
    hours: list[dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: secrets.token_hex(16))
    answered: dict[int, int] = field(default_factory=dict)
    charged: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for row in self.rows:
            self.answered[row["account_id"]] = self.answered.get(row["account_id"], 0) + row["requests"]
            self.charged[row["account_id"]] = self.charged.get(row["account_id"], 0) + row["cost"]


class UsageMeter:
    def __init__(self, engine: AsyncEngine, *, flush_s: float = FLUSH_S, balance_ttl_s: float = BALANCE_TTL_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.engine, self.flush_s, self.balance_ttl_s, self.clock = engine, flush_s, balance_ttl_s, clock
        self._counts: dict[Key, list] = {}  # -> [requests, failed, input tokens, cost in micro-USD]
        self._hours: dict[Hour, list] = {}  # -> [answered, input tokens], all accounts
        self._month: dict[int, list] = {}  # account -> [month start, answered as read or written]
        self._unwritten: dict[int, int] = {}  # account -> answered, counted since the last batch
        self._balance: dict[int, list] = {}  # prepaid account -> [balance as read or written, when read]
        self._uncharged: dict[int, float] = {}  # account -> cost counted since the last batch
        self._reserved: dict[int, float] = {}  # prepaid account -> the most its requests in flight may cost
        self._fractions: dict[Key, float] = {}  # row -> the fraction of a micro-USD of cost not written yet
        self._waiting: list[Batch] = []  # batches not written yet, oldest first
        self._reading: dict[int, asyncio.Future[None]] = {}  # account -> its balance read in progress
        self._writes = 0  # writes started
        self._writing = False
        self._dropped = 0  # counts dropped beyond MAX_PENDING since the last write
        self._lock = asyncio.Lock()  # one write at a time

    def record(self, principal: Principal, model_id: str, *, answered: bool, input_tokens: int = 0,
               price: float = 0.0) -> None:
        """``price``: the model's, in USD per million input tokens."""
        now = dt.datetime.now(dt.UTC)
        key = (principal.account_id, principal.key_id, model_id, now.date())
        counts = self._counts.get(key)
        if counts is None:
            if self.pending() >= MAX_PENDING:
                self._dropped += 1
                return
            counts = self._counts[key] = [0, 0, 0, 0.0]
        counts[0 if answered else 1] += 1
        counts[2] += input_tokens
        if answered:
            hour = self._hours.setdefault((model_id, now.replace(minute=0, second=0, microsecond=0)), [0, 0])
            hour[0] += 1
            hour[1] += input_tokens
            cost = input_tokens * price
            counts[3] += cost
            self._unwritten[principal.account_id] = self._unwritten.get(principal.account_id, 0) + 1
            self._uncharged[principal.account_id] = self._uncharged.get(principal.account_id, 0.0) + cost

    def pending(self) -> int:
        """Usage rows not written yet."""
        return len(self._counts) + sum(len(b.rows) for b in self._waiting)

    def within_quota(self, principal: Principal) -> bool:
        if principal.monthly_quota is None:
            return True
        month = month_start(dt.datetime.now(dt.UTC).date())
        state = self._month.get(principal.account_id)
        if state is None or state[0] != month:
            return True  # not read yet this month: the next key lookup reads it
        waiting = sum(b.answered.get(principal.account_id, 0) for b in self._waiting)
        return state[1] + self._unwritten.get(principal.account_id, 0) + waiting < principal.monthly_quota

    async def has_credit(self, principal: Principal) -> bool:
        """Whether the account may be served: always, unless it is prepaid and its balance, less what is charged and
        reserved, is used up."""
        return not principal.prepaid or await self._available(principal.account_id) > 0

    async def reserve(self, principal: Principal, cost: float) -> bool:
        """``has_credit``, and if so ``cost`` (the most the request may cost) is reserved until ``release``."""
        if not principal.prepaid:
            return True
        if await self._available(principal.account_id) <= 0:  # nothing awaited from here on: check and reserve at once
            return False
        self._reserved[principal.account_id] = self._reserved.get(principal.account_id, 0.0) + cost
        return True

    def release(self, principal: Principal, cost: float) -> None:
        """End a reservation; the request's real cost, if any, is ``record``ed first."""
        if principal.prepaid:
            left = self._reserved.get(principal.account_id, 0.0) - cost
            if left > 1e-9:
                self._reserved[principal.account_id] = left
            else:
                self._reserved.pop(principal.account_id, None)

    async def _available(self, account_id: int) -> float:
        state = self._balance.get(account_id)
        if state is None or self.clock() - state[1] >= self.balance_ttl_s:
            await self.read_balance(account_id)
            state = self._balance[account_id]
        waiting = sum(b.charged.get(account_id, 0) for b in self._waiting)
        return state[0] - self._uncharged.get(account_id, 0.0) - waiting - self._reserved.get(account_id, 0.0)

    async def read_balance(self, account_id: int) -> None:
        """Read the account's balance; concurrent requests of the account share one read."""
        reading = self._reading.get(account_id)
        if reading is None:
            reading = self._reading[account_id] = asyncio.ensure_future(self._read_balance(account_id))
            reading.add_done_callback(lambda done: self._read_done(account_id, done))
        await asyncio.shield(reading)

    async def _read_balance(self, account_id: int) -> None:
        balance = await self._read(select(accounts.c.balance).where(accounts.c.id == account_id))
        self._balance[account_id] = [int(balance or 0), self.clock()]

    def _read_done(self, account_id: int, done: asyncio.Future[None]) -> None:
        del self._reading[account_id]
        if not done.cancelled():
            done.exception()  # retrieved: the waiters, if any, get it raised

    def forget_balance(self, account_id: int) -> None:
        """The account's balance changed in the database: read it again at its next request."""
        self._balance.pop(account_id, None)

    async def read_month(self, account_id: int) -> None:
        month = month_start(dt.datetime.now(dt.UTC).date())
        used = await self._read(select(func.coalesce(func.sum(usage_daily.c.requests), 0)).where(
            usage_daily.c.account_id == account_id, usage_daily.c.day >= month))
        self._month[account_id] = [month, int(used)]

    async def _read(self, query: Any) -> Any:
        """The query's scalar, read again until no write overlapped the read: a write's counts would otherwise be in
        the result and in the meter both, or in neither."""
        while True:
            if self._writing:
                await asyncio.sleep(READ_RETRY_S)
                continue
            writes = self._writes
            async with self.engine.connect() as conn:
                value = (await conn.execute(query)).scalar()
            if not self._writing and self._writes == writes:
                return value

    async def flush(self) -> None:
        """Write the batches waiting, then what was counted since. Raises if a write fails; what is left waits for
        the next."""
        async with self._lock:
            await self._write_waiting()
            batch = self._take()
            if batch is not None:
                self._waiting.append(batch)
                await self._write_waiting()

    def _take(self) -> Batch | None:
        if self._dropped:
            log.error("%d usage counts dropped: %d rows wait to be written already", self._dropped, self.pending())
            self._dropped = 0
        counts, self._counts = self._counts, {}
        hours, self._hours = self._hours, {}
        self._unwritten, self._uncharged = {}, {}
        if not counts:
            return None
        rows = []
        for key, c in sorted(counts.items()):  # one order in every process: two writing the same rows never deadlock
            cost = c[3] + self._fractions.pop(key, 0.0)
            whole = math.floor(cost + 1e-9)  # 350 x 0.04 is 14, not 13 and a fraction of 0.99...
            if cost - whole > 1e-9:
                self._fractions[key] = cost - whole
            account, key_id, model, day = key
            rows.append({"account_id": account, "key_id": key_id, "model_id": model, "day": day, "requests": c[0],
                         "failed": c[1], "input_tokens": c[2], "cost": whole})
        today = dt.datetime.now(dt.UTC).date()
        self._fractions = {k: f for k, f in self._fractions.items() if k[3] >= today - dt.timedelta(days=1)}
        return Batch(rows, [{"model_id": model, "hour": hour, "requests": n[0], "input_tokens": n[1]}
                            for (model, hour), n in sorted(hours.items())])

    async def _write_waiting(self) -> None:
        while self._waiting:
            batch = self._waiting[0]
            self._writes += 1
            self._writing = True
            try:
                await self._commit(batch)
            except IntegrityError:
                if len(batch.rows) + bool(batch.hours) > 1:  # written row by row, so only the refused rows are lost
                    hours = [Batch([], batch.hours)] if batch.hours else []
                    self._waiting[:1] = hours + [Batch([row]) for row in batch.rows]
                else:
                    log.error("a usage row the database refuses is dropped: %s", (batch.rows or batch.hours)[0])
                    self._waiting.pop(0)
                continue
            except BaseException:
                metrics.USAGE_FLUSH_FAILURES.inc()
                raise
            finally:
                self._writing = False
            self._waiting.pop(0)
            for row in batch.rows:
                state = self._month.get(row["account_id"])
                if state is not None and month_start(row["day"]) == state[0]:
                    state[1] += row["requests"]
            for account, cost in batch.charged.items():
                state = self._balance.get(account)
                if state is not None:
                    state[0] -= cost

    async def _commit(self, batch: Batch) -> None:
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert

        def adding(table, keys: list[str], counts: tuple[str, ...]):
            statement = insert(table)
            return statement.on_conflict_do_update(
                index_elements=keys, set_={c: getattr(table.c, c) + getattr(statement.excluded, c) for c in counts})
        now = dt.datetime.now(dt.UTC)
        async with self.engine.begin() as conn:
            if (await conn.execute(select(usage_flushes.c.id).where(usage_flushes.c.id == batch.id))).first():
                return  # written already: its first write's confirmation was lost
            await conn.execute(insert(usage_flushes).values(id=batch.id, written_at=now))
            if batch.rows:
                await conn.execute(adding(usage_daily, ["account_id", "key_id", "model_id", "day"],
                                          ("requests", "failed", "input_tokens", "cost")), batch.rows)
            if batch.hours:
                await conn.execute(adding(usage_hourly, ["model_id", "hour"], ("requests", "input_tokens")),
                                   batch.hours)
            for account, cost in sorted(batch.charged.items()):  # one order in every process: no deadlocks
                if cost:
                    await conn.execute(update(accounts).where(accounts.c.id == account)
                                       .values(balance=accounts.c.balance - cost))
            await conn.execute(delete(usage_flushes).where(usage_flushes.c.written_at < now - FLUSH_IDS_KEPT))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.flush_s)
            try:
                await self.flush()
            except Exception:
                log.exception("writing usage failed; kept for the next write")
