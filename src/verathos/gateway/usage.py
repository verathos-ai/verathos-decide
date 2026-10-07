"""Usage per account, key, model and UTC day: counted in memory, written every ``flush_s`` in one statement.

A failed write keeps the counts for the next one, up to ``MAX_PENDING`` rows; nothing is written per request. For
monthly quotas the meter also keeps each account's answered requests this month: as read from the database (which
includes other gateway processes' written counts), plus what this process wrote since and has not written yet. Reads
and writes take turns, so a count is never in both or neither.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.gateway import metrics
from verathos.gateway.db import usage_daily

if TYPE_CHECKING:
    from verathos.gateway.accounts import Principal

log = logging.getLogger(__name__)
FLUSH_S = 5.0
MAX_PENDING = 100_000
Key = tuple[int, int, str, dt.date]  # account, key, model, day


def month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


class UsageMeter:
    def __init__(self, engine: AsyncEngine, *, flush_s: float = FLUSH_S) -> None:
        self.engine, self.flush_s = engine, flush_s
        self._counts: dict[Key, list[int]] = {}  # -> [requests, failed, input tokens]
        self._month: dict[int, list] = {}  # account -> [month start, answered as read or written]
        self._unwritten: dict[int, int] = {}  # account -> answered, not written yet
        self._lock = asyncio.Lock()

    def record(self, principal: Principal, model_id: str, *, answered: bool, input_tokens: int = 0) -> None:
        key = (principal.account_id, principal.key_id, model_id, dt.datetime.now(dt.UTC).date())
        counts = self._counts.get(key)
        if counts is None:
            counts = self._counts[key] = [0, 0, 0]
        counts[0 if answered else 1] += 1
        counts[2] += input_tokens
        if answered:
            self._unwritten[principal.account_id] = self._unwritten.get(principal.account_id, 0) + 1

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
        if not counts:
            return
        rows = [{"account_id": a, "key_id": k, "model_id": m, "day": d, "requests": c[0], "failed": c[1],
                 "input_tokens": c[2]} for (a, k, m, d), c in counts.items()]
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        statement = insert(usage_daily)
        statement = statement.on_conflict_do_update(
            index_elements=["account_id", "key_id", "model_id", "day"],
            set_={c: getattr(usage_daily.c, c) + getattr(statement.excluded, c)
                  for c in ("requests", "failed", "input_tokens")})
        try:
            async with self.engine.begin() as conn:
                await conn.execute(statement, rows)
            for (account, _, _, day), c in counts.items():
                state = self._month.get(account)
                if state is not None and month_start(day) == state[0]:
                    state[1] += c[0]
        except Exception:
            for key, c in counts.items():  # keep them for the next write
                if len(self._counts) >= MAX_PENDING:
                    break
                mine = self._counts.setdefault(key, [0, 0, 0])
                for i in range(3):
                    mine[i] += c[i]
            for account, n in unwritten.items():
                self._unwritten[account] = self._unwritten.get(account, 0) + n
            metrics.USAGE_FLUSH_FAILURES.inc()
            raise

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.flush_s)
            try:
                await self.flush()
            except Exception:
                log.exception("writing usage failed; kept for the next write")
