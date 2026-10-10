"""Referrals at the gateway: referrers' codes, the accounts they referred, and those accounts' top-ups and refunds as
they count for the catalog's referral pool, reported to the auditor.

Codes (``protocol.referral``). A hotkey has a code by its signed registration, which the account service (the web
app) forwards after checking on chain that the hotkey holds a UID and is not the subnet owner's. The gateway takes it
when the hotkey's signature verifies, it was signed for this netuid and ``referral_site`` within the last 5 minutes,
the code is well formed, no code folds to the same form, read as it looks (``looks``: digits as letters) it
contains no reserved name (``RESERVED_NAMES``, the catalog's model ids and ``referral_reserved``) and is not a reserved
word nor has one as a hyphen-separated part (``RESERVED_WORDS``), and its hotkey holds fewer than ``MAX_CODES`` codes
not revoked. A code is never deleted, so it is never given out twice. A move takes every code of one hotkey to another
of its coldkey (the account service checks the coldkey on chain), signed by the new hotkey, if the new one then holds
at most ``MAX_CODES`` codes not revoked. The operator may revoke a code: it takes no new accounts, and its accounts'
top-ups stop counting.

Attribution. An account has at most one code (``referrals``), set once by the account service, only while the account
has no top-up: its first top-up fixes its referrer.

What counts. A top-up of a referred account counts for its code if it was credited after the referral, within the
pool's ``term_days`` of it and before the code was revoked, up to the pool's ``account_cap_usd`` for the account: what
counts of the account's top-ups credited within a window's length before or after it stays within the cap, whatever
order they are decided in (a refund frees none of it). A refund counts negative, at most what its top-up counted less
earlier refunds of it. A top-up decided while the catalog has no referral pool counts nothing. Each credit is decided
once (``referral_reports``) and reported once.

The reporter. Every ``REPORT_S`` one gateway process (PostgreSQL's advisory lock ``LOCK_KEY``; the others skip the
round) decides the new credits of referred accounts, then posts what the auditor has not taken, signed by the gateway:
registrations and moves in the order they happened, then, once every registration is out, entries
(``protocol.gateway.ReferralReport``). An entry's id is drawn at random when its credit is decided and kept with it,
so it leads back to no credit or payment. The auditor takes every item once, so a post whose answer was lost is sent
again as it is. If it no longer knows a code an entry names (its database started over), every registration and move
goes again, then those entries. An entry counts at most ``MAX_REFERRAL_MICROS``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from bittensor_wallet import Keypair
from sqlalchemy import Date, distinct, exists, func, insert, literal_column, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from verathos.catalog import Catalog, ReferralPool
from verathos.gateway.db import (
    BOUGHT,
    accounts,
    credits,
    referral_codes,
    referral_moves,
    referral_reports,
    referrals,
)
from verathos.protocol.gateway import (
    MAX_REFERRAL_ENTRIES,
    MAX_REFERRAL_MICROS,
    MAX_REFERRAL_SIGNED,
    REFERRALS_PATH,
    ReferralEntry,
    ReferralReport,
)
from verathos.protocol.http import post_idempotent
from verathos.protocol.referral import SignedMove, SignedRegistration, fold, fresh, well_formed
from verathos.protocol.requests import sign_request
from verathos.protocol.tls import Service
from verathos.throughput.policy import BLOCK_TIME_S

log = logging.getLogger(__name__)
MAX_CODES = 3
RESERVED_NAMES = ("verathos", "decide", "bittensor", "opentensor", "taostats")
RESERVED_WORDS = ("official", "support", "admin", "team", "help")
REPORT_S = 30.0
BATCH = 1000  # credits decided per round
LOCK_KEY = 0x7665726174686F73
MAX_DAYS = 400  # of a referrer's figures


class Refused(Exception):
    """What the referral ledger refuses: the HTTP status, the error type and why."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


_LOOKS = str.maketrans("01i34578", "olleastb")  # digits read as the letters they look like; 1, i and l alike


def looks(text: str) -> str:
    """``text`` as it reads: lower case, hyphens dropped, digits as the letters they look like, 1, i and l alike."""
    return text.lower().replace("-", "").translate(_LOOKS)


def reserved(code: str, names: Iterable[str] = ()) -> str | None:
    """The reserved name or word ``code`` holds, if any (see the module), read as it looks (``looks``)."""
    reading = looks(code)
    for name in (*RESERVED_NAMES, *names):
        if looks(name) and looks(name) in reading:
            return name
    words = {looks(word) for word in RESERVED_WORDS}
    return next((part for part in (code, *code.split("-")) if looks(part) in words), None)


@dataclass(frozen=True)
class Decided:
    """A credit of a referred account, decided: what it counts (micro-USD)."""

    kind: str
    reference: str
    refund_of: str | None
    created: dt.datetime
    counted: int


def counted(kind: str, amount: int, refund_of: str | None, created: dt.datetime, referred: dt.datetime,
            revoked: dt.datetime | None, earlier: list[Decided], pool: ReferralPool | None, window_s: float) -> int:
    """What a credit of a referred account counts for its code (see the module), given the account's credits decided
    before it (``earlier``)."""
    if kind == "refund":
        topup = next((e for e in earlier if e.kind == "topup" and e.reference == refund_of), None)
        if topup is None or topup.counted <= 0:
            return 0
        returned = -sum(e.counted for e in earlier if e.kind == "refund" and e.refund_of == refund_of)
        return -min(-amount, max(topup.counted - returned, 0))
    if pool is None or created < referred or (revoked is not None and created >= revoked):
        return 0
    if pool.term_days is not None and created >= referred + dt.timedelta(days=pool.term_days):
        return 0
    if pool.account_cap_usd is None:
        return min(amount, MAX_REFERRAL_MICROS)
    near = sum(e.counted for e in earlier
               if e.kind == "topup" and abs((e.created - created).total_seconds()) < window_s)
    return max(0, min(amount, round(pool.account_cap_usd * 1_000_000) - near, MAX_REFERRAL_MICROS))


class Referrals:
    """The referral ledger in the gateway's database (see the module). ``catalog`` gives the catalog followed: its
    model ids are reserved, and its referral pool decides what counts."""

    def __init__(self, engine: AsyncEngine, *, netuid: int, site: str | None, reserved_names: Iterable[str] = (),
                 catalog: Callable[[], Catalog | None] = lambda: None,
                 clock: Callable[[], float] = time.time) -> None:
        self.engine, self.netuid, self.site, self.catalog, self.clock = engine, netuid, site, catalog, clock
        self.reserved_names = tuple(reserved_names)

    # ---- codes ----

    def _check(self, netuid: int, site: str, issued_ms: int) -> None:
        if self.site is None:
            raise Refused(503, "unavailable", "this gateway takes no referral codes: referral_site is not set")
        if netuid != self.netuid:
            raise Refused(400, "invalid_request", f"signed for netuid {netuid}, not {self.netuid}")
        if site != self.site:
            raise Refused(400, "invalid_request", f"signed for the site {site!r}, not {self.site!r}")
        if not fresh(issued_ms, int(self.clock() * 1000)):
            raise Refused(400, "invalid_request", "signed more than 5 minutes ago, or ahead of this gateway's clock")

    @staticmethod
    async def _lock_hotkeys(conn: Any, *hotkeys: str) -> None:
        """Registrations and moves of these hotkeys one at a time, until the transaction ends (PostgreSQL; SQLite
        writes one at a time anyway)."""
        if conn.dialect.name != "postgresql":
            return
        for hotkey in sorted(set(hotkeys)):  # one order everywhere: no deadlocks
            key = int.from_bytes(hashlib.sha256(f"referral {hotkey}".encode()).digest()[:8], "big", signed=True)
            await conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})

    def names(self) -> list[str]:
        """The names besides ``RESERVED_NAMES`` no code may contain: the catalog's model ids and the configured
        ones."""
        cat = self.catalog()
        return [*(m.model_id for m in cat.models), *self.reserved_names] if cat else list(self.reserved_names)

    async def register(self, signed: SignedRegistration) -> tuple[dict[str, Any], bool]:
        """Take a code's registration (see the module). -> (the code, whether it is new); the same code registered
        again for its hotkey is returned unchanged. Raises Refused."""
        r = signed.registration
        self._check(r.netuid, r.site, r.issued_ms)
        if not signed.verify():
            raise Refused(400, "invalid_request", "the signature does not verify against the registration's hotkey")
        if not well_formed(r.code):
            raise Refused(400, "invalid_request", f"{r.code!r} is not a code: 8 characters of "
                                                  "abcdefghjkmnpqrstuvwxyz23456789, or 4 to 24 of a-z, digits and "
                                                  "single hyphens starting with a letter")
        if name := reserved(r.code, self.names()):
            raise Refused(400, "invalid_request", f"{r.code!r} holds the reserved name {name!r}")
        for last in (False, True):
            try:
                async with self.engine.begin() as conn:
                    await self._lock_hotkeys(conn, r.hotkey)
                    taken = (await conn.execute(select(referral_codes.c.code, referral_codes.c.hotkey).where(
                        referral_codes.c.folded == fold(r.code)))).first()
                    if taken is not None and tuple(taken) == (r.code, r.hotkey):
                        return await self.code(r.code), False
                    if taken is not None:
                        raise Refused(409, "conflict", f"{r.code!r} is taken" if taken.code == r.code else
                                      f"{r.code!r} reads as the code {taken.code!r}, which is taken")
                    held = (await conn.execute(select(func.count()).select_from(referral_codes).where(
                        referral_codes.c.hotkey == r.hotkey, referral_codes.c.revoked_at.is_(None)))).scalar()
                    if held >= MAX_CODES:
                        raise Refused(409, "conflict", f"the hotkey holds {MAX_CODES} codes already")
                    await conn.execute(insert(referral_codes).values(  # its time taken under the hotkey's lock
                        code=r.code, folded=fold(r.code), hotkey=r.hotkey, registration=signed.model_dump_json(),
                        created_at=dt.datetime.now(dt.UTC)))
                return await self.code(r.code), True
            except IntegrityError:  # registered at the same time by another call: as it stands now
                if last:
                    raise
        raise AssertionError("unreachable")

    async def code(self, code: str) -> dict[str, Any] | None:
        """The code that ``code`` reads as (in any case, folded), with its hotkey; None if there is none."""
        if not well_formed(code.lower()):
            return None
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(
                referral_codes.c.code, referral_codes.c.hotkey, referral_codes.c.created_at,
                referral_codes.c.revoked_at).where(referral_codes.c.folded == fold(code)))).first()
        return None if row is None else dict(row._mapping)

    async def codes(self, hotkey: str) -> list[dict[str, Any]]:
        """The hotkey's codes, revoked ones too, oldest first."""
        async with self.engine.connect() as conn:
            rows = await conn.execute(select(referral_codes.c.code, referral_codes.c.hotkey,
                                             referral_codes.c.created_at, referral_codes.c.revoked_at)
                                      .where(referral_codes.c.hotkey == hotkey).order_by(referral_codes.c.created_at,
                                                                                        referral_codes.c.code))
            return [dict(r._mapping) for r in rows]

    async def move(self, signed: SignedMove) -> tuple[dict[str, Any], bool]:
        """Move every code of ``from_hotkey`` to ``to_hotkey``. -> ({from_hotkey, to_hotkey, codes: the codes
        to_hotkey holds now}, whether the move is new); the same signed move again is not made twice. Raises Refused."""
        m = signed.move
        self._check(m.netuid, m.site, m.issued_ms)
        if not signed.verify():
            raise Refused(400, "invalid_request", "the signature does not verify against the hotkey the codes move to")
        if m.from_hotkey == m.to_hotkey:
            raise Refused(400, "invalid_request", "from_hotkey and to_hotkey are the same hotkey")
        try:
            async with self.engine.begin() as conn:
                await self._lock_hotkeys(conn, m.from_hotkey, m.to_hotkey)
                created = not (await conn.execute(select(referral_moves.c.id).where(
                    referral_moves.c.signature == signed.signature))).first()
                if created:
                    held = dict((await conn.execute(
                        select(referral_codes.c.hotkey, func.count()).where(
                            referral_codes.c.hotkey.in_((m.from_hotkey, m.to_hotkey)),
                            referral_codes.c.revoked_at.is_(None)).group_by(referral_codes.c.hotkey))).all())
                    if sum(held.values()) > MAX_CODES:
                        raise Refused(409, "conflict", f"{m.to_hotkey} would hold more than {MAX_CODES} codes")
                    moved = (await conn.execute(update(referral_codes).where(
                        referral_codes.c.hotkey == m.from_hotkey).values(hotkey=m.to_hotkey))).rowcount
                    if not moved:
                        raise Refused(404, "not_found", f"{m.from_hotkey} holds no referral code")
                    await conn.execute(insert(referral_moves).values(  # its time taken under the hotkeys' locks
                        from_hotkey=m.from_hotkey, to_hotkey=m.to_hotkey, move=signed.model_dump_json(),
                        signature=signed.signature, created_at=dt.datetime.now(dt.UTC)))
        except IntegrityError:  # the same move made at the same time by another call
            created = False
        return {"from_hotkey": m.from_hotkey, "to_hotkey": m.to_hotkey,
                "codes": [c["code"] for c in await self.codes(m.to_hotkey)]}, created

    async def revoke(self, code: str) -> dict[str, Any] | None:
        """Revoke the code ``code`` reads as (once); None if there is none."""
        found = await self.code(code)
        if found is None:
            return None
        async with self.engine.begin() as conn:
            await conn.execute(update(referral_codes).where(referral_codes.c.code == found["code"],
                                                            referral_codes.c.revoked_at.is_(None))
                               .values(revoked_at=dt.datetime.now(dt.UTC)))
        return await self.code(found["code"])

    # ---- attribution ----

    async def refer(self, account: str, code: str) -> tuple[dict[str, Any], bool]:
        """Set the account's referral code, once and only before its first top-up. -> (the referral, whether it is
        new); the same code again returns it unchanged. Raises LookupError for an unknown account or a code that does
        not exist or is revoked, Refused (409) once the account has another code or a top-up."""
        found = await self.code(code)
        if found is None or found["revoked_at"] is not None:
            raise LookupError(f"no referral code {code!r}")
        for last in (False, True):
            try:
                async with self.engine.begin() as conn:
                    account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account)
                                                     .with_for_update())).scalar()  # not while a top-up is credited
                    if account_id is None:
                        raise LookupError(f"no account {account!r}")
                    current = (await conn.execute(select(referrals.c.code).where(
                        referrals.c.account_id == account_id))).scalar()
                    if current is not None and current != found["code"]:
                        raise Refused(409, "conflict", f"the account was referred with {current!r}: its referrer is "
                                                       "fixed")
                    created = current is None
                    if created:
                        if (await conn.execute(select(exists().where(credits.c.account_id == account_id,
                                                                     credits.c.kind == "topup")))).scalar():
                            raise Refused(409, "conflict", "the account has topped up: its referrer is fixed")
                        await conn.execute(insert(referrals).values(account_id=account_id, code=found["code"]))
                return (await self.referral(account))["referral"], created
            except IntegrityError:  # referred at the same time by another call: as it stands now
                if last:
                    raise
        raise AssertionError("unreachable")

    async def referral(self, account: str) -> dict[str, Any] | None:
        """{referral: {code, hotkey, referred_at} or None, topped_up}; None for an unknown account."""
        async with self.engine.connect() as conn:
            account_id = (await conn.execute(select(accounts.c.id).where(accounts.c.name == account))).scalar()
            if account_id is None:
                return None
            row = (await conn.execute(select(referrals.c.code, referral_codes.c.hotkey, referrals.c.referred_at)
                                      .join(referral_codes, referral_codes.c.code == referrals.c.code)
                                      .where(referrals.c.account_id == account_id))).first()
            topped = (await conn.execute(select(exists().where(credits.c.account_id == account_id,
                                                               credits.c.kind == "topup")))).scalar()
        return {"referral": None if row is None else dict(row._mapping), "topped_up": bool(topped)}

    async def referrer(self, hotkey: str, days: int = 30, now: dt.datetime | None = None) -> dict[str, Any]:
        """A referrer's figures over the last ``days`` UTC days, today's included, per day and code: accounts
        referred, accounts that topped up, their top-ups and refunds, and what counted for the referral pool
        (micro-USD); per code, the accounts it referred all time. Never an account, only counts and sums. Raises
        ValueError for days outside 1 to ``MAX_DAYS``."""
        if not 1 <= days <= MAX_DAYS:
            raise ValueError(f"days is 1 to {MAX_DAYS}")
        now = now or dt.datetime.now(dt.UTC)
        start = now.date() - dt.timedelta(days=days - 1)
        since = dt.datetime.combine(start, dt.time(), dt.UTC)
        codes = await self.codes(hotkey)
        names = [c["code"] for c in codes]
        figures = ("referred", "paying", "topups_micros", "refunds_micros", "counted_micros")
        buckets = {start + dt.timedelta(days=i): {c: dict.fromkeys(figures, 0) for c in names} for i in range(days)}
        referred_day, credited_day = self._day(referrals.c.referred_at), self._day(credits.c.created_at)
        referred_credits = credits.join(referrals, referrals.c.account_id == credits.c.account_id)

        def per_day(value: Any, kind: str) -> Any:
            return (select(referrals.c.code, credited_day, value).select_from(referred_credits)
                    .where(referrals.c.code.in_(names), credits.c.created_at >= since, credits.c.kind == kind)
                    .group_by(referrals.c.code, credited_day))

        async with self.engine.connect() as conn:
            def add(rows: Any, figure: str, sign: int = 1) -> None:
                for code, day, value in rows:
                    if (bucket := buckets.get(_date(day))) is not None:
                        bucket[code][figure] += sign * int(value or 0)

            add(await conn.execute(select(referrals.c.code, referred_day, func.count()).where(
                referrals.c.code.in_(names), referrals.c.referred_at >= since).group_by(referrals.c.code,
                                                                                        referred_day)), "referred")
            add(await conn.execute(per_day(func.count(distinct(credits.c.account_id)), "topup")), "paying")
            add(await conn.execute(per_day(func.sum(credits.c.amount), "topup")), "topups_micros")
            add(await conn.execute(per_day(func.sum(credits.c.amount), "refund")), "refunds_micros", -1)
            add(await conn.execute(select(referral_reports.c.code, credited_day, func.sum(referral_reports.c.amount))
                                   .join(credits, credits.c.id == referral_reports.c.credit_id)
                                   .where(referral_reports.c.code.in_(names), credits.c.created_at >= since)
                                   .group_by(referral_reports.c.code, credited_day)), "counted_micros")
            paying = (await conn.execute(select(func.count(distinct(credits.c.account_id))).select_from(
                referred_credits).where(referrals.c.code.in_(names), credits.c.created_at >= since,
                                        credits.c.kind == "topup"))).scalar()
            per_code = dict((await conn.execute(select(referrals.c.code, func.count()).where(
                referrals.c.code.in_(names)).group_by(referrals.c.code))).all())
        totals = {f: sum(b[c][f] for b in buckets.values() for c in names) for f in figures}
        return {"hotkey": hotkey, "as_of": now.isoformat(timespec="seconds"), "since": start.isoformat(),
                "days": days, "codes": [{**c, "referred": int(per_code.get(c["code"], 0))} for c in codes],
                "totals": {**totals, "paying": int(paying or 0)},
                "buckets": [{"start": day.isoformat(), "codes": figures_} for day, figures_ in buckets.items()]}

    def _day(self, column: Any) -> Any:
        """The UTC day of a timestamp column, in SQL (PostgreSQL would take the session's time zone)."""
        if self.engine.dialect.name == "postgresql":
            return func.date(func.timezone(literal_column("'UTC'"), column), type_=Date)
        return func.date(column, type_=Date)

    # ---- what counts, reported ----

    async def _locked(self, conn: Any) -> bool:
        """Whether this process holds the reporter's lock for the transaction (always on SQLite: one process)."""
        if conn.dialect.name != "postgresql":
            return True
        return bool((await conn.execute(text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": LOCK_KEY})).scalar())

    async def decide(self, pool: ReferralPool | None, epoch_blocks: int) -> list[int]:
        """Decide what the credits of referred accounts not decided yet count (see the module), at most ``BATCH`` of
        them, oldest first. -> the micro-USD each counts (empty if there were none, or another process decides)."""
        window_s = pool.window_s(epoch_blocks, BLOCK_TIME_S) if pool is not None else 0.0
        async with self.engine.begin() as conn:
            if not await self._locked(conn):
                return []
            rows = (await conn.execute(
                select(credits.c.id, credits.c.account_id, credits.c.amount, credits.c.kind, credits.c.reference,
                       credits.c.refund_of, credits.c.created_at, referrals.c.code, referrals.c.referred_at,
                       referral_codes.c.revoked_at)
                .join(referrals, referrals.c.account_id == credits.c.account_id)
                .join(referral_codes, referral_codes.c.code == referrals.c.code)
                .outerjoin(referral_reports, referral_reports.c.credit_id == credits.c.id)
                .where(credits.c.kind.in_(BOUGHT), referral_reports.c.credit_id.is_(None))
                .order_by(credits.c.id).limit(BATCH))).all()
            earlier: dict[int, list[Decided]] = {}
            for account in sorted({r.account_id for r in rows}):
                earlier[account] = [Decided(kind, reference, refund_of, _utc(created), int(amount)) for
                                    kind, reference, refund_of, created, amount in await conn.execute(
                    select(credits.c.kind, credits.c.reference, credits.c.refund_of, credits.c.created_at,
                           referral_reports.c.amount).join(referral_reports,
                                                           referral_reports.c.credit_id == credits.c.id)
                    .where(credits.c.account_id == account))]
            decided = []
            for r in rows:
                created = _utc(r.created_at)
                amount = counted(r.kind, int(r.amount), r.refund_of, created, _utc(r.referred_at),
                                 _utc(r.revoked_at) if r.revoked_at is not None else None, earlier[r.account_id],
                                 pool, window_s)
                earlier[r.account_id].append(Decided(r.kind, r.reference, r.refund_of, created, amount))
                decided.append({"credit_id": r.id, "code": r.code, "amount": amount,
                                "entry_id": secrets.token_hex(32)})  # its id at the auditor, kept for every post
            if decided:
                await conn.execute(insert(referral_reports), decided)
        if decided:
            log.info("referrals: %d top-ups and refunds of referred accounts decided, %d of them count (%+.2f USD)",
                     len(decided), sum(d["amount"] != 0 for d in decided), sum(d["amount"] for d in decided) / 1e6)
        return [d["amount"] for d in decided]

    async def report(self, post: Callable[[ReferralReport], Awaitable[dict[str, Any]]]) -> int:
        """Post what the auditor has not taken (see the module) with ``post``, which returns the auditor's answer or
        raises; mark what it took. -> the items posted (0 if none, or another process reports)."""
        async with self.engine.begin() as conn:
            if conn.dialect.name == "postgresql":  # codes and moves read from one snapshot
                await conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            if not await self._locked(conn):
                return 0
            pending = (await conn.execute(select(
                referral_codes.c.code, referral_codes.c.registration, referral_codes.c.created_at).where(
                referral_codes.c.reported_at.is_(None)).order_by(referral_codes.c.created_at, referral_codes.c.code)
                .limit(MAX_REFERRAL_SIGNED + 1))).all()
            moves = (await conn.execute(select(referral_moves.c.id, referral_moves.c.move, referral_moves.c.created_at)
                                        .where(referral_moves.c.reported_at.is_(None)).order_by(referral_moves.c.id)
                                        .limit(MAX_REFERRAL_SIGNED))).all()
            # in the order they happened: the auditor takes a post's registrations before its moves, and applies a
            # move to the codes it took before the move
            codes = []
            for c in pending:
                if len(codes) == MAX_REFERRAL_SIGNED or (moves and _utc(c.created_at) > _utc(moves[0].created_at)):
                    break
                codes.append(c)
            later = pending[len(codes)] if len(pending) > len(codes) else None
            if later is not None:
                moves = [m for m in moves if _utc(m.created_at) < _utc(later.created_at)]
            entries = [] if later is not None else (await conn.execute(  # entries once their codes are out
                select(referral_reports.c.credit_id, referral_reports.c.entry_id, referral_reports.c.code,
                       referral_reports.c.amount, credits.c.created_at)
                .join(credits, credits.c.id == referral_reports.c.credit_id)
                .where(referral_reports.c.reported_at.is_(None), referral_reports.c.amount != 0)
                .order_by(referral_reports.c.credit_id).limit(MAX_REFERRAL_ENTRIES))).all()
            if not (codes or moves or entries):
                return 0
            report = ReferralReport(
                codes=[SignedRegistration.model_validate_json(c.registration) for c in codes],
                moves=[SignedMove.model_validate_json(m.move) for m in moves],
                entries=[ReferralEntry(id=e.entry_id, code=e.code, amount_micros=int(e.amount),
                                       credited_ms=int(_utc(e.created_at).timestamp() * 1000)) for e in entries])
            answer = await post(report)
            unknown = set(answer.get("unknown_codes") or []) if isinstance(answer, dict) else set()
            now = dt.datetime.now(dt.UTC)
            if codes:
                await conn.execute(update(referral_codes).where(referral_codes.c.code.in_([c.code for c in codes]))
                                   .values(reported_at=now))
            if moves:
                await conn.execute(update(referral_moves).where(referral_moves.c.id.in_([m.id for m in moves]))
                                   .values(reported_at=now))
            taken = [e.credit_id for e in entries if e.code not in unknown]
            if taken:
                await conn.execute(update(referral_reports).where(referral_reports.c.credit_id.in_(taken))
                                   .values(reported_at=now))
            if unknown:
                log.warning("referrals: the auditor does not know the codes %s: every registration and move goes "
                            "again", ", ".join(sorted(unknown)))
                await conn.execute(update(referral_codes).values(reported_at=None))
                await conn.execute(update(referral_moves).values(reported_at=None))
        log.info("referrals: reported %d registrations, %d moves and %d entries to the auditor", len(codes),
                 len(moves), len(taken))
        return len(codes) + len(moves) + len(entries)


class ReferralReporter:
    """Decides and reports, every ``REPORT_S``, under the catalog followed (``catalog``; nothing while there is
    none)."""

    def __init__(self, referrals: Referrals, keypair: Keypair, client: httpx.AsyncClient, *,
                 auditor: Callable[[], Awaitable[Service | None]], auditor_hotkey: str,
                 catalog: Callable[[], Catalog | None]) -> None:
        self.referrals, self.keypair, self.client = referrals, keypair, client
        self.auditor, self.auditor_hotkey, self.catalog = auditor, auditor_hotkey, catalog

    async def step(self) -> None:
        cat = self.catalog()
        if cat is None:
            return
        await self.referrals.decide(cat.scoring.referral_pool, cat.epoch_blocks)
        await self.referrals.report(self.post)

    async def post(self, report: ReferralReport) -> dict[str, Any]:
        auditor = await self.auditor()
        if auditor is None:
            raise RuntimeError("the auditor's address is unknown")
        body = report.model_dump_json().encode()

        def headers() -> dict[str, str]:
            return {"Content-Type": "application/json",
                    **sign_request(self.keypair, "POST", REFERRALS_PATH, body, self.auditor_hotkey)}
        response = await post_idempotent(self.client, auditor.url.rstrip("/") + REFERRALS_PATH, body, headers,
                                         timeout=10, tls=auditor.pin)
        return json.loads(response.content)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(REPORT_S)
            try:
                await self.step()
            except Exception as exc:
                log.warning("referrals: not reported (tried again): %s: %s", type(exc).__name__, exc)


def _date(value: dt.date | str) -> dt.date:
    return value if isinstance(value, dt.date) else dt.date.fromisoformat(value)


def _utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; they are UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
