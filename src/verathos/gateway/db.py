"""The gateway's tables (SQLAlchemy Core), its database engine and the schema check.

PostgreSQL (asyncpg) in production, SQLite (aiosqlite) for development and tests. The schema is managed by Alembic
(``migrations/``): ``verathos-gateway db upgrade`` brings a database to the current version, and the gateway refuses
to start on a database at any other version.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    false,
    func,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

MIGRATIONS = Path(__file__).parent / "migrations"
STATEMENT_TIMEOUT_MS, LOCK_TIMEOUT_MS = 10_000, 5_000
COMMAND_TIMEOUT_S, POOL_TIMEOUT_S = 15.0, 10.0
ID = BigInteger().with_variant(Integer, "sqlite")  # SQLite autoincrements only INTEGER keys

metadata = MetaData()

accounts = Table(
    "accounts", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("name", String(128), nullable=False, unique=True),
    Column("status", String(16), nullable=False, server_default="active"),  # active | suspended
    Column("rate_rps", Float, nullable=False),
    Column("burst", Integer, nullable=False),
    Column("monthly_quota", BigInteger, nullable=True),  # requests per calendar month (UTC); None: unlimited
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("prepaid", Boolean, nullable=False, server_default=false()),  # served only while the balance is positive
    Column("balance", BigInteger, nullable=False, server_default="0"),  # micro-USD: the credits minus the usage cost
)

api_keys = Table(
    "api_keys", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("account_id", ID, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("key_hash", String(64), nullable=False, unique=True),  # SHA-256 of the key; the key is never stored
    Column("prefix", String(16), nullable=False, unique=True),  # the key's first characters, to recognise it
    Column("name", String(128), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("expires_at", DateTime(timezone=True), nullable=True),
    Column("revoked_at", DateTime(timezone=True), nullable=True),
)

usage_daily = Table(
    "usage_daily", metadata,
    Column("account_id", ID, ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True),
    Column("key_id", ID, ForeignKey("api_keys.id", ondelete="CASCADE"), primary_key=True),
    Column("model_id", String(128), primary_key=True),
    Column("day", Date, primary_key=True),  # UTC
    Column("requests", BigInteger, nullable=False),  # answered
    Column("failed", BigInteger, nullable=False),  # accepted but not answered (no capacity, miners failed)
    Column("input_tokens", BigInteger, nullable=False),
    Column("cost", BigInteger, nullable=False, server_default="0"),  # micro-USD charged
)

usage_hourly = Table(  # every account's answered requests together, per model and hour: the network's totals
    "usage_hourly", metadata,
    Column("model_id", String(128), primary_key=True),
    Column("hour", DateTime(timezone=True), primary_key=True),  # its start, UTC
    Column("requests", BigInteger, nullable=False),
    Column("input_tokens", BigInteger, nullable=False),
)

usage_flushes = Table(  # the usage batches written, so a batch written again is not counted twice
    "usage_flushes", metadata,
    Column("id", String(32), primary_key=True),
    Column("written_at", DateTime(timezone=True), nullable=False, index=True),
)

credits = Table(
    "credits", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("account_id", ID, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("amount", BigInteger, nullable=False),  # micro-USD; negative only for an adjustment
    Column("kind", String(16), nullable=False),  # grant | topup | adjust
    Column("reference", String(128), nullable=False, unique=True),  # the payment or grant it records, once
    Column("note", String(256), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


def create_engine(url: str) -> AsyncEngine:
    """On PostgreSQL every statement is bounded: the server cancels one after ``STATEMENT_TIMEOUT_MS`` or one waiting
    ``LOCK_TIMEOUT_MS`` for a lock (a transaction left open by hand), the client gives up after ``COMMAND_TIMEOUT_S``
    (an unreachable server), and a connection is waited for at most ``POOL_TIMEOUT_S``."""
    if url.startswith("postgresql"):
        return create_async_engine(
            url, pool_size=10, max_overflow=10, pool_pre_ping=True, pool_recycle=1800, pool_timeout=POOL_TIMEOUT_S,
            connect_args={"command_timeout": COMMAND_TIMEOUT_S,
                          "server_settings": {"statement_timeout": str(STATEMENT_TIMEOUT_MS),
                                              "lock_timeout": str(LOCK_TIMEOUT_MS)}})
    return create_async_engine(url)


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))  # the config parser interpolates %
    return config


def upgrade(url: str) -> None:
    """Bring the database to the current schema (run outside an event loop)."""
    command.upgrade(alembic_config(url), "head")


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config("")).get_current_head()


async def schema_revision(engine: AsyncEngine) -> str | None:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: MigrationContext.configure(c).get_current_revision())
