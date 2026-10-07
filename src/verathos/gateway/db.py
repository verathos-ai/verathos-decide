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
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    func,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

MIGRATIONS = Path(__file__).parent / "migrations"
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
)

api_keys = Table(
    "api_keys", metadata,
    Column("id", ID, primary_key=True, autoincrement=True),
    Column("account_id", ID, ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("key_hash", String(64), nullable=False, unique=True),  # SHA-256 of the key; the key is never stored
    Column("prefix", String(16), nullable=False),  # the key's first characters, to recognise it
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
)


def create_engine(url: str) -> AsyncEngine:
    if url.startswith("postgresql"):
        return create_async_engine(url, pool_size=10, max_overflow=10, pool_pre_ping=True, pool_recycle=1800)
    return create_async_engine(url)


def alembic_config(url: str) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", url)
    return config


def upgrade(url: str) -> None:
    """Bring the database to the current schema (run outside an event loop)."""
    command.upgrade(alembic_config(url), "head")


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config("")).get_current_head()


async def schema_revision(engine: AsyncEngine) -> str | None:
    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: MigrationContext.configure(c).get_current_revision())
