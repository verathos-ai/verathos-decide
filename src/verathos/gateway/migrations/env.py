"""Alembic environment for the gateway's database (async engines)."""

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from verathos.gateway.db import metadata

config = context.config


def run(connection) -> None:
    context.configure(connection=connection, target_metadata=metadata,
                      render_as_batch=connection.dialect.name == "sqlite")
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    engine = create_async_engine(config.get_main_option("sqlalchemy.url"))
    async with engine.connect() as connection:
        await connection.run_sync(run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_online())
