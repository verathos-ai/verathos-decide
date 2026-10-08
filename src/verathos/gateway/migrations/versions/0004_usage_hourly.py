"""answered requests and input tokens per model and hour, all accounts together

The usage written before is entered per day, at the day's first hour.

Revision ID: 0004
Revises: 0003
"""

import datetime as dt

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    hourly = op.create_table(
        "usage_hourly",
        sa.Column("model_id", sa.String(128), primary_key=True),
        sa.Column("hour", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("requests", sa.BigInteger, nullable=False),
        sa.Column("input_tokens", sa.BigInteger, nullable=False),
    )
    daily = sa.table("usage_daily", sa.column("model_id", sa.String), sa.column("day", sa.Date),
                     sa.column("requests", sa.BigInteger), sa.column("input_tokens", sa.BigInteger))
    rows = op.get_bind().execute(sa.select(daily.c.model_id, daily.c.day, sa.func.sum(daily.c.requests),
                                           sa.func.sum(daily.c.input_tokens))
                                 .group_by(daily.c.model_id, daily.c.day)).all()
    if rows:
        op.bulk_insert(hourly, [{"model_id": model, "hour": dt.datetime.combine(day, dt.time(), dt.UTC),
                                 "requests": requests, "input_tokens": tokens}
                                for model, day, requests, tokens in rows])


def downgrade() -> None:
    op.drop_table("usage_hourly")
