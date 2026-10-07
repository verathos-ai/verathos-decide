"""accounts, API keys and daily usage

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

ID = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", ID, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("rate_rps", sa.Float, nullable=False),
        sa.Column("burst", sa.Integer, nullable=False),
        sa.Column("monthly_quota", sa.BigInteger, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "api_keys",
        sa.Column("id", ID, primary_key=True, autoincrement=True),
        sa.Column("account_id", ID, sa.ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "usage_daily",
        sa.Column("account_id", ID, sa.ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("key_id", ID, sa.ForeignKey("api_keys.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model_id", sa.String(128), primary_key=True),
        sa.Column("day", sa.Date, primary_key=True),
        sa.Column("requests", sa.BigInteger, nullable=False),
        sa.Column("failed", sa.BigInteger, nullable=False),
        sa.Column("input_tokens", sa.BigInteger, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("usage_daily")
    op.drop_table("api_keys")
    op.drop_table("accounts")
