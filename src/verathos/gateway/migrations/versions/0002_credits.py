"""prepaid accounts: balances, the credit ledger and the cost of usage

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

ID = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def upgrade() -> None:
    op.add_column("accounts", sa.Column("prepaid", sa.Boolean, nullable=False, server_default=sa.false()))
    op.add_column("accounts", sa.Column("balance", sa.BigInteger, nullable=False, server_default="0"))
    op.add_column("usage_daily", sa.Column("cost", sa.BigInteger, nullable=False, server_default="0"))
    op.create_table(
        "credits",
        sa.Column("id", ID, primary_key=True, autoincrement=True),
        sa.Column("account_id", ID, sa.ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("amount", sa.BigInteger, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("reference", sa.String(128), nullable=False, unique=True),
        sa.Column("note", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("credits")
    with op.batch_alter_table("usage_daily") as batch:
        batch.drop_column("cost")
    with op.batch_alter_table("accounts") as batch:
        batch.drop_column("balance")
        batch.drop_column("prepaid")
