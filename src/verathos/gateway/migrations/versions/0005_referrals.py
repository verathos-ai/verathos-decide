"""referral codes, the accounts they referred, and those accounts' top-ups and refunds as they count; credit kind
refund

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

ID = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def upgrade() -> None:
    op.add_column("credits", sa.Column("refund_of", sa.String(128), nullable=True))
    op.create_table(
        "referral_codes",
        sa.Column("code", sa.String(24), primary_key=True),
        sa.Column("folded", sa.String(24), nullable=False, unique=True),
        sa.Column("hotkey", sa.String(64), nullable=False, index=True),
        sa.Column("registration", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "referral_moves",
        sa.Column("id", ID, primary_key=True, autoincrement=True),
        sa.Column("from_hotkey", sa.String(64), nullable=False),
        sa.Column("to_hotkey", sa.String(64), nullable=False),
        sa.Column("move", sa.Text, nullable=False),
        sa.Column("signature", sa.String(256), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "referrals",
        sa.Column("account_id", ID, sa.ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("code", sa.String(24), sa.ForeignKey("referral_codes.code"), nullable=False, index=True),
        sa.Column("referred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "referral_reports",
        sa.Column("credit_id", ID, sa.ForeignKey("credits.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("code", sa.String(24), nullable=False),
        sa.Column("amount", sa.BigInteger, nullable=False),
        sa.Column("reported_at", sa.DateTime(timezone=True), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_table("referral_reports")
    op.drop_table("referrals")
    op.drop_table("referral_moves")
    op.drop_table("referral_codes")
    with op.batch_alter_table("credits") as batch:
        batch.drop_column("refund_of")
