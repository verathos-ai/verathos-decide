"""a random id per referral report, which names its entry to the auditor

The reports kept get one each.

Revision ID: 0006
Revises: 0005
"""

import secrets

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("referral_reports", sa.Column("entry_id", sa.String(64), nullable=True))
    reports = sa.table("referral_reports", sa.column("credit_id", sa.BigInteger), sa.column("entry_id", sa.String))
    bind = op.get_bind()
    kept = bind.execute(sa.select(reports.c.credit_id)).scalars().all()
    if kept:
        bind.execute(reports.update().where(reports.c.credit_id == sa.bindparam("credit"))
                     .values(entry_id=sa.bindparam("entry")),
                     [{"credit": credit, "entry": secrets.token_hex(32)} for credit in kept])
    with op.batch_alter_table("referral_reports") as batch:
        batch.alter_column("entry_id", existing_type=sa.String(64), nullable=False)
        batch.create_unique_constraint("uq_referral_reports_entry_id", ["entry_id"])


def downgrade() -> None:
    with op.batch_alter_table("referral_reports") as batch:
        batch.drop_constraint("uq_referral_reports_entry_id", type_="unique")
        batch.drop_column("entry_id")
