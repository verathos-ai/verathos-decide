"""usage batches written, so a batch written again is not counted twice

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_flushes",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("written_at", sa.DateTime(timezone=True), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table("usage_flushes")
