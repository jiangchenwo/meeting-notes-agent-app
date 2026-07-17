from __future__ import annotations

from alembic import op

from notes_agent_v2.persistence.models import Base


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())
    for table in ("stage_artifacts", "facts", "documents", "critic_issues", "quality_reports", "model_call_records"):
        op.execute(
            f"CREATE TRIGGER {table}_immutable_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} records are immutable'); END"
        )
        op.execute(
            f"CREATE TRIGGER {table}_immutable_delete BEFORE DELETE ON {table} "
            f"BEGIN SELECT RAISE(ABORT, '{table} records are immutable'); END"
        )


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
