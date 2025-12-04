"""Add agent_run_logs table

Revision ID: 0fc1f13544dc
Revises: 40606aaa49f9
Create Date: 2025-12-02 15:12:42.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0fc1f13544dc"
down_revision: Union[str, None] = "40606aaa49f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS agent_run_logs (
                run_id                  TEXT PRIMARY KEY,
                log_content             TEXT NOT NULL,
                exported                INTEGER DEFAULT 0,
                created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id)
            )
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_run_logs_exported ON agent_run_logs(exported) WHERE exported = 0
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_run_logs_created_at ON agent_run_logs(created_at)
        """
    )


def downgrade() -> None:
    pass
