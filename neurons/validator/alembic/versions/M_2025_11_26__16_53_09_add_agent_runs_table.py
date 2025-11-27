"""Add agent_runs table

Revision ID: 40606aaa49f9
Revises: d679a148c4f2
Create Date: 2025-11-26 16:53:09.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "40606aaa49f9"
down_revision: Union[str, None] = "d679a148c4f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id                  TEXT PRIMARY KEY,
                unique_event_id         TEXT NOT NULL,
                agent_version_id        TEXT NOT NULL,
                miner_uid               INTEGER NOT NULL,
                miner_hotkey            TEXT NOT NULL,
                status                  TEXT NOT NULL,
                exported                INTEGER DEFAULT 0,
                is_final                INTEGER DEFAULT 1,
                created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (unique_event_id) REFERENCES events(unique_event_id),
                FOREIGN KEY (agent_version_id) REFERENCES miner_agents(version_id)
            )
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_event ON agent_runs(unique_event_id)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_agent ON agent_runs(agent_version_id)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_miner ON agent_runs(miner_uid, miner_hotkey)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_status ON agent_runs(status)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_exported ON agent_runs(exported) WHERE exported = 0
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_is_final ON agent_runs(is_final)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_agent_runs_created_at ON agent_runs(created_at)
        """
    )


def downgrade() -> None:
    pass
