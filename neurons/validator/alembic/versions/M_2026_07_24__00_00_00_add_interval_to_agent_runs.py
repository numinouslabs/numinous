"""Add interval_start_minutes to agent_runs

Revision ID: c8d5a2e6f741
Revises: b4e2f9a1c7d3
Create Date: 2026-07-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "c8d5a2e6f741"
down_revision: Union[str, None] = "b4e2f9a1c7d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_INTERVAL_FROM_CREATED_AT = (
    "CAST(julianday(date(created_at)) - julianday('2024-01-01') AS INTEGER) * 1440"
)


def upgrade() -> None:
    op.execute("PRAGMA foreign_keys = OFF")

    op.execute(
        """
            CREATE TABLE agent_runs_new (
                run_id                  TEXT PRIMARY KEY,
                unique_event_id         TEXT NOT NULL,
                agent_version_id        TEXT NOT NULL,
                miner_uid               INTEGER NOT NULL,
                miner_hotkey            TEXT NOT NULL,
                track                   TEXT NOT NULL DEFAULT 'MAIN',
                status                  TEXT NOT NULL,
                interval_start_minutes  INTEGER NOT NULL,
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
        f"""
            INSERT INTO agent_runs_new (
                run_id,
                unique_event_id,
                agent_version_id,
                miner_uid,
                miner_hotkey,
                track,
                status,
                interval_start_minutes,
                exported,
                is_final,
                created_at,
                updated_at
            )
            SELECT
                run_id,
                unique_event_id,
                agent_version_id,
                miner_uid,
                miner_hotkey,
                track,
                status,
                {_INTERVAL_FROM_CREATED_AT},
                exported,
                is_final,
                created_at,
                updated_at
            FROM agent_runs
        """
    )

    op.execute("DROP TABLE agent_runs")
    op.execute("ALTER TABLE agent_runs_new RENAME TO agent_runs")

    op.execute("CREATE INDEX idx_agent_runs_event ON agent_runs(unique_event_id)")
    op.execute("CREATE INDEX idx_agent_runs_agent ON agent_runs(agent_version_id)")
    op.execute("CREATE INDEX idx_agent_runs_miner ON agent_runs(miner_uid, miner_hotkey)")
    op.execute("CREATE INDEX idx_agent_runs_status ON agent_runs(status)")
    op.execute("CREATE INDEX idx_agent_runs_exported ON agent_runs(exported) WHERE exported = 0")
    op.execute("CREATE INDEX idx_agent_runs_is_final ON agent_runs(is_final)")
    op.execute("CREATE INDEX idx_agent_runs_created_at ON agent_runs(created_at)")

    op.execute(
        """
            CREATE INDEX idx_agent_runs_event_agent_interval
            ON agent_runs(unique_event_id, agent_version_id, interval_start_minutes)
        """
    )

    op.execute("PRAGMA foreign_keys = ON")
