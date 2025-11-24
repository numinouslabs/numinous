"""Add miner agents table

Revision ID: a3f8d2e91c4b
Revises: c8a61b7b8c8a
Create Date: 2025-10-28 12:06:33.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3f8d2e91c4b"
down_revision: Union[str, None] = "c8a61b7b8c8a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS miner_agents (
                version_id              TEXT PRIMARY KEY,
                miner_uid               INTEGER NOT NULL,
                miner_hotkey            TEXT NOT NULL,
                agent_name              TEXT NOT NULL,
                version_number          INTEGER NOT NULL,
                file_path               TEXT NOT NULL,
                pulled_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                created_at              DATETIME NOT NULL,
                UNIQUE(miner_uid, miner_hotkey, version_number)
            )
        """
    )

    op.execute(
        """
            CREATE INDEX idx_miner_agents_lookup ON miner_agents(miner_uid, miner_hotkey)
        """
    )

    op.execute(
        """
            CREATE INDEX idx_miner_agents_pulled ON miner_agents(pulled_at)
        """
    )
