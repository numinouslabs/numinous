"""Add the graph forecasting scores table

Revision ID: 3e8a5f0c7b21
Revises: 7c2e9b4d1a6f
Create Date: 2026-10-16 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3e8a5f0c7b21"
down_revision: Union[str, None] = "7c2e9b4d1a6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_scores (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                scored_at DATETIME NOT NULL,
                score REAL NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, scored_at)
            )
        """
    )
    op.execute(
        """
            CREATE INDEX IF NOT EXISTS idx_dkg_scores_scored_at ON dkg_scores (scored_at)
        """
    )
