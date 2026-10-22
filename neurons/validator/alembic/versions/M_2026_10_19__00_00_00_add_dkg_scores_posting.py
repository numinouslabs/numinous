"""Record the version that scored each graph forecasting score and when it was posted

Revision ID: 8d3b6f1a4c92
Revises: 5f2c8a1e9d47
Create Date: 2026-10-19 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8d3b6f1a4c92"
down_revision: Union[str, None] = "5f2c8a1e9d47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            DROP TABLE IF EXISTS dkg_scores
        """
    )
    op.execute(
        """
            CREATE TABLE dkg_scores (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                scored_at DATETIME NOT NULL,
                score REAL NOT NULL,
                spec_version INTEGER NOT NULL,
                posted_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, scored_at)
            )
        """
    )
    op.execute(
        """
            CREATE INDEX idx_dkg_scores_scored_at ON dkg_scores (scored_at)
        """
    )
