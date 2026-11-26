"""Keep each graph forecast's distributions and baselines, from before it is sent whole

Revision ID: 9a4d2e7c5b18
Revises: 6c1e9a3f7b24
Create Date: 2026-11-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "9a4d2e7c5b18"
down_revision: Union[str, None] = "6c1e9a3f7b24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE dkg_forecasts_with_distributions (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL,
                opened TEXT,
                distributions TEXT,
                baselines TEXT,
                invalid_reason TEXT,
                failed_check TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, question_id),
                FOREIGN KEY (miner_uid, miner_hotkey, question_id)
                    REFERENCES dkg_forecast_hashes (miner_uid, miner_hotkey, question_id),
                CHECK (opened IS NULL OR invalid_reason IS NULL),
                CHECK (
                    opened IS NOT NULL OR invalid_reason IS NOT NULL OR distributions IS NOT NULL
                )
            )
        """
    )
    op.execute(
        """
            INSERT INTO dkg_forecasts_with_distributions (
                miner_uid, miner_hotkey, question_id, opened, invalid_reason, failed_check,
                created_at, updated_at
            )
            SELECT
                miner_uid, miner_hotkey, question_id, opened, invalid_reason, failed_check,
                created_at, updated_at
            FROM dkg_forecasts
        """
    )
    op.execute("DROP TABLE dkg_forecasts")
    op.execute("ALTER TABLE dkg_forecasts_with_distributions RENAME TO dkg_forecasts")
