"""Key the graph forecasting hashes, forecasts and results on the miner uid, and pull them again

Revision ID: 9b1d4e7a2c58
Revises: 3e8a5f0c7b21
Create Date: 2026-10-16 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b1d4e7a2c58"
down_revision: Union[str, None] = "3e8a5f0c7b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dkg_statement_results")
    op.execute("DROP TABLE IF EXISTS dkg_forecasts")
    op.execute("DROP TABLE IF EXISTS dkg_forecast_hashes")
    op.execute(
        """
            CREATE TABLE dkg_forecast_hashes (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL REFERENCES dkg_questions (question_id),
                sealed_sha256 TEXT NOT NULL,
                received_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, question_id)
            )
        """
    )
    op.execute(
        """
            CREATE INDEX idx_dkg_forecast_hashes_question_id ON dkg_forecast_hashes (question_id)
        """
    )
    op.execute(
        """
            CREATE TABLE dkg_forecasts (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL,
                opened TEXT,
                invalid_reason TEXT,
                failed_check TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, question_id),
                FOREIGN KEY (miner_uid, miner_hotkey, question_id)
                    REFERENCES dkg_forecast_hashes (miner_uid, miner_hotkey, question_id),
                CHECK ((opened IS NULL) != (invalid_reason IS NULL))
            )
        """
    )
    op.execute(
        """
            CREATE TABLE dkg_statement_results (
                miner_uid INTEGER NOT NULL,
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL,
                statement_position INTEGER NOT NULL,
                outcome BOOLEAN NOT NULL,
                reason TEXT NOT NULL,
                event_id TEXT,
                reported_at DATETIME,
                judged_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_uid, miner_hotkey, question_id, statement_position),
                FOREIGN KEY (miner_uid, miner_hotkey, question_id)
                    REFERENCES dkg_forecasts (miner_uid, miner_hotkey, question_id)
            )
        """
    )
    op.execute(
        """
            UPDATE dkg_rounds SET hashes_pulled_at = NULL, updated_at = CURRENT_TIMESTAMP
        """
    )
    op.execute("DELETE FROM dkg_results_pulls")
