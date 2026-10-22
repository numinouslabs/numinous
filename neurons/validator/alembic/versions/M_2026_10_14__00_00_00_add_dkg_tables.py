"""Add the graph forecasting tables

Revision ID: 7c2e9b4d1a6f
Revises: f2a7c4d19e83
Create Date: 2026-10-14 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c2e9b4d1a6f"
down_revision: Union[str, None] = "f2a7c4d19e83"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_rounds (
                round_id TEXT NOT NULL PRIMARY KEY,
                deadline_at DATETIME NOT NULL,
                horizon_end_at DATETIME NOT NULL,
                hashes_pulled_at DATETIME,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
    )
    op.execute(
        """
            CREATE INDEX IF NOT EXISTS idx_dkg_rounds_deadline_at ON dkg_rounds (deadline_at)
        """
    )
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_questions (
                question_id TEXT NOT NULL PRIMARY KEY,
                round_id TEXT NOT NULL REFERENCES dkg_rounds (round_id),
                entity_id TEXT NOT NULL,
                entity TEXT NOT NULL,
                counterparty_id TEXT,
                counterparty TEXT,
                relationship TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
    )
    op.execute(
        """
            CREATE INDEX IF NOT EXISTS idx_dkg_questions_round_id ON dkg_questions (round_id)
        """
    )
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_forecast_hashes (
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL REFERENCES dkg_questions (question_id),
                sealed_sha256 TEXT NOT NULL,
                received_at DATETIME NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_hotkey, question_id)
            )
        """
    )
    op.execute(
        """
            CREATE INDEX IF NOT EXISTS idx_dkg_forecast_hashes_question_id
                ON dkg_forecast_hashes (question_id)
        """
    )
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_forecasts (
                miner_hotkey TEXT NOT NULL,
                question_id TEXT NOT NULL,
                opened TEXT,
                invalid_reason TEXT,
                failed_check TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (miner_hotkey, question_id),
                FOREIGN KEY (miner_hotkey, question_id)
                    REFERENCES dkg_forecast_hashes (miner_hotkey, question_id),
                CHECK ((opened IS NULL) != (invalid_reason IS NULL))
            )
        """
    )
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_statement_results (
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
                PRIMARY KEY (miner_hotkey, question_id, statement_position),
                FOREIGN KEY (miner_hotkey, question_id)
                    REFERENCES dkg_forecasts (miner_hotkey, question_id)
            )
        """
    )
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS dkg_results_pulls (
                started_at DATETIME NOT NULL PRIMARY KEY
            )
        """
    )
