"""Backfill agent_runs from predictions

Revision ID: e330661ab24a
Revises: 40606aaa49f9
Create Date: 2025-11-28 15:32:05.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e330661ab24a"
down_revision: Union[str, None] = "0fc1f13544dc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            INSERT OR IGNORE INTO agent_runs (
                run_id,
                unique_event_id,
                agent_version_id,
                miner_uid,
                miner_hotkey,
                status,
                exported,
                is_final,
                created_at,
                updated_at
            )
            SELECT
                run_id,
                unique_event_id,
                version_id,
                miner_uid,
                miner_hotkey,
                'SUCCESS',
                0,
                1,
                submitted,
                submitted
            FROM predictions
            WHERE run_id IS NOT NULL
              AND version_id IS NOT NULL
        """
    )


def downgrade() -> None:
    pass
