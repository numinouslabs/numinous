"""Index the graph forecasting statement results that are right

Revision ID: 5f2c8a1e9d47
Revises: 9b1d4e7a2c58
Create Date: 2026-10-16 00:00:02.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5f2c8a1e9d47"
down_revision: Union[str, None] = "9b1d4e7a2c58"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE INDEX IF NOT EXISTS idx_dkg_statement_results_rights
                ON dkg_statement_results (miner_uid, miner_hotkey, question_id)
                WHERE outcome = 1
        """
    )
