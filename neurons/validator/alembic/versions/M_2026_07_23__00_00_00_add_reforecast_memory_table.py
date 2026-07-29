"""Add reforecast_memory table

Revision ID: a3f1c8b74e29
Revises: 5e7f9a0b3c6d
Create Date: 2026-07-23 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "a3f1c8b74e29"
down_revision: Union[str, None] = "5e7f9a0b3c6d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS reforecast_memory (
                run_id TEXT NOT NULL PRIMARY KEY,
                memory TEXT NOT NULL,
                interval_start_minutes INTEGER NOT NULL,
                exported BOOLEAN NOT NULL DEFAULT false,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
    )
