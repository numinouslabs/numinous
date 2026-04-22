"""Add sources table

Revision ID: 5e7f9a0b3c6d
Revises: 4d6e8f0a2b5c
Create Date: 2026-04-16 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5e7f9a0b3c6d"
down_revision: Union[str, None] = "4d6e8f0a2b5c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            CREATE TABLE IF NOT EXISTS sources (
                run_id TEXT NOT NULL PRIMARY KEY,
                sources TEXT NOT NULL,
                exported BOOLEAN NOT NULL DEFAULT false,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
    )
