"""add_agent_fields_to_predictions

Revision ID: 80b905a8fd2f
Revises: a3f8d2e91c4b
Create Date: 2025-10-31 16:04:43.870131

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "80b905a8fd2f"
down_revision: Union[str, None] = "a3f8d2e91c4b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE predictions ADD COLUMN run_id TEXT;
        """
    )

    op.execute(
        """
        ALTER TABLE predictions ADD COLUMN version_id TEXT;
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_predictions_version_id
            ON predictions(version_id);
        """
    )
