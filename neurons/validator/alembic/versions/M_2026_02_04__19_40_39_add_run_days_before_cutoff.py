"""add_run_days_before_cutoff

Revision ID: 0580d6156c28
Revises: 7b7b6f2c2f1a
Create Date: 2026-02-04 19:40:39.429493

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0580d6156c28"
down_revision: Union[str, None] = "7b7b6f2c2f1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE events
            ADD COLUMN run_days_before_cutoff INTEGER NOT NULL DEFAULT 2
        """
    )
