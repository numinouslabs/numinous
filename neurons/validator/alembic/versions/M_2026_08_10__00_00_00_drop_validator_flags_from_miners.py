"""Drop write-only validator flag columns from miners table

Revision ID: f2a7c4d19e83
Revises: c8d5a2e6f741
Create Date: 2026-08-10 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2a7c4d19e83"
down_revision: Union[str, None] = "c8d5a2e6f741"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE miners DROP COLUMN is_validating
        """
    )
    op.execute(
        """
            ALTER TABLE miners DROP COLUMN validator_permit
        """
    )


def downgrade() -> None:
    pass
