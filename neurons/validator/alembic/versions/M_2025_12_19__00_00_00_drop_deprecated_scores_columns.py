"""Drop deprecated columns from scores table

Revision ID: 7b7b6f2c2f1a
Revises: e330661ab24a
Create Date: 2025-12-19 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b7b6f2c2f1a"
down_revision: Union[str, None] = "e330661ab24a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN metagraph_score
        """
    )
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN other_data
        """
    )
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN alternative_metagraph_score
        """
    )
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN alternative_other_data
        """
    )
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN alternative_processed
        """
    )
    op.execute(
        """
            ALTER TABLE scores DROP COLUMN processed
        """
    )


def downgrade() -> None:
    pass
