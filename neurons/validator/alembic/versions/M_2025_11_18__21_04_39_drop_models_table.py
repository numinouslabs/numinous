"""drop_models_table

Revision ID: 09014f49f29f
Revises: 80b905a8fd2f
Create Date: 2025-11-18 21:04:39.869743

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "09014f49f29f"
down_revision: Union[str, None] = "80b905a8fd2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            DROP TABLE IF EXISTS models
        """
    )
