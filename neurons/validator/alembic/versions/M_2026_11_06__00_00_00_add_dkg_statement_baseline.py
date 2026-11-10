"""Keep the baseline the backend priced each graph forecasting statement at

Revision ID: 6c1e9a3f7b24
Revises: 2b7e4c9d1f63
Create Date: 2026-11-06 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "6c1e9a3f7b24"
down_revision: Union[str, None] = "2b7e4c9d1f63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
            ALTER TABLE dkg_statement_results ADD COLUMN baseline REAL
        """
    )
