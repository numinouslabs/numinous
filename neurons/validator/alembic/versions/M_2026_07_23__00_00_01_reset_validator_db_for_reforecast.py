"""Reset validator db for reforecast - flush all data

Revision ID: b4e2f9a1c7d3
Revises: a3f1c8b74e29
Create Date: 2026-07-23 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import Inspector, text

revision: str = "b4e2f9a1c7d3"
down_revision: Union[str, None] = "a3f1c8b74e29"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)

    table_names = [table for table in inspector.get_table_names() if table != "alembic_version"]

    if conn.dialect.name == "sqlite":
        op.execute(text("PRAGMA foreign_keys=OFF"))

    for table in table_names:
        print(f"Flushing data from table: {table}")
        op.execute(text(f"DELETE FROM {table}"))

    if conn.dialect.name == "sqlite":
        op.execute(text("PRAGMA foreign_keys=ON"))
