"""Drop the tables of the old forecasting flow

Revision ID: 2b7e4c9d1f63
Revises: 8d3b6f1a4c92
Create Date: 2026-10-19 00:00:01.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2b7e4c9d1f63"
down_revision: Union[str, None] = "8d3b6f1a4c92"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_TABLES = [
    "agent_run_logs",
    "agent_runs",
    "predictions",
    "events",
    "miner_agents",
    "reasoning",
    "reforecast_memory",
    "scores",
    "sources",
]


def upgrade() -> None:
    for table in OLD_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
