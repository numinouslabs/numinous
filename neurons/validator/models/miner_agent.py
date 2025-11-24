from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MinerAgentsModel(BaseModel):
    version_id: str
    miner_uid: int
    miner_hotkey: str
    agent_name: str
    version_number: int
    file_path: str
    pulled_at: Optional[datetime] = None
    created_at: datetime
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    @property
    def primary_key(self):
        return [
            "version_id",
        ]


MINER_AGENTS_FIELDS = MinerAgentsModel.model_fields.keys()
