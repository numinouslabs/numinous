from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class WeightsModel(BaseModel):
    miner_uid: int
    miner_hotkey: str
    metagraph_score: float
    aggregated_at: Optional[datetime] = None

    @property
    def primary_key(self):
        return ["miner_uid", "miner_hotkey"]
