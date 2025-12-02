from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import torch_or_numpy_to_int
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import NuminousLogger

MOVING_AVERAGE_EVENTS = 101  # How many previous events to consider for the moving average
BURN_WEIGHT = 0.80  # Burn UID gets this percentage
WINNER_WEIGHT = 0.99  # Winner gets this percentage of remaining (after burn)
DECAY_POWER = 1.0  # Decay steepness: 1.0=inverse rank, 1.5=steeper, 0.5=gentler


class MetagraphScoring(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    logger: NuminousLogger
    metagraph: IfMetagraph

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        logger: NuminousLogger,
        metagraph: IfMetagraph,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations
        self.metagraph = metagraph

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self):
        return "metagraph-scoring"

    @property
    def interval_seconds(self):
        return self.interval

    def get_owner_neuron_uid(self) -> int:
        owner_uid = None
        owner_hotkey = self.metagraph.owner_hotkey

        for idx, uid in enumerate(self.metagraph.uids):
            int_uid = torch_or_numpy_to_int(uid)
            hotkey = self.metagraph.hotkeys[idx]

            if hotkey == owner_hotkey:
                owner_uid = int_uid
                break

        assert owner_uid is not None, "Owner uid not found in metagraph uids"

        return owner_uid

    async def run(self):
        events_to_score = await self.db_operations.get_events_for_metagraph_scoring(
            max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate metagraph scores.")
        else:
            burn_uid = self.get_owner_neuron_uid()
            self.logger.debug(
                "Found events to calculate metagraph scores.",
                extra={"n_events": len(events_to_score), "burn_uid": burn_uid},
            )

            for event in events_to_score:
                self.logger.debug(
                    "Processing event for metagraph scoring.",
                    extra={"event_id": event["event_id"]},
                )

                try:
                    res = await self.db_operations.set_metagraph_scores(
                        event["event_id"],
                        n_events=MOVING_AVERAGE_EVENTS,
                        burn_weight=BURN_WEIGHT,
                        winner_weight=WINNER_WEIGHT,
                        decay_power=DECAY_POWER,
                        burn_uid=burn_uid,
                    )
                    if res == []:
                        self.logger.debug(
                            "Metagraph scores calculated successfully.",
                            extra={"event_id": event["event_id"]},
                        )
                    else:
                        raise Exception("Error calculating metagraph scores.")
                except Exception:
                    self.errors_count += 1
                    self.logger.exception(
                        "Error calculating metagraph scores.",
                        extra={"event_id": event["event_id"]},
                    )

        self.logger.debug(
            "Metagraph scoring task completed.",
            extra={"errors_count": self.errors_count},
        )

        self.errors_count = 0
