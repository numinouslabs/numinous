from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.logger.logger import NuminousLogger

MOVING_AVERAGE_EVENTS = 101  # How many previous events to consider for the moving average
WINNER_WEIGHT = 0.99  # Winner gets this percentage
DECAY_POWER = 1.0  # Decay steepness: 1.0=inverse rank, 1.5=steeper, 0.5=gentler


class MetagraphScoring(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        page_size: int,
        db_operations: DatabaseOperations,
        logger: NuminousLogger,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        self.interval = interval_seconds
        self.page_size = page_size
        self.db_operations = db_operations

        self.errors_count = 0
        self.logger = logger

    @property
    def name(self):
        return "metagraph-scoring"

    @property
    def interval_seconds(self):
        return self.interval

    async def run(self):
        events_to_score = await self.db_operations.get_events_for_metagraph_scoring(
            max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate metagraph scores.")
        else:
            self.logger.debug(
                "Found events to calculate metagraph scores.",
                extra={"n_events": len(events_to_score)},
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
                        winner_weight=WINNER_WEIGHT,
                        decay_power=DECAY_POWER,
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
