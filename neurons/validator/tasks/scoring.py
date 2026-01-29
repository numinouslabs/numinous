import copy
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from bittensor import AsyncSubtensor

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.agent_runs import AgentRunsModel
from neurons.validator.models.event import EventsModel
from neurons.validator.models.prediction import PredictionsModel
from neurons.validator.models.score import ScoresModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import pydantic_models_to_dataframe
from neurons.validator.utils.common.interval import (
    AGGREGATION_INTERVAL_LENGTH_MINUTES,
    SCORING_WINDOW_INTERVALS,
    align_to_interval,
    minutes_since_epoch,
    to_utc,
)
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version

# controls the clipping of predictions [CLIP_EPS, 1 - CLIP_EPS]
CLIP_EPS = 1e-2
# controls the distance mean-min answer penalty for miners which are unresponsive
UPTIME_PENALTY_DISTANCE = 1 / 3

DEFAULT_POWER_DECAY_WEIGHT_EXPONENT = 2


# this is just for avoiding typos in column names
@dataclass
class ScoreNames:
    miner_uid: str = "miner_uid"
    miner_hotkey: str = "miner_hotkey"
    registered_date: str = "registered_date"
    miner_registered_minutes: str = "miner_registered_minutes"
    interval_idx: str = "interval_idx"
    interval_start: str = "interval_start"
    interval_end: str = "interval_end"
    weight: str = "weight"
    interval_agg_prediction: str = "interval_agg_prediction"
    weighted_prediction: str = "weighted_prediction"
    weighted_prediction_sum: str = "weighted_prediction_sum"
    weight_sum: str = "weight_sum"
    rema_prediction: str = "rema_prediction"
    rema_peer_score: str = "rema_peer_score"


class Scoring(AbstractTask):
    interval: float
    page_size: int
    db_operations: DatabaseOperations
    netuid: int
    subtensor_cm: AsyncSubtensor
    logger: NuminousLogger

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        netuid: int,
        subtensor: AsyncSubtensor,
        logger: NuminousLogger,
        page_size: int = 100,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        # Validate db_operations
        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(netuid, int) or netuid < 0:
            raise ValueError("netuid must be a non-negative integer.")

        if not isinstance(subtensor, AsyncSubtensor):
            raise TypeError("subtensor must be an instance of AsyncSubtensor.")

        # get current hotkeys and uids
        # regularly update these during and after each event scoring
        self.netuid = netuid
        self.subtensor_cm = subtensor
        self.current_hotkeys = None
        self.n_hotkeys = None
        self.current_uids = None
        self.current_miners_df = None

        self.spec_version = spec_version

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.miners_last_reg = None

        self.errors_count = 0
        self.logger = logger
        self.page_size = page_size

    @property
    def name(self):
        return "scoring"

    @property
    def interval_seconds(self):
        return self.interval

    def copy_metagraph_state(self):
        # hotkeys is a list[str] and uids is a numpy array
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)
        self.current_miners_df = pd.DataFrame(
            {
                ScoreNames.miner_hotkey: self.current_hotkeys,
                ScoreNames.miner_uid: self.current_uids.tolist(),
            }
        )

    async def miners_last_reg_sync(self) -> bool:
        miners_last_reg_rows = await self.db_operations.get_miners_last_registration()
        if not miners_last_reg_rows:
            self.errors_count += 1
            self.logger.error("No miners found in the DB, skipping scoring!")
            return False

        miners_last_reg = pydantic_models_to_dataframe(miners_last_reg_rows)
        # for some reason, someone decided to store the miner_uid as a string in the DB
        miners_last_reg[ScoreNames.miner_uid] = miners_last_reg[ScoreNames.miner_uid].astype(
            pd.Int64Dtype()
        )

        # inner join to current miners
        self.miners_last_reg = pd.merge(
            miners_last_reg,
            self.current_miners_df,
            on=[ScoreNames.miner_uid, ScoreNames.miner_hotkey],
            how="inner",
        )

        if self.miners_last_reg.empty:
            self.logger.error(
                "No overlap in miners between DB and metagraph, skipping scoring!",
                extra={
                    "db_hotkeys[:10]": self.current_miners_df[ScoreNames.miner_hotkey][
                        :10
                    ].tolist(),
                    "metagraph_hotkeys[:10]": miners_last_reg[ScoreNames.miner_hotkey][
                        :10
                    ].tolist(),
                },
            )
            return False

        # calculate the reg_data as minutes since reference data
        self.miners_last_reg[ScoreNames.miner_registered_minutes] = (
            self.miners_last_reg[ScoreNames.registered_date]
            .apply(to_utc)
            .apply(minutes_since_epoch)
        )

        return True

    @staticmethod
    def set_right_cutoff(input_event: EventsModel):
        event = copy.deepcopy(input_event)
        effective_cutoff = min(
            to_utc(input_event.cutoff), to_utc(input_event.resolved_at), datetime.now(timezone.utc)
        )

        event.cutoff = effective_cutoff
        event.resolved_at = effective_cutoff
        event.registered_date = to_utc(event.registered_date)
        return event

    @staticmethod
    def power_decay_weight(
        idx: int, n_intervals: int, exponent: int = DEFAULT_POWER_DECAY_WEIGHT_EXPONENT
    ):
        """
        General power-law decay:
        w (i) = 1 - (i / (n_intervals - 1)) ** exponent
        """
        if n_intervals <= 1:
            return 1.0

        x = idx / (n_intervals - 1)

        return 1 - x**exponent

    def get_intervals_df(
        self, event_registered_start_minutes: int, event_cutoff_start_minutes: int
    ) -> pd.DataFrame:
        n_intervals = (
            event_cutoff_start_minutes - event_registered_start_minutes
        ) // AGGREGATION_INTERVAL_LENGTH_MINUTES
        if n_intervals <= 0:
            self.logger.error(
                "n_intervals computed to be <= 0",
                extra={
                    "n_intervals": n_intervals,
                    "effective_cutoff_start_minutes": event_cutoff_start_minutes,
                    "registered_date_start_minutes": event_registered_start_minutes,
                },
            )
            return pd.DataFrame(
                columns=[
                    ScoreNames.interval_idx,
                    ScoreNames.interval_start,
                    ScoreNames.interval_end,
                    ScoreNames.weight,
                ]
            )

        intervals = pd.DataFrame({ScoreNames.interval_idx: np.arange(n_intervals)})
        intervals[ScoreNames.interval_start] = (
            event_registered_start_minutes
            + intervals[ScoreNames.interval_idx] * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )
        intervals[ScoreNames.interval_end] = (
            intervals[ScoreNames.interval_start] + AGGREGATION_INTERVAL_LENGTH_MINUTES
        )
        # Reverse exponential MA weights:
        intervals[ScoreNames.weight] = intervals[ScoreNames.interval_idx].apply(
            lambda idx: self.power_decay_weight(idx, n_intervals)
        )

        return intervals

    def prepare_predictions_df(
        self, predictions: list[PredictionsModel], miners: pd.DataFrame
    ) -> pd.DataFrame:
        # consider predictions only for valid miners
        predictions_df = pydantic_models_to_dataframe(predictions)
        predictions_df.rename(
            columns={
                "interval_start_minutes": ScoreNames.interval_start,
            },
            inplace=True,
        )
        predictions_df[ScoreNames.miner_uid] = predictions_df[ScoreNames.miner_uid].astype(
            pd.Int64Dtype()
        )
        predictions_df = pd.merge(
            miners[[ScoreNames.miner_uid, ScoreNames.miner_hotkey]],
            predictions_df,
            on=[ScoreNames.miner_uid, ScoreNames.miner_hotkey],
            how="left",
        )
        predictions_df[ScoreNames.interval_agg_prediction] = predictions_df[
            ScoreNames.interval_agg_prediction
        ].clip(CLIP_EPS, 1 - CLIP_EPS)
        return predictions_df

    def get_interval_scores_base(
        self, predictions_df: pd.DataFrame, miners: pd.DataFrame, intervals: pd.DataFrame
    ) -> pd.DataFrame:
        # all miners should have a row for each interval, then left join with predictions
        miners["key"] = 1
        intervals["key"] = 1
        miners_intervals = pd.merge(
            miners[
                [
                    "key",
                    ScoreNames.miner_uid,
                    ScoreNames.miner_hotkey,
                    ScoreNames.miner_registered_minutes,
                ]
            ],
            intervals,
            on="key",
        )

        interval_scores_df = pd.merge(
            miners_intervals,
            predictions_df,
            on=[ScoreNames.miner_uid, ScoreNames.miner_hotkey, ScoreNames.interval_start],
            how="left",
        )
        # keep only columns needed for scoring
        interval_scores_df = interval_scores_df[
            [
                ScoreNames.miner_uid,
                ScoreNames.miner_hotkey,
                ScoreNames.miner_registered_minutes,
                ScoreNames.interval_idx,
                ScoreNames.interval_start,
                ScoreNames.interval_end,
                ScoreNames.weight,
                ScoreNames.interval_agg_prediction,
            ]
        ]
        return interval_scores_df

    def return_empty_scores_df(self, reason: str, event_id: str) -> pd.DataFrame:
        self.errors_count += 1
        self.logger.error(
            reason,
            extra={
                "event_id": event_id,
            },
        )
        return pd.DataFrame(
            columns=[
                ScoreNames.miner_uid,
                ScoreNames.miner_hotkey,
                ScoreNames.rema_prediction,
                ScoreNames.rema_peer_score,
            ]
        )

    def fill_unresponsive_miners(
        self,
        interval_scores: pd.DataFrame,
        failed_runs: list[AgentRunsModel],
        imputed_prediction: float = 0.5,
    ) -> pd.DataFrame:
        interval_scores_df = interval_scores.copy()
        interval_scores_df[ScoreNames.interval_agg_prediction] = interval_scores_df[
            ScoreNames.interval_agg_prediction
        ].astype("Float64")

        failed_miners = {(run.miner_uid, run.miner_hotkey) for run in failed_runs}

        # Only impute 0.5 for miners with failed runs
        missing_predictions = interval_scores_df[ScoreNames.interval_agg_prediction].isnull()
        has_failed_run = interval_scores_df.apply(
            lambda row: (row[ScoreNames.miner_uid], row[ScoreNames.miner_hotkey]) in failed_miners,
            axis=1,
        )

        should_impute = missing_predictions & has_failed_run
        interval_scores_df.loc[
            should_impute, ScoreNames.interval_agg_prediction
        ] = imputed_prediction

        interval_scores_df = interval_scores_df.dropna(subset=[ScoreNames.interval_agg_prediction])

        return interval_scores_df

    def aggregate_predictions_by_miner(self, interval_scores_df: pd.DataFrame) -> pd.DataFrame:
        # Calculate weighted predictions
        interval_scores_df[ScoreNames.weighted_prediction] = (
            interval_scores_df[ScoreNames.interval_agg_prediction]
            * interval_scores_df[ScoreNames.weight]
        )

        # Group by miner and calculate weighted average prediction
        scores_df = (
            interval_scores_df.groupby([ScoreNames.miner_uid, ScoreNames.miner_hotkey])
            .agg(
                weighted_prediction_sum=(ScoreNames.weighted_prediction, "sum"),
                weight_sum=(ScoreNames.weight, "sum"),
            )
            .reset_index()
        )
        scores_df[ScoreNames.rema_prediction] = (
            scores_df[ScoreNames.weighted_prediction_sum] / scores_df[ScoreNames.weight_sum]
        )

        return scores_df[
            [
                ScoreNames.miner_uid,
                ScoreNames.miner_hotkey,
                ScoreNames.rema_prediction,
            ]
        ]

    async def score_event(
        self, event: EventsModel, predictions: list[PredictionsModel]
    ) -> pd.DataFrame:
        # outcome is text in DB :|
        outcome = float(event.outcome)
        outcome_round = int(round(outcome))  # for safety, should be 0 or 1

        # Convert cutoff and now to minutes since epoch, then align them to the interval start
        event_cutoff_minutes = minutes_since_epoch(event.cutoff)
        event_cutoff_start_minutes = align_to_interval(event_cutoff_minutes)

        # Calculate scoring window start (last N intervals before cutoff)
        scoring_window_start_minutes = event_cutoff_start_minutes - (
            SCORING_WINDOW_INTERVALS * AGGREGATION_INTERVAL_LENGTH_MINUTES
        )

        intervals = self.get_intervals_df(
            event_registered_start_minutes=scoring_window_start_minutes,
            event_cutoff_start_minutes=event_cutoff_start_minutes,
        )
        if intervals.empty:
            await self.db_operations.mark_event_as_discarded(unique_event_id=event.unique_event_id)
            return self.return_empty_scores_df(
                "No intervals to score - event discarded.", event.event_id
            )

        # Do not score miners which registered after the scoring window started
        miners = self.miners_last_reg[
            self.miners_last_reg[ScoreNames.miner_registered_minutes]
            <= scoring_window_start_minutes
        ].copy()
        if miners.empty:
            return self.return_empty_scores_df("No miners to score.", event.event_id)

        # prepare predictions
        predictions_df = self.prepare_predictions_df(predictions=predictions, miners=miners)
        if predictions_df.empty:
            return self.return_empty_scores_df("No predictions to score.", event.event_id)

        interval_scores_df = self.get_interval_scores_base(
            predictions_df=predictions_df, miners=miners, intervals=intervals
        )

        failed_runs = await self.db_operations.get_failed_agent_runs_for_event(
            event_id=event.unique_event_id
        )

        # Fill missing predictions with 0.5 for miners whose code failed, then drop truly-missing
        interval_scores_df = self.fill_unresponsive_miners(
            interval_scores_df, failed_runs=failed_runs, imputed_prediction=0.5
        )

        # Aggregate predictions by miner using weighted average
        scores_df = self.aggregate_predictions_by_miner(interval_scores_df)

        # Compute Brier score: (prediction - outcome)²
        current_outcome = float(outcome_round)
        scores_df[ScoreNames.rema_peer_score] = (
            scores_df[ScoreNames.rema_prediction] - current_outcome
        ) ** 2

        return scores_df

    async def export_scores_to_db(self, scores_df: pd.DataFrame, event_id: str):
        sanitized_scores = scores_df.copy()
        fill_values = {
            ScoreNames.miner_uid: -1,  # should not happen
            ScoreNames.miner_hotkey: "unknown",  # should not happen
            ScoreNames.rema_prediction: -998,  # marker for missing predictions
            ScoreNames.rema_peer_score: 0.0,
        }
        sanitized_scores.fillna(value=fill_values, inplace=True)

        records = sanitized_scores.to_dict(orient="records")
        scores = []
        for record in records:
            try:
                score = ScoresModel(
                    event_id=event_id,
                    miner_uid=record[ScoreNames.miner_uid],
                    miner_hotkey=record[ScoreNames.miner_hotkey],
                    prediction=record[ScoreNames.rema_prediction],
                    event_score=record[ScoreNames.rema_peer_score],
                    spec_version=self.spec_version,
                )
                scores.append(score)
            except Exception:
                self.errors_count += 1
                self.logger.error(
                    "Error while creating a score record.",
                    extra={"record": record},
                )

        if not scores:
            self.errors_count += 1
            self.logger.error("No scores to export.", extra={"event_id": event_id})
            return

        await self.db_operations.insert_scores(scores)

    async def run(self):
        async with self.subtensor_cm as subtensor:
            self.metagraph = await subtensor.metagraph(netuid=self.netuid, lite=True)

        self.copy_metagraph_state()

        miners_synced = await self.miners_last_reg_sync()
        if not miners_synced:
            return

        # TODO: do not score more than page_size=100 events at a time.
        events_to_score = await self.db_operations.get_events_for_scoring(
            # max_events=self.page_size
        )
        if not events_to_score:
            self.logger.debug("No events to calculate scores.")
        else:
            self.logger.debug(
                "Found events to calculate scores.", extra={"n_events": len(events_to_score)}
            )

            for event in events_to_score:
                unique_event_id = event.unique_event_id
                event_id = event.event_id
                event = self.set_right_cutoff(event)
                self.logger.debug(
                    "Calculating scores for an event.",
                    extra={
                        "event_id": event_id,
                        "event_registered_date": event.registered_date.isoformat(),
                        "event_cutoff": event.cutoff.isoformat(),
                        "event_resolved_at": event.resolved_at.isoformat(),
                    },
                )

                predictions = await self.db_operations.get_predictions_for_scoring(
                    unique_event_id=unique_event_id
                )
                if not predictions:
                    self.errors_count += 1
                    self.logger.error(
                        "There are no predictions for a settled event - discarding.",
                        extra={"event_id": event_id},
                    )
                    await self.db_operations.mark_event_as_discarded(
                        unique_event_id=unique_event_id
                    )
                    continue

                scores_df = await self.score_event(event, predictions)
                if scores_df.empty:
                    self.logger.error(
                        "Scores could not be calculated for an event.",
                        extra={"event_id": event_id},
                    )
                    continue
                else:
                    self.logger.debug(
                        "Scores calculated, sample below.",
                        extra={
                            "event_id": event_id,
                            "scores": scores_df.head(n=5).to_dict(orient="index"),
                            "len_scores": len(scores_df),
                        },
                    )

                await self.export_scores_to_db(scores_df, event_id)
                await self.db_operations.mark_event_as_processed(unique_event_id=unique_event_id)

        self.logger.debug(
            "Scoring run finished. Resetting errors count.",
            extra={"errors_count_in_logs": self.errors_count},
        )
        self.errors_count = 0
