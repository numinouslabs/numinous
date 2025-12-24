import copy
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
import numpy as np
import pandas as pd
from bittensor import AsyncSubtensor
from bittensor.utils.weight_utils import process_weights
from bittensor_wallet.wallet import Wallet

from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.numinous_client import GetWeightsResponse
from neurons.validator.models.weights import WeightsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.converters import pydantic_models_to_dataframe
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.if_metagraph import IfMetagraph
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version


# this is just for avoiding typos in column names
@dataclass
class SWNames:
    miner_uid: str = "miner_uid"
    miner_hotkey: str = "miner_hotkey"
    event_score: str = "event_score"
    metagraph_score: str = "metagraph_score"
    event_id: str = "event_id"
    spec_version_name: str = "spec_version"
    created_at: str = "created_at"
    raw_weights: str = "raw_weights"


class SetWeights(AbstractTask):
    interval: float
    db_operations: DatabaseOperations
    logger: NuminousLogger
    netuid: int
    subtensor: AsyncSubtensor
    wallet: Wallet  # type: ignore

    def __init__(
        self,
        interval_seconds: float,
        db_operations: DatabaseOperations,
        logger: NuminousLogger,
        metagraph: IfMetagraph,
        netuid: int,
        subtensor: AsyncSubtensor,
        wallet: Wallet,  # type: ignore
        api_client: NuminousClient,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(db_operations, DatabaseOperations):
            raise TypeError("db_operations must be an instance of DatabaseOperations.")

        if not isinstance(api_client, NuminousClient):
            raise TypeError("api_client must be an instance of NuminousClient.")

        self.interval = interval_seconds
        self.db_operations = db_operations
        self.logger = logger

        self.metagraph = metagraph
        self.netuid = netuid
        self.subtensor = subtensor
        self.wallet = wallet
        self.api_client = api_client

        self.current_hotkeys = None
        self.n_hotkeys = None
        self.current_uids = None
        self.current_miners_df = None

        self.last_set_weights_at = round(time.time())
        self.spec_version = spec_version

    @property
    def name(self):
        return "set-weights"

    @property
    def interval_seconds(self):
        return self.interval

    async def metagraph_lite_sync(self):
        # sync the metagraph
        await self.metagraph.sync()

        # hotkeys is list[str], uids is np.ndarray
        self.current_hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        self.n_hotkeys = len(self.current_hotkeys)
        self.current_uids = copy.deepcopy(self.metagraph.uids)
        self.current_miners_df = pd.DataFrame(
            {
                "miner_hotkey": self.current_hotkeys,
                "miner_uid": self.current_uids.tolist(),
            }
        )

    async def time_to_set_weights(self):
        weights_rate_limit = await self.subtensor.weights_rate_limit(netuid=self.netuid)

        # do not attempt to set weights more often than the rate limit
        blocks_since_last_attempt = (
            round(time.time()) - self.last_set_weights_at
        ) // BLOCK_DURATION

        last_set_weights_at_dt = datetime.fromtimestamp(
            self.last_set_weights_at, tz=timezone.utc
        ).isoformat(timespec="seconds")

        if blocks_since_last_attempt < weights_rate_limit:
            self.logger.debug(
                "Not setting the weights - not enough blocks passed.",
                extra={
                    "last_set_weights_at": last_set_weights_at_dt,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": weights_rate_limit,
                },
            )
            return False
        else:
            self.logger.debug(
                "Attempting to set the weights - enough blocks passed.",
                extra={
                    "last_set_weights_at": last_set_weights_at_dt,
                    "blocks_since_last_attempt": blocks_since_last_attempt,
                    "weights_rate_limit": weights_rate_limit,
                },
            )
            # reset the last set weights time here to avoid attempts rate limit
            self.last_set_weights_at = round(time.time())

            return True

    def merge_weights_with_metagraph(self, weights_from_api) -> pd.DataFrame:
        merged_weights = pydantic_models_to_dataframe(weights_from_api)
        merged_weights = pd.merge(
            self.current_miners_df,
            merged_weights,
            on=[SWNames.miner_uid, SWNames.miner_hotkey],
            how="left",
        )

        stats = {
            "len_weights_from_api": len(weights_from_api),
            "len_merged_weights": len(merged_weights),
            "len_current_miners": len(self.current_miners_df),
            "len_non_zero_weights": int((merged_weights[SWNames.metagraph_score].notna()).sum()),
        }
        self.logger.debug("Merged API weights with current metagraph", extra=stats)

        merged_weights = merged_weights[
            [SWNames.miner_uid, SWNames.miner_hotkey, SWNames.metagraph_score]
        ]
        data_types = {
            SWNames.miner_uid: "int",
            SWNames.miner_hotkey: "str",
            SWNames.metagraph_score: "float",
        }
        merged_weights = merged_weights.astype(data_types)
        merged_weights[SWNames.metagraph_score] = merged_weights[SWNames.metagraph_score].fillna(
            0.0
        )

        return merged_weights

    def check_scores_sanity(self, filtered_scores: pd.DataFrame) -> bool:
        # Do some sanity checks before and throw assert exceptions if there are issues

        # we should have unique miner_uids
        assert filtered_scores[
            SWNames.miner_uid
        ].is_unique, (
            f"miner_uids are not unique: {filtered_scores[SWNames.miner_uid].value_counts()[:5]}"
        )
        # we should have the same miner_uids as the current metagraph
        assert set(filtered_scores[SWNames.miner_uid]) == set(
            self.current_miners_df[SWNames.miner_uid]
        ), "The miner_uids are not the same as the current metagraph"

        # metagraph scores should not be all 0.0
        assert (
            not filtered_scores[SWNames.metagraph_score].eq(0.0).all()
        ), "All metagraph_scores are 0.0. This is not expected."

        # metagraph scores should not sum up to 0.0
        # redundant, they are positive and not all 0.0
        assert (
            not filtered_scores[SWNames.metagraph_score].sum() == 0.0
        ), "The sum of metagraph_scores is 0.0."

        # no NaNs/nulls in filtered_scores in any column
        assert not filtered_scores.isnull().values.any(), "There are NaNs in filtered_scores."

        return True

    def renormalize_weights(self, filtered_scores: pd.DataFrame) -> pd.DataFrame:
        # this is re-normalizing the weights for the current miners
        normalized_scores = filtered_scores.copy()

        # normalize the metagraph scores - guaranteed that sum is strictly positive
        normalized_scores[SWNames.raw_weights] = normalized_scores[SWNames.metagraph_score].div(
            normalized_scores[SWNames.metagraph_score].sum()
        )

        # for debug, log top 5 and bottom 5 miners by raw_weights
        top_5 = normalized_scores.nlargest(5, SWNames.raw_weights)
        bottom_5 = normalized_scores.nsmallest(5, SWNames.raw_weights)
        self.logger.debug(
            "Top 5 and bottom 5 miners by raw_weights",
            extra={
                "top_5": top_5.to_dict(),
                "bottom_5": bottom_5.to_dict(),
                "sum_scores": normalized_scores[SWNames.metagraph_score].sum(),
            },
        )

        return normalized_scores

    async def preprocess_weights(
        self, normalized_scores: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        miner_uids = np.array(normalized_scores[SWNames.miner_uid].values, dtype=np.int64)
        raw_weights = np.array(normalized_scores[SWNames.raw_weights].values, dtype=np.float32)

        min_allowed_weights = await self.subtensor.min_allowed_weights(netuid=self.netuid)
        max_weight_limit = await self.subtensor.max_weight_limit(netuid=self.netuid)

        processed_uids, processed_weights = process_weights(
            uids=miner_uids,
            weights=raw_weights,
            num_neurons=self.metagraph.n,
            min_allowed_weights=min_allowed_weights,
            max_weight_limit=max_weight_limit,
        )

        if (
            processed_uids is None
            or processed_weights is None
            or len(processed_uids) == 0
            or len(processed_weights) == 0
        ):
            self.logger.error(
                "Failed to process the weights - received None or empty tensors.",
                extra={
                    "processed_uids[:10]": (
                        processed_uids.tolist()[:10] if processed_uids is not None else None
                    ),
                    "processed_weights[:10]": (
                        processed_weights.tolist()[:10] if processed_weights is not None else None
                    ),
                },
            )
            raise ValueError("Failed to process the weights - received None or empty tensors.")

        # process_weights excludes the zero weights
        mask = raw_weights != 0
        if not np.array_equal(processed_uids, miner_uids[mask]):
            self.logger.error(
                "Processed UIDs do not match the original UIDs.",
                extra={
                    "processed_uids[:10]": processed_uids.tolist()[:10],
                    "original_uids[:10]": miner_uids.tolist()[:10],
                },
            )
            raise ValueError("Processed UIDs do not match the original UIDs.")

        if not np.allclose(processed_weights, raw_weights[mask], atol=1e-5, rtol=1e-5):
            self.logger.warning(
                "Processed weights do not match the original weights.",
                extra={
                    "processed_weights[:10]": [
                        round(w, 5) for w in processed_weights.tolist()[:10]
                    ],
                    "original_weights[:10]": [round(w, 5) for w in raw_weights.tolist()[:10]],
                },
            )

        return processed_uids, processed_weights

    async def subtensor_set_weights(
        self, processed_uids: np.ndarray, processed_weights: np.ndarray
    ):
        async with self.subtensor as subtensor:
            response = await subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=processed_uids,
                weights=processed_weights,
                version_key=self.spec_version,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                wait_for_revealed_execution=False,
                max_attempts=2,
                raise_error=True,
            )

            if not response.success:
                extra = {
                    "fail_msg": response.message,
                    "processed_uids[:10]": processed_uids.tolist()[:10],
                    "processed_weights[:10]": processed_weights.tolist()[:10],
                    "exception": f"{type(response.error).__name__}: {response.error}",
                }
                log_msg = "Failed to set the weights."
                if "No attempt made" in response.message:
                    # do not consider this as an error - pollutes the logs
                    self.logger.warning(log_msg, extra=extra)
                else:
                    self.logger.error(log_msg, extra=extra)
            else:
                self.logger.debug(
                    "Weights set successfully.",
                    extra={
                        "last_set_weights_at": datetime.fromtimestamp(
                            self.last_set_weights_at, tz=timezone.utc
                        ).isoformat(timespec="seconds")
                    },
                )

    def get_owner_neuron(self):
        owner_uid = None
        owner_hotkey = self.metagraph.owner_hotkey

        for idx, uid in enumerate(self.metagraph.uids):
            int_uid = int(uid)
            hotkey = self.metagraph.hotkeys[idx]

            if hotkey == owner_hotkey:
                owner_uid = int_uid
                break

        assert owner_uid is not None, "Owner uid not found in metagraph uids"

        return {"uid": owner_uid, "hotkey": owner_hotkey}

    def _convert_api_weights_to_weights(
        self, api_response: GetWeightsResponse
    ) -> list[WeightsModel]:
        weights = []
        for weight in api_response.weights:
            weights.append(
                WeightsModel(
                    miner_uid=weight.miner_uid,
                    miner_hotkey=weight.miner_hotkey,
                    metagraph_score=weight.aggregated_weight,
                    aggregated_at=api_response.aggregated_at,
                )
            )

        self.logger.debug(
            "Converted API response to weights",
            extra={
                "num_weights": len(weights),
                "aggregated_at": api_response.aggregated_at.isoformat(),
            },
        )

        return weights

    async def run(self):
        await self.metagraph_lite_sync()

        can_set_weights = await self.time_to_set_weights()
        if not can_set_weights:
            return

        try:
            api_response = await self.api_client.get_weights()

            self.logger.info(
                "Fetched centralized weights from API",
                extra={
                    "aggregated_at": api_response.aggregated_at.isoformat(),
                    "num_weights": len(api_response.weights),
                    "count": api_response.count,
                },
            )

            weights_from_api = self._convert_api_weights_to_weights(api_response)

        except aiohttp.ClientResponseError as e:
            if e.status == 503:
                self.logger.warning(
                    "Backend has no weights available yet (503). Skipping set_weights this round.",
                    extra={"status": e.status},
                )
                return
            else:
                self.logger.error(
                    "Failed to fetch weights from backend API",
                    extra={"status": e.status, "message": str(e)},
                )
                raise

        except Exception as e:
            self.logger.exception(
                "Unexpected error fetching weights from API",
                extra={"error_type": type(e).__name__},
            )
            raise

        if not weights_from_api:
            raise ValueError("Failed to get weights from API (empty response).")

        merged_weights = self.merge_weights_with_metagraph(weights_from_api)

        self.check_scores_sanity(merged_weights)

        normalized_scores = self.renormalize_weights(merged_weights)

        uids, weights = await self.preprocess_weights(normalized_scores)

        await self.subtensor_set_weights(processed_uids=uids, processed_weights=weights)
