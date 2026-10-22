import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import numpy as np
from bittensor import ErrorCode, Metagraph
from bittensor import SetWeights as SetWeightsIntent
from bittensor import Subtensor, Wallet

from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import DkgScoreModel
from neurons.validator.scheduler.task import AbstractTask
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.common.weights import U16_MAX, process_weights
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version

SCORES_IN_EFFECT_AFTER = timedelta(hours=1)
TOP_MINERS = 25
WEIGHT_EXPONENT = 2
ORDERING_SHARE = 0.02


class _RegisteredMiner(NamedTuple):
    uid: int
    hotkey: str
    block_at_registration: int


def _registered_miners(metagraph: Metagraph) -> list[_RegisteredMiner]:
    return [
        _RegisteredMiner(neuron.uid, neuron.hotkey, neuron.block_at_registration)
        for neuron in metagraph.neurons
    ]


def _top_weights(uids: np.ndarray, weights: np.ndarray) -> list[tuple[int, float]]:
    order = np.argsort(-weights, kind="stable")[:5]
    return [(int(uids[index]), round(float(weights[index]), 5)) for index in order]


def _owner_uid(metagraph: Metagraph) -> int:
    for neuron in metagraph.neurons:
        if neuron.hotkey == metagraph.owner_hotkey:
            return neuron.uid

    raise ValueError("The subnet owner is not registered in the metagraph.")


def _weights_from_scores(
    miners: list[_RegisteredMiner], scores: list[DkgScoreModel], owner_uid: int
) -> dict[int, float]:
    score_of = {(score.miner_uid, score.miner_hotkey): score.score for score in scores}
    scored = [miner for miner in miners if score_of.get((miner.uid, miner.hotkey), 0.0) > 0]
    ranked = sorted(
        scored,
        key=lambda miner: (
            -score_of[(miner.uid, miner.hotkey)],
            -miner.block_at_registration,
            miner.uid,
        ),
    )

    if not ranked:
        return {owner_uid: 1.0}

    powered = {
        miner.uid: score_of[(miner.uid, miner.hotkey)] ** WEIGHT_EXPONENT
        for miner in ranked[:TOP_MINERS]
    }
    powered_total = sum(powered.values())
    rank_total = sum(1 / rank for rank in range(1, len(ranked) + 1))
    weights = {
        miner.uid: ORDERING_SHARE / rank / rank_total
        + (1 - ORDERING_SHARE) * powered.get(miner.uid, 0.0) / powered_total
        for rank, miner in enumerate(ranked, start=1)
    }
    return dict(sorted(weights.items()))


class SetWeights(AbstractTask):
    interval: float
    dkg_operations: DkgDatabaseOperations
    logger: NuminousLogger
    netuid: int
    network: str
    wallet: Wallet
    timeout_seconds: float

    def __init__(
        self,
        interval_seconds: float,
        dkg_operations: DkgDatabaseOperations,
        logger: NuminousLogger,
        netuid: int,
        network: str,
        wallet: Wallet,
        timeout_seconds: float = 60.0 * 15,
    ):
        if not isinstance(interval_seconds, float) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive number (float).")

        if not isinstance(dkg_operations, DkgDatabaseOperations):
            raise TypeError("dkg_operations must be an instance of DkgDatabaseOperations.")

        if not isinstance(timeout_seconds, float) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number (float).")

        if not isinstance(netuid, int) or netuid < 0:
            raise ValueError("netuid must be a non-negative integer.")

        if not isinstance(network, str) or not network:
            raise ValueError("network must be a non-empty string.")

        self.interval = interval_seconds
        self.dkg_operations = dkg_operations
        self.logger = logger

        self.netuid = netuid
        self.network = network
        self.wallet = wallet
        self.timeout_seconds = timeout_seconds

        self.last_set_weights_at = round(time.time())
        self.spec_version = spec_version

    @property
    def name(self):
        return "set-weights"

    @property
    def interval_seconds(self):
        return self.interval

    async def time_to_set_weights(self):
        self.logger.debug("Weights rate limit", extra={"step": "fetching"})

        weights_rate_limit = await self.chain_client.hyperparameters.weights_rate_limit(self.netuid)

        self.logger.debug(
            "Weights rate limit",
            extra={"step": "fetched", "weights_rate_limit": weights_rate_limit},
        )

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

    async def preprocess_weights(
        self, weights: dict[int, float], num_uids: int
    ) -> tuple[np.ndarray, np.ndarray]:
        miner_uids = np.array(list(weights), dtype=np.int64)
        raw_weights = np.array(list(weights.values()), dtype=np.float32)

        min_allowed_weights = await self.chain_client.hyperparameters.min_allowed_weights(
            self.netuid
        )
        max_weight_limit_u16 = await self.chain_client.hyperparameters.max_weight_limit(self.netuid)
        max_weight_limit = max_weight_limit_u16 / U16_MAX

        processed_uids, processed_weights = process_weights(
            uids=miner_uids,
            weights=raw_weights,
            num_neurons=num_uids,
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

        self.logger.debug(
            "Weights processed for the chain",
            extra={
                "min_allowed_weights": min_allowed_weights,
                "max_weight_limit": max_weight_limit,
                "uids": len(miner_uids),
                "uids_with_a_weight": len(processed_uids),
                "top_before": _top_weights(miner_uids, raw_weights),
                "top_after": _top_weights(processed_uids, processed_weights),
            },
        )

        return processed_uids, processed_weights

    async def subtensor_set_weights(
        self,
        processed_uids: np.ndarray,
        processed_weights: np.ndarray,
    ):
        intent = SetWeightsIntent(
            netuid=self.netuid,
            weights={
                int(uid): float(weight) for uid, weight in zip(processed_uids, processed_weights)
            },
            version_key=self.spec_version,
        )

        response = await self.chain_client.execute(
            intent,
            self.wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            retries=2,
        )

        if not response.success:
            extra = {
                "fail_msg": response.message,
                "processed_uids[:10]": processed_uids.tolist()[:10],
                "processed_weights[:10]": processed_weights.tolist()[:10],
                "exception": f"{type(response.error).__name__}: {response.error}",
            }
            log_msg = "Failed to set the weights."

            if response.error is not None and response.error.code == ErrorCode.RATE_LIMITED:
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

    async def run(self):
        async with asyncio.timeout(delay=self.timeout_seconds):
            async with Subtensor(self.network) as chain_client:
                self.chain_client = chain_client

                if not await self.time_to_set_weights():
                    return

                scores = await self.dkg_operations.get_latest_scores(
                    scored_until=datetime.now(timezone.utc) - SCORES_IN_EFFECT_AFTER
                )

                if not scores:
                    self.logger.warning("No DKG scores in effect yet, weights not set.")
                    return

                metagraph = await chain_client.subnets.metagraph(self.netuid)
                owner_uid = _owner_uid(metagraph)
                weights = _weights_from_scores(_registered_miners(metagraph), scores, owner_uid)

                self.logger.info(
                    "Weights from DKG scores",
                    extra={
                        "scored_at": scores[0].scored_at.isoformat(),
                        "miners_with_a_weight": len(weights),
                        "all_to_owner": set(weights) == {owner_uid},
                    },
                )

                weight_of_every_uid = {
                    neuron.uid: weights.get(neuron.uid, 0.0) for neuron in metagraph.neurons
                }
                uids, processed_weights = await self.preprocess_weights(
                    weight_of_every_uid, metagraph.num_uids
                )

                await self.subtensor_set_weights(
                    processed_uids=uids, processed_weights=processed_weights
                )
