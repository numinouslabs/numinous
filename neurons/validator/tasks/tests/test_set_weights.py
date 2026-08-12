import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock

import aiohttp
import numpy as np
import pandas as pd
import pytest
from bittensor import ChainError, ErrorCode, ExtrinsicResult
from bittensor import SetWeights as SetWeightsIntent
from bittensor import Wallet
from freezegun import freeze_time

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.models.numinous_client import GetWeightsResponse, MinerWeight
from neurons.validator.models.weights import WeightsModel
from neurons.validator.numinous_client.client import NuminousClient
from neurons.validator.tasks.set_weights import SetWeights, SWNames
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version


class TestSetWeights:
    @pytest.fixture
    def db_operations(self, db_client: DatabaseClient):
        logger = MagicMock(spec=NuminousLogger)

        return DatabaseOperations(db_client=db_client, logger=logger)

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "hotkey2"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)
        bt_wallet.hotkey.ss58_address = "hotkey2"

        return bt_wallet

    def _neuron(self, uid: int, hotkey: str):
        neuron = MagicMock()
        neuron.uid = uid
        neuron.hotkey = hotkey
        return neuron

    def _metagraph(self, uids: list[int], hotkeys: list[str]):
        metagraph = MagicMock()
        metagraph.neurons = [self._neuron(uid, hotkey) for uid, hotkey in zip(uids, hotkeys)]
        metagraph.hotkeys = hotkeys
        metagraph.num_uids = len(uids)
        return metagraph

    @pytest.fixture
    def mock_subtensor(self):
        metagraph = self._metagraph([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])

        chain_client = AsyncMock()
        chain_client.subnets.metagraph = AsyncMock(return_value=metagraph)
        chain_client.hyperparameters.weights_rate_limit = AsyncMock(return_value=100)
        chain_client.hyperparameters.min_allowed_weights = AsyncMock(return_value=1)
        # raw u16; task divides by U16_MAX -> 10.0, matching the old mocked limit
        chain_client.hyperparameters.max_weight_limit = AsyncMock(return_value=655350)
        chain_client.execute = AsyncMock(return_value=ExtrinsicResult(success=True, message="ok"))

        subtensor = MagicMock()
        subtensor.return_value.__aenter__ = AsyncMock(return_value=chain_client)
        subtensor.return_value.__aexit__ = AsyncMock(return_value=False)
        subtensor.chain_client = chain_client

        return subtensor

    @pytest.fixture
    def set_weights_task(
        self,
        db_operations: DatabaseOperations,
        bt_wallet: Wallet,
        mock_subtensor: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # run() builds a fresh Subtensor(network) each tick; patch the constructor so it
        # returns our mock connection instead of opening a real websocket.
        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", mock_subtensor)

        api_client = MagicMock(spec=NuminousClient)

        logger = MagicMock(spec=NuminousLogger)

        with freeze_time("2024-12-27 07:00:00"):
            return SetWeights(
                interval_seconds=60.0,
                db_operations=db_operations,
                logger=logger,
                netuid=155,
                network="test",
                wallet=bt_wallet,
                api_client=api_client,
            )

    def test_init(self, set_weights_task: SetWeights):
        unit = set_weights_task

        assert isinstance(unit, SetWeights)

        assert unit.interval_seconds == 60.0
        assert unit.db_operations is not None
        assert unit.logger is not None
        assert unit.netuid == 155
        assert unit.network == "test"
        assert unit.wallet is not None
        assert unit.api_client is not None

        assert unit.spec_version == spec_version

    def test_copy_metagraph_state(self, set_weights_task: SetWeights):
        unit = set_weights_task

        unit.metagraph = self._metagraph([1, 2, 3, 4], ["hotkey1", "hotkey2", "hotkey3", "hotkey4"])

        unit.copy_metagraph_state()

        assert unit.current_miners_df.miner_uid.tolist() == [1, 2, 3, 4]
        assert unit.current_miners_df.miner_hotkey.tolist() == [
            "hotkey1",
            "hotkey2",
            "hotkey3",
            "hotkey4",
        ]
        assert unit.current_miners_df.index.size == 4
        assert unit.current_hotkeys == ["hotkey1", "hotkey2", "hotkey3", "hotkey4"]
        assert unit.current_uids == [1, 2, 3, 4]
        assert unit.n_hotkeys == 4

    @pytest.mark.parametrize(
        "delta,expected",
        [
            (99 * BLOCK_DURATION, False),
            (101 * BLOCK_DURATION, True),
        ],
    )
    async def test_time_to_set_weights(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock, delta, expected
    ):
        # Set internal instances
        set_weights_task.chain_client = mock_subtensor.chain_client

        now = time.time()
        set_weights_task.last_set_weights_at = now - delta

        result = await set_weights_task.time_to_set_weights()

        assert result is expected

    async def test_merge_weights_with_metagraph(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock
    ):
        # Set internal instances
        set_weights_task.chain_client = mock_subtensor.chain_client
        set_weights_task.metagraph = await mock_subtensor.chain_client.subnets.metagraph(
            set_weights_task.netuid
        )

        unit = set_weights_task

        unit.copy_metagraph_state()

        weights_from_api = [
            WeightsModel(
                miner_uid=1,
                miner_hotkey="hotkey1",
                metagraph_score=0.8,
                aggregated_at=datetime.now(timezone.utc) - timedelta(minutes=3),
            ),
            WeightsModel(
                miner_uid=4,
                miner_hotkey="hotkey4",
                metagraph_score=0.5,
                aggregated_at=datetime.now(timezone.utc),
            ),
        ]

        merged_weights = unit.merge_weights_with_metagraph(weights_from_api)

        assert len(merged_weights) == 3
        assert merged_weights.loc[0].miner_uid == 1
        assert merged_weights.loc[1].miner_uid == 2
        assert merged_weights.loc[2].miner_uid == 3
        assert merged_weights.loc[0].miner_hotkey == "hotkey1"
        assert merged_weights.loc[1].miner_hotkey == "hotkey2"
        assert merged_weights.loc[2].miner_hotkey == "hotkey3"
        assert merged_weights.loc[0].metagraph_score == 0.8
        assert merged_weights.loc[1].metagraph_score == 0.0
        assert merged_weights.loc[2].metagraph_score == 0.0

        expected_stats = {
            "len_weights_from_api": 2,
            "len_merged_weights": 3,
            "len_current_miners": 3,
            "len_non_zero_weights": 1,
        }

        assert unit.logger.debug.call_count == 1
        assert unit.logger.debug.call_args[0][0] == "Merged API weights with current metagraph"
        assert unit.logger.debug.call_args[1]["extra"] == expected_stats

    @pytest.mark.parametrize(
        "data,raises",
        [
            # Valid case: unique miner_uid and non-zero metagraph scores.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                False,
            ),
            # Duplicate miner_uid: should raise assertion.
            (
                {
                    SWNames.miner_uid: [1, 1, 2],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey11111", "hotkey2"],
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
            # All metagraph_score 0: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
                    SWNames.metagraph_score: [0.0, 0.0, 0.0],
                },
                True,
            ),
            # Missing miner_hotkey: should raise.
            (
                {
                    SWNames.miner_uid: [1, 2, 3],
                    SWNames.miner_hotkey: ["hotkey1", None, "hotkey3"],
                    SWNames.metagraph_score: [0.8, 0.9, 1.0],
                },
                True,
            ),
        ],
    )
    async def test_check_scores_insanity(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock, data, raises
    ):
        # Set internal instances
        set_weights_task.chain_client = mock_subtensor.chain_client
        set_weights_task.metagraph = await mock_subtensor.chain_client.subnets.metagraph(
            set_weights_task.netuid
        )

        # Need to sync to load task instance data
        set_weights_task.copy_metagraph_state()

        df = pd.DataFrame(data)

        if raises:
            with pytest.raises(AssertionError):
                set_weights_task.check_scores_sanity(df)
        else:
            assert set_weights_task.check_scores_sanity(df) is True

    def test_renormalize_weights(self, set_weights_task: SetWeights):
        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.metagraph_score: [0.8, 0.9, 1.3],
        }
        df = pd.DataFrame(data)
        normalized = set_weights_task.renormalize_weights(df)

        assert SWNames.raw_weights in normalized.columns
        total = sum(data[SWNames.metagraph_score])
        expected = [score / total for score in data[SWNames.metagraph_score]]

        pd.testing.assert_series_equal(
            normalized[SWNames.raw_weights],
            pd.Series(expected, name=SWNames.raw_weights),
        )

        set_weights_task.logger.debug.assert_called()

        assert (
            set_weights_task.logger.debug.call_args[0][0]
            == "Top 5 and bottom 5 miners by raw_weights"
        )
        assert set_weights_task.logger.debug.call_args[1]["extra"]["sum_scores"] == total

    async def test_preprocess_weights_success(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock, monkeypatch
    ):
        # Set internal instances
        set_weights_task.chain_client = mock_subtensor.chain_client
        set_weights_task.metagraph = await mock_subtensor.chain_client.subnets.metagraph(
            set_weights_task.netuid
        )

        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.5],
        }
        normalized = pd.DataFrame(data)

        uids, weights = await set_weights_task.preprocess_weights(normalized)

        # Expect the same non-zero entries.
        np.testing.assert_allclose(weights, np.array(data[SWNames.raw_weights], dtype=np.float32))
        np.testing.assert_array_equal(uids, np.array(data[SWNames.miner_uid], dtype=np.int64))

    async def test_preprocess_weights_edge_cases(
        self, set_weights_task, mock_subtensor: AsyncMock, monkeypatch
    ):
        # Set internal instances
        set_weights_task.chain_client = mock_subtensor.chain_client
        set_weights_task.metagraph = await mock_subtensor.chain_client.subnets.metagraph(
            set_weights_task.netuid
        )

        data = {
            SWNames.miner_uid: [1, 2, 3],
            SWNames.miner_hotkey: ["hotkey1", "hotkey2", "hotkey3"],
            SWNames.raw_weights: [0.2, 0.3, 0.0],
        }
        normalized = pd.DataFrame(data)

        uids, weights = await set_weights_task.preprocess_weights(normalized)
        np.testing.assert_allclose(weights, np.array([0.4, 0.6], dtype=np.float32))
        np.testing.assert_array_equal(uids, np.array([1, 2], dtype=np.int64))

        # Weights get normalized
        assert set_weights_task.logger.warning.call_count == 1
        assert (
            set_weights_task.logger.warning.call_args[0][0]
            == "Processed weights do not match the original weights."
        )

        set_weights_task.logger.warning.reset_mock()

        # Force return empty tensors
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float32),
            ),
        )

        with pytest.raises(
            expected_exception=ValueError,
            match="Failed to process the weights - received None or empty tensors.",
        ):
            await set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Failed to process the weights - received None or empty tensors."
        )

        set_weights_task.logger.error.reset_mock()

        # Force return different UIDs
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (np.array([1, 2, 4], dtype=np.int64), weights),
        )
        with pytest.raises(
            expected_exception=ValueError, match="Processed UIDs do not match the original UIDs."
        ):
            await set_weights_task.preprocess_weights(normalized)

        assert set_weights_task.logger.error.call_count == 1
        assert (
            set_weights_task.logger.error.call_args[0][0]
            == "Processed UIDs do not match the original UIDs."
        )

        set_weights_task.logger.error.reset_mock()

        # Force return different weights
        monkeypatch.setattr(
            "neurons.validator.tasks.set_weights.process_weights",
            lambda uids, weights, **kwargs: (uids, np.array([0.2, 0.3, 0.4], dtype=np.float32)),
        )

        normalized.loc[2, SWNames.raw_weights] = 0.41
        uids, weights = await set_weights_task.preprocess_weights(normalized)

        np.testing.assert_array_equal(uids, np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_allclose(weights, np.array([0.2, 0.3, 0.4], dtype=np.float32))

        assert set_weights_task.logger.warning.call_count == 1
        assert set_weights_task.logger.error.call_count == 0
        assert (
            set_weights_task.logger.warning.call_args[0][0]
            == "Processed weights do not match the original weights."
        )

    @pytest.mark.parametrize(
        "success,message,error_code,expected_log",
        [
            (True, "Success", None, "Weights set successfully."),
            (
                False,
                "Rate limited",
                "RATE_LIMITED",
                "Failed to set the weights.",
            ),  # warning case
            (
                False,
                "Other failure",
                "OTHER",
                "Failed to set the weights.",
            ),  # error case
        ],
    )
    async def test_subtensor_set_weights(
        self, set_weights_task: SetWeights, success, message, error_code, expected_log
    ):
        processed_uids = np.array([1, 2, 3], dtype=np.int64)
        processed_weights = np.array([0.2, 0.3, 0.5], dtype=np.float32)

        error = None
        if not success:
            error = MagicMock(spec=ChainError)
            error.code = ErrorCode.RATE_LIMITED if error_code == "RATE_LIMITED" else MagicMock()

        chain_client = AsyncMock()
        chain_client.execute = AsyncMock(
            return_value=ExtrinsicResult(success=success, message=message, error=error)
        )

        set_weights_task.chain_client = chain_client

        set_weights_task.logger.debug.reset_mock()
        set_weights_task.logger.warning.reset_mock()
        set_weights_task.logger.error.reset_mock()

        await set_weights_task.subtensor_set_weights(processed_uids, processed_weights)

        expected_intent = SetWeightsIntent(
            netuid=set_weights_task.netuid,
            weights={1: 0.2, 2: 0.3, 3: 0.5},
            version_key=set_weights_task.spec_version,
        )

        chain_client.execute.assert_awaited_once()
        called_intent, called_wallet = chain_client.execute.await_args.args
        assert called_intent.netuid == expected_intent.netuid
        assert called_intent.version_key == expected_intent.version_key
        assert called_intent.weights == pytest.approx(expected_intent.weights)
        assert called_wallet == set_weights_task.wallet
        assert chain_client.execute.await_args.kwargs == {
            "wait_for_inclusion": True,
            "wait_for_finalization": True,
            "retries": 2,
        }

        if success:
            set_weights_task.logger.debug.assert_called_with(
                expected_log,
                extra={"last_set_weights_at": ANY},
            )
        else:
            extra = {
                "fail_msg": message,
                "processed_uids[:10]": [1, 2, 3],
                "processed_weights[:10]": ANY,
                "exception": ANY,
            }
            if error_code == "RATE_LIMITED":
                set_weights_task.logger.warning.assert_called_with(expected_log, extra=extra)
            else:
                set_weights_task.logger.error.assert_called_with(expected_log, extra=extra)

    def test_get_owner_neuron(self, set_weights_task: SetWeights):
        unit = set_weights_task

        unit.metagraph = self._metagraph([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])
        unit.metagraph.owner_hotkey = "hotkey1"

        owner = unit.get_owner_neuron()

        assert owner["uid"] == 1
        assert owner["hotkey"] == "hotkey1"

    def test_get_owner_neuron_not_found(self, set_weights_task: SetWeights):
        unit = set_weights_task

        unit.metagraph = self._metagraph([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])
        unit.metagraph.owner_hotkey = "hotkey_not_in_metagraph"

        with pytest.raises(AssertionError, match="Owner uid not found in metagraph uids"):
            unit.get_owner_neuron()

    def test_convert_api_weights_to_weights(self, set_weights_task: SetWeights):
        unit = set_weights_task

        api_response = GetWeightsResponse(
            aggregated_at=datetime(2025, 1, 30, 12, 0, 0, tzinfo=timezone.utc),
            weights=[
                MinerWeight(miner_uid=1, miner_hotkey="hk1", aggregated_weight=0.6),
                MinerWeight(miner_uid=2, miner_hotkey="hk2", aggregated_weight=0.4),
            ],
            count=2,
        )

        result = unit._convert_api_weights_to_weights(api_response)

        assert len(result) == 2
        assert isinstance(result[0], WeightsModel)
        assert result[0].miner_uid == 1
        assert result[0].miner_hotkey == "hk1"
        assert result[0].metagraph_score == 0.6
        assert result[0].aggregated_at == datetime(2025, 1, 30, 12, 0, 0, tzinfo=timezone.utc)

        assert result[1].miner_uid == 2
        assert result[1].miner_hotkey == "hk2"
        assert result[1].metagraph_score == 0.4

    async def test_run_successful_x(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock, monkeypatch, db_client
    ):
        unit = set_weights_task

        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

        created_at = datetime.now(timezone.utc) - timedelta(days=1)

        mock_api_response = GetWeightsResponse(
            aggregated_at=created_at,
            weights=[
                MinerWeight(miner_uid=3, miner_hotkey="hk3", aggregated_weight=0.835),
                MinerWeight(miner_uid=4, miner_hotkey="hk4", aggregated_weight=0.165),
            ],
            count=2,
        )

        metagraph = self._metagraph([0, 1, 2, 3, 4], ["hk0", "hk1", "hk2", "hk3", "hk4"])
        metagraph.owner_hotkey = "hk3"

        mock_subtensor.chain_client.execute = AsyncMock(
            return_value=ExtrinsicResult(success=True, message="Weights set")
        )
        mock_subtensor.chain_client.subnets.metagraph = AsyncMock(return_value=metagraph)

        unit.api_client.get_weights = AsyncMock(return_value=mock_api_response)

        await unit.run()

        assert unit.logger.debug.call_count == 8
        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 0

        debug_calls = unit.logger.debug.call_args_list

        assert debug_calls[0][0][0] == "Weights rate limit"
        assert debug_calls[0][1]["extra"]["step"] == "fetching"
        assert debug_calls[1][0][0] == "Weights rate limit"
        assert debug_calls[1][1]["extra"]["step"] == "fetched"

        assert debug_calls[2][0][0] == "Attempting to set the weights - enough blocks passed."
        assert debug_calls[2][1]["extra"]["blocks_since_last_attempt"] >= 100

        assert debug_calls[3][0][0] == "Fetched centralized weights from API"
        assert debug_calls[4][0][0] == "Converted API response to weights"
        assert debug_calls[5][0][0] == "Merged API weights with current metagraph"
        assert debug_calls[6][0][0] == "Top 5 and bottom 5 miners by raw_weights"
        assert debug_calls[7][0][0] == "Weights set successfully."

        assert mock_subtensor.chain_client.execute.await_count == 1
        submitted_intent = mock_subtensor.chain_client.execute.await_args.args[0]
        assert submitted_intent.uids == [3, 4]
        assert submitted_intent.weights == pytest.approx([0.835, 0.165], rel=1e-5)

        # Check subtensor and metagraph context manager setup
        mock_subtensor.return_value.__aenter__.assert_awaited_once()
        mock_subtensor.return_value.__aexit__.assert_awaited_once()
        mock_subtensor.chain_client.subnets.metagraph.assert_awaited_once_with(unit.netuid)

    async def test_run_with_api_weights(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock
    ):
        unit = set_weights_task

        mock_api_response = GetWeightsResponse(
            aggregated_at=datetime(2025, 1, 30, 12, 0, 0, tzinfo=timezone.utc),
            weights=[
                MinerWeight(miner_uid=1, miner_hotkey="hotkey1", aggregated_weight=0.5),
                MinerWeight(miner_uid=2, miner_hotkey="hotkey2", aggregated_weight=0.3),
                MinerWeight(miner_uid=3, miner_hotkey="hotkey3", aggregated_weight=0.2),
            ],
            count=3,
        )

        mock_subtensor.chain_client.execute = AsyncMock(
            return_value=ExtrinsicResult(success=True, message="Success")
        )

        unit.api_client.get_weights = AsyncMock(return_value=mock_api_response)

        await unit.run()

        unit.api_client.get_weights.assert_called_once()
        mock_subtensor.chain_client.execute.assert_called_once()

    async def test_run_raises_http_errors(self, set_weights_task: SetWeights):
        unit = set_weights_task

        error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Internal Server Error",
        )
        unit.api_client.get_weights = AsyncMock(side_effect=error)

        with pytest.raises(aiohttp.ClientResponseError):
            await unit.run()

    async def test_run_set_weights_timeout(
        self, set_weights_task: SetWeights, mock_subtensor: AsyncMock
    ):
        unit = set_weights_task

        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

        mock_api_response = GetWeightsResponse(
            aggregated_at=datetime.now(timezone.utc) - timedelta(days=1),
            weights=[
                MinerWeight(miner_uid=1, miner_hotkey="hotkey1", aggregated_weight=0.835),
                MinerWeight(miner_uid=3, miner_hotkey="hotkey3", aggregated_weight=0.165),
            ],
            count=2,
        )
        unit.api_client.get_weights = AsyncMock(return_value=mock_api_response)

        async def execute_mock(*args, **kwargs):
            await asyncio.sleep(2)

            return ExtrinsicResult(success=True, message="Message")

        mock_subtensor.chain_client.execute = AsyncMock(side_effect=execute_mock)

        unit.timeout_seconds = 0.01

        with pytest.raises(asyncio.TimeoutError):
            await unit.run()

        mock_subtensor.return_value.__aexit__.assert_awaited_once()

    def _build_connection(self, execute_behaviour):
        metagraph = self._metagraph([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])

        chain_client = AsyncMock()
        chain_client.subnets.metagraph = AsyncMock(return_value=metagraph)
        chain_client.hyperparameters.weights_rate_limit = AsyncMock(return_value=100)
        chain_client.hyperparameters.min_allowed_weights = AsyncMock(return_value=1)
        chain_client.hyperparameters.max_weight_limit = AsyncMock(return_value=655350)
        chain_client.execute = AsyncMock(side_effect=execute_behaviour)

        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=chain_client)
        connection.__aexit__ = AsyncMock(return_value=False)
        connection.chain_client = chain_client
        return connection

    @staticmethod
    def _weights_api_response() -> GetWeightsResponse:
        return GetWeightsResponse(
            aggregated_at=datetime.now(timezone.utc) - timedelta(days=1),
            weights=[
                MinerWeight(miner_uid=1, miner_hotkey="hotkey1", aggregated_weight=0.5),
                MinerWeight(miner_uid=2, miner_hotkey="hotkey2", aggregated_weight=0.5),
            ],
            count=2,
        )

    async def test_run_builds_fresh_subtensor_each_tick(
        self, set_weights_task: SetWeights, monkeypatch
    ):
        unit = set_weights_task

        constructed = []

        def subtensor_factory(network):
            connection = self._build_connection(
                lambda *args, **kwargs: ExtrinsicResult(success=True, message="ok")
            )
            constructed.append(connection)
            return connection

        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", subtensor_factory)

        unit.api_client.get_weights = AsyncMock(return_value=self._weights_api_response())

        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        await unit.run()

        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        await unit.run()

        # Two ticks -> two distinct connections, neither reused.
        assert len(constructed) == 2
        assert constructed[0] is not constructed[1]
        constructed[0].chain_client.execute.assert_awaited_once()
        constructed[1].chain_client.execute.assert_awaited_once()

    @pytest.mark.parametrize(
        "wedge_error",
        [
            # Subscription wedge
            ChainError(
                "Websocket sending exception: Unable to reconnect because there are "
                "currently open subscriptions."
            ),
            # Stale-nonce wedge
            ChainError(
                "{'code': 1010, 'message': 'Invalid Transaction', 'data': 'Transaction is outdated'}"
            ),
        ],
    )
    async def test_run_recovers_from_wedged_connection_next_tick(
        self, set_weights_task: SetWeights, monkeypatch, wedge_error
    ):
        unit = set_weights_task

        wedged_connection = self._build_connection(wedge_error)
        healthy_connection = self._build_connection(
            lambda *args, **kwargs: ExtrinsicResult(success=True, message="ok")
        )

        connections = iter([wedged_connection, healthy_connection])
        constructed = []

        def subtensor_factory(network):
            connection = next(connections)
            constructed.append(connection)
            return connection

        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", subtensor_factory)

        unit.api_client.get_weights = AsyncMock(return_value=self._weights_api_response())

        # First tick: wedged connection -> extrinsic rejected, error propagates.
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        with pytest.raises(ChainError):
            await unit.run()

        # Next tick: fresh connection with no inherited state -> recovers automatically.
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        await unit.run()

        assert constructed == [wedged_connection, healthy_connection]
        wedged_connection.chain_client.execute.assert_awaited_once()
        healthy_connection.chain_client.execute.assert_awaited_once()
