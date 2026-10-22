import asyncio
import random
import time
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, MagicMock

import numpy as np
import pytest
from bittensor import ChainError, ErrorCode, ExtrinsicResult
from bittensor import SetWeights as SetWeightsIntent
from bittensor import Wallet
from bittensor.intents.weights import normalize
from freezegun import freeze_time

import neurons.validator.tasks.set_weights as set_weights
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import DkgScoreModel
from neurons.validator.tasks.set_weights import (
    ORDERING_SHARE,
    SCORES_IN_EFFECT_AFTER,
    SetWeights,
    _owner_uid,
    _registered_miners,
    _RegisteredMiner,
    _weights_from_scores,
)
from neurons.validator.utils.common.interval import BLOCK_DURATION
from neurons.validator.utils.logger.logger import NuminousLogger
from neurons.validator.version import __spec_version__ as spec_version

SCORED_AT = datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
OWNER_UID = 7


def scores_of(score_by_miner: dict[tuple[int, str], float]) -> list[DkgScoreModel]:
    return [
        DkgScoreModel(
            miner_uid=uid, miner_hotkey=hotkey, scored_at=SCORED_AT, score=score, spec_version=3013
        )
        for (uid, hotkey), score in score_by_miner.items()
    ]


def registered(uid: int, block_at_registration: int) -> _RegisteredMiner:
    return _RegisteredMiner(uid, f"hotkey{uid}", block_at_registration)


def neuron(uid: int, hotkey: str) -> MagicMock:
    registered_neuron = MagicMock()
    registered_neuron.uid = uid
    registered_neuron.hotkey = hotkey
    registered_neuron.block_at_registration = 1000 + uid
    return registered_neuron


def metagraph_of(uids: list[int], hotkeys: list[str]) -> MagicMock:
    metagraph = MagicMock()
    metagraph.neurons = [neuron(uid, hotkey) for uid, hotkey in zip(uids, hotkeys)]
    metagraph.num_uids = len(uids)
    metagraph.owner_hotkey = hotkeys[0]
    return metagraph


def ordering(rank: int, ranked_miners: int) -> float:
    return ORDERING_SHARE / rank / sum(1 / position for position in range(1, ranked_miners + 1))


class TestWeightsFromScores:
    def test_the_top_get_their_squared_scores_and_every_scored_miner_an_ordering_share(self):
        miners = [registered(1, 100), registered(2, 100)]
        scores = scores_of({(1, "hotkey1"): 3.0, (2, "hotkey2"): 1.0})

        weights = _weights_from_scores(miners, scores, OWNER_UID)

        assert weights == pytest.approx(
            {
                1: (1 - ORDERING_SHARE) * 0.9 + ordering(1, 2),
                2: (1 - ORDERING_SHARE) * 0.1 + ordering(2, 2),
            }
        )
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_without_an_ordering_share_the_weights_are_the_squared_scores(self, monkeypatch):
        monkeypatch.setattr(set_weights, "ORDERING_SHARE", 0.0)
        miners = [registered(1, 100), registered(2, 100)]
        scores = scores_of({(1, "hotkey1"): 3.0, (2, "hotkey2"): 1.0})

        assert _weights_from_scores(miners, scores, OWNER_UID) == pytest.approx({1: 0.9, 2: 0.1})

    def test_miners_without_a_score_get_no_weight(self):
        miners = [registered(1, 100), registered(2, 100), registered(3, 100)]
        scores = scores_of({(1, "hotkey1"): 2.0, (2, "hotkey2"): 0.0})

        assert _weights_from_scores(miners, scores, OWNER_UID) == {1: 1.0}

    def test_beyond_the_top_miners_only_the_ordering_share_is_paid(self):
        miners = [registered(uid, 100) for uid in range(1, 61)]
        scores = scores_of({(uid, f"hotkey{uid}"): float(uid) for uid in range(1, 61)})

        weights = _weights_from_scores(miners, scores, OWNER_UID)

        assert list(weights) == list(range(1, 61))
        for uid in range(1, 36):
            assert weights[uid] == pytest.approx(ordering(61 - uid, 60))
        assert weights[36] > ordering(61 - 36, 60)
        assert sum(weights.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("seed", range(40))
    def test_the_weights_never_increase_down_the_ranking(self, seed: int):
        generator = random.Random(seed)
        scored_miners = generator.randint(1, 256)
        miners = [registered(uid, generator.randint(1, 5000)) for uid in range(256)]
        if seed % 4 == 0:
            score_by_uid = {uid: 1.0 for uid in range(scored_miners)}
            score_by_uid[0] = 100.0
        else:
            score_by_uid = {uid: generator.expovariate(1) for uid in range(scored_miners)}
        scores = scores_of({(uid, f"hotkey{uid}"): score for uid, score in score_by_uid.items()})

        weights = _weights_from_scores(miners, scores, OWNER_UID)

        ranked = sorted(
            [miner for miner in miners if miner.uid in score_by_uid],
            key=lambda miner: (-score_by_uid[miner.uid], -miner.block_at_registration, miner.uid),
        )
        in_rank_order = [weights[miner.uid] for miner in ranked]
        assert all(higher >= lower for higher, lower in zip(in_rank_order, in_rank_order[1:]))
        assert min(in_rank_order) > 0
        assert set(weights) == set(score_by_uid)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_every_scored_miner_keeps_a_weight_after_the_chain_rounding(self):
        miners = [registered(uid, 100) for uid in range(256)]
        score_by_uid = {uid: 1.0 for uid in range(1, 250)}
        score_by_uid[0] = 100.0
        scores = scores_of({(uid, f"hotkey{uid}"): score for uid, score in score_by_uid.items()})

        weights = _weights_from_scores(miners, scores, OWNER_UID)
        on_chain_uids, _ = normalize(list(weights), list(weights.values()))

        assert set(on_chain_uids) == set(score_by_uid)

    def test_a_smaller_ordering_share_would_lose_scored_miners_in_the_chain_rounding(
        self, monkeypatch
    ):
        monkeypatch.setattr(set_weights, "ORDERING_SHARE", 0.01)
        miners = [registered(uid, 100) for uid in range(256)]
        score_by_uid = {uid: 1.0 for uid in range(1, 250)}
        score_by_uid[0] = 100.0
        scores = scores_of({(uid, f"hotkey{uid}"): score for uid, score in score_by_uid.items()})

        weights = _weights_from_scores(miners, scores, OWNER_UID)
        on_chain_uids, _ = normalize(list(weights), list(weights.values()))

        assert len(on_chain_uids) < len(score_by_uid)

    def test_a_tie_at_the_cut_off_goes_to_the_newer_registration(self, monkeypatch):
        monkeypatch.setattr(set_weights, "TOP_MINERS", 2)
        miners = [registered(1, 100), registered(2, 300), registered(3, 200)]
        scores = scores_of({(1, "hotkey1"): 5.0, (2, "hotkey2"): 1.0, (3, "hotkey3"): 1.0})

        weights = _weights_from_scores(miners, scores, OWNER_UID)

        assert weights[3] == pytest.approx(ordering(3, 3))
        assert weights[2] > weights[3]

    def test_a_tie_registered_in_the_same_block_goes_to_the_lower_uid(self, monkeypatch):
        monkeypatch.setattr(set_weights, "TOP_MINERS", 2)
        miners = [registered(1, 100), registered(9, 300), registered(4, 300)]
        scores = scores_of({(1, "hotkey1"): 5.0, (9, "hotkey9"): 1.0, (4, "hotkey4"): 1.0})

        weights = _weights_from_scores(miners, scores, OWNER_UID)

        assert weights[9] == pytest.approx(ordering(3, 3))
        assert weights[4] > weights[9]

    @pytest.mark.parametrize(
        "score_by_miner",
        [{}, {(1, "hotkey1"): 0.0, (2, "hotkey2"): 0.0}],
    )
    def test_everything_goes_to_the_owner_when_no_miner_has_a_score(self, score_by_miner):
        miners = [registered(1, 100), registered(2, 100)]

        assert _weights_from_scores(miners, scores_of(score_by_miner), OWNER_UID) == {
            OWNER_UID: 1.0
        }

    def test_a_score_of_a_hotkey_no_longer_on_its_uid_is_ignored(self):
        miners = [registered(1, 100), registered(2, 100)]
        scores = scores_of({(1, "5RegisteredBefore"): 4.0, (2, "hotkey2"): 1.0})

        assert _weights_from_scores(miners, scores, OWNER_UID) == {2: 1.0}

    def test_the_weight_exponent_sets_how_much_the_best_are_favoured(self, monkeypatch):
        monkeypatch.setattr(set_weights, "WEIGHT_EXPONENT", 1)
        monkeypatch.setattr(set_weights, "ORDERING_SHARE", 0.0)
        miners = [registered(1, 100), registered(2, 100)]
        scores = scores_of({(1, "hotkey1"): 3.0, (2, "hotkey2"): 1.0})

        assert _weights_from_scores(miners, scores, OWNER_UID) == pytest.approx({1: 0.75, 2: 0.25})


class TestMetagraph:
    def test_registered_miners_carry_their_registration_block(self):
        metagraph = metagraph_of([1, 2], ["hotkey1", "hotkey2"])

        assert _registered_miners(metagraph) == [
            _RegisteredMiner(1, "hotkey1", 1001),
            _RegisteredMiner(2, "hotkey2", 1002),
        ]

    def test_the_owner_uid_is_the_uid_of_the_owner_hotkey(self):
        metagraph = metagraph_of([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])
        metagraph.owner_hotkey = "hotkey2"

        assert _owner_uid(metagraph) == 2

    def test_an_owner_missing_from_the_metagraph_is_an_error(self):
        metagraph = metagraph_of([1, 2], ["hotkey1", "hotkey2"])
        metagraph.owner_hotkey = "hotkey_not_in_metagraph"

        with pytest.raises(ValueError, match="The subnet owner is not registered"):
            _owner_uid(metagraph)


class TestSetWeights:
    @pytest.fixture
    def dkg_operations(self) -> AsyncMock:
        dkg_operations = AsyncMock(spec=DkgDatabaseOperations)
        dkg_operations.get_latest_scores.return_value = scores_of(
            {(1, "hotkey1"): 2.0, (3, "hotkey3"): 1.0}
        )

        return dkg_operations

    @pytest.fixture
    def bt_wallet(self):
        hotkey_mock = MagicMock()
        hotkey_mock.sign = MagicMock(side_effect=lambda x: x.encode("utf-8"))
        hotkey_mock.ss58_address = "hotkey2"

        bt_wallet = MagicMock(spec=Wallet)
        bt_wallet.get_hotkey = MagicMock(return_value=hotkey_mock)
        bt_wallet.hotkey.ss58_address = "hotkey2"

        return bt_wallet

    def _chain_client(self, execute_behaviour) -> AsyncMock:
        chain_client = AsyncMock()
        chain_client.subnets.metagraph = AsyncMock(
            return_value=metagraph_of([1, 2, 3], ["hotkey1", "hotkey2", "hotkey3"])
        )
        chain_client.hyperparameters.weights_rate_limit = AsyncMock(return_value=100)
        chain_client.hyperparameters.min_allowed_weights = AsyncMock(return_value=1)
        # raw u16; task divides by U16_MAX -> 10.0
        chain_client.hyperparameters.max_weight_limit = AsyncMock(return_value=655350)
        chain_client.execute = AsyncMock(side_effect=execute_behaviour)
        return chain_client

    def _connection(self, execute_behaviour) -> MagicMock:
        chain_client = self._chain_client(execute_behaviour)

        connection = MagicMock()
        connection.__aenter__ = AsyncMock(return_value=chain_client)
        connection.__aexit__ = AsyncMock(return_value=False)
        connection.chain_client = chain_client
        return connection

    @pytest.fixture
    def mock_subtensor(self):
        chain_client = self._chain_client(
            lambda *args, **kwargs: ExtrinsicResult(success=True, message="ok")
        )

        subtensor = MagicMock()
        subtensor.return_value.__aenter__ = AsyncMock(return_value=chain_client)
        subtensor.return_value.__aexit__ = AsyncMock(return_value=False)
        subtensor.chain_client = chain_client

        return subtensor

    @pytest.fixture
    def set_weights_task(
        self,
        dkg_operations: AsyncMock,
        bt_wallet: Wallet,
        mock_subtensor: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # run() builds a fresh Subtensor(network) each tick; patch the constructor so it
        # returns our mock connection instead of opening a real websocket.
        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", mock_subtensor)

        with freeze_time("2024-12-27 07:00:00"):
            return SetWeights(
                interval_seconds=60.0,
                dkg_operations=dkg_operations,
                logger=MagicMock(spec=NuminousLogger),
                netuid=155,
                network="test",
                wallet=bt_wallet,
            )

    def test_init(self, set_weights_task: SetWeights):
        unit = set_weights_task

        assert unit.interval_seconds == 60.0
        assert unit.dkg_operations is not None
        assert unit.logger is not None
        assert unit.netuid == 155
        assert unit.network == "test"
        assert unit.wallet is not None
        assert unit.spec_version == spec_version
        assert unit.name == "set-weights"

    def test_init_rejects_operations_that_are_not_the_dkg_ones(self, bt_wallet: Wallet):
        with pytest.raises(TypeError):
            SetWeights(
                interval_seconds=60.0,
                dkg_operations=MagicMock(),
                logger=MagicMock(spec=NuminousLogger),
                netuid=155,
                network="test",
                wallet=bt_wallet,
            )

    @pytest.mark.parametrize(
        "delta,expected",
        [
            (99 * BLOCK_DURATION, False),
            (101 * BLOCK_DURATION, True),
        ],
    )
    async def test_time_to_set_weights(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, delta, expected
    ):
        set_weights_task.chain_client = mock_subtensor.chain_client
        set_weights_task.last_set_weights_at = time.time() - delta

        assert await set_weights_task.time_to_set_weights() is expected

    async def test_preprocess_weights_success(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock
    ):
        set_weights_task.chain_client = mock_subtensor.chain_client

        uids, weights = await set_weights_task.preprocess_weights({1: 0.2, 2: 0.3, 3: 0.5}, 3)

        np.testing.assert_allclose(weights, np.array([0.2, 0.3, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(uids, np.array([1, 2, 3], dtype=np.int64))

    async def test_preprocess_weights_edge_cases(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, monkeypatch
    ):
        set_weights_task.chain_client = mock_subtensor.chain_client
        weights_with_a_zero = {1: 0.2, 2: 0.3, 3: 0.0}

        uids, weights = await set_weights_task.preprocess_weights(weights_with_a_zero, 3)
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
            await set_weights_task.preprocess_weights(weights_with_a_zero, 3)

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
            await set_weights_task.preprocess_weights(weights_with_a_zero, 3)

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

        uids, weights = await set_weights_task.preprocess_weights({1: 0.2, 2: 0.3, 3: 0.41}, 3)

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

    @freeze_time("2026-10-16 17:30:00")
    async def test_run_sets_the_weights_of_the_scores_in_effect(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, dkg_operations: AsyncMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

        await unit.run()

        dkg_operations.get_latest_scores.assert_awaited_once_with(
            scored_until=datetime(2026, 10, 16, 17, 30, tzinfo=timezone.utc)
            - SCORES_IN_EFFECT_AFTER
        )
        submitted_intent = mock_subtensor.chain_client.execute.await_args.args[0]
        assert submitted_intent.uids == [1, 3]
        assert submitted_intent.weights == pytest.approx([0.79733, 0.20267], abs=1e-5)
        unit.logger.info.assert_called_once_with(
            "Weights from DKG scores",
            extra={
                "scored_at": SCORED_AT.isoformat(),
                "miners_with_a_weight": 2,
                "all_to_owner": False,
            },
        )
        assert [call.args[0] for call in unit.logger.debug.call_args_list] == [
            "Weights rate limit",
            "Weights rate limit",
            "Attempting to set the weights - enough blocks passed.",
            "Weights processed for the chain",
            "Weights set successfully.",
        ]
        processed_log = unit.logger.debug.call_args_list[3].kwargs["extra"]
        assert processed_log == {
            "min_allowed_weights": 1,
            "max_weight_limit": 10.0,
            "uids": 3,
            "uids_with_a_weight": 2,
            "top_before": [(1, 0.79733), (3, 0.20267), (2, 0.0)],
            "top_after": [(1, 0.79733), (3, 0.20267)],
        }
        assert unit.logger.warning.call_count == 0
        assert unit.logger.error.call_count == 0
        mock_subtensor.return_value.__aenter__.assert_awaited_once()
        mock_subtensor.return_value.__aexit__.assert_awaited_once()
        mock_subtensor.chain_client.subnets.metagraph.assert_awaited_once_with(unit.netuid)

    async def test_run_gives_the_chain_limits_every_uid_in_order_with_zeros(
        self, set_weights_task: SetWeights, monkeypatch
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        received = {}
        real_process_weights = set_weights.process_weights

        def process_and_record(**arguments):
            received.update(arguments)
            return real_process_weights(**arguments)

        monkeypatch.setattr(set_weights, "process_weights", process_and_record)

        await unit.run()

        assert received["uids"].tolist() == [1, 2, 3]
        assert received["weights"].tolist() == pytest.approx([0.79733, 0.0, 0.20267], abs=1e-5)
        assert received["num_neurons"] == 3

    async def test_run_caps_a_miner_above_the_max_weight_limit_of_the_chain(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        mock_subtensor.chain_client.hyperparameters.max_weight_limit = AsyncMock(
            return_value=round(0.6 * 65535)
        )

        await unit.run()

        submitted_intent = mock_subtensor.chain_client.execute.await_args.args[0]
        assert submitted_intent.uids == [1, 3]
        assert submitted_intent.weights == pytest.approx([0.6, 0.4], abs=1e-4)
        assert max(submitted_intent.weights) <= 0.6 + 1e-5
        unit.logger.warning.assert_called_once()
        assert (
            unit.logger.warning.call_args.args[0]
            == "Processed weights do not match the original weights."
        )

    async def test_run_sends_nothing_when_the_chain_requires_more_weights_than_miners_scored(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        mock_subtensor.chain_client.hyperparameters.min_allowed_weights = AsyncMock(return_value=3)

        with pytest.raises(ValueError, match="Processed UIDs do not match the original UIDs."):
            await unit.run()

        mock_subtensor.chain_client.execute.assert_not_awaited()

    async def test_run_burns_everything_to_the_owner_when_no_miner_has_a_score(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, dkg_operations: AsyncMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        dkg_operations.get_latest_scores.return_value = scores_of(
            {(1, "hotkey1"): 0.0, (2, "hotkey2"): 0.0, (3, "hotkey3"): 0.0}
        )

        await unit.run()

        submitted_intent = mock_subtensor.chain_client.execute.await_args.args[0]
        assert submitted_intent.uids == [1]
        assert submitted_intent.weights == pytest.approx([1.0])
        assert unit.logger.info.call_args.kwargs["extra"]["all_to_owner"] is True

    async def test_run_sets_no_weights_without_scores_in_effect(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, dkg_operations: AsyncMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION
        dkg_operations.get_latest_scores.return_value = []

        await unit.run()

        unit.logger.warning.assert_called_once_with("No DKG scores in effect yet, weights not set.")
        mock_subtensor.chain_client.subnets.metagraph.assert_not_awaited()
        mock_subtensor.chain_client.execute.assert_not_awaited()

    async def test_run_reads_nothing_before_the_rate_limit_allows_it(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock, dkg_operations: AsyncMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time()

        await unit.run()

        dkg_operations.get_latest_scores.assert_not_awaited()
        mock_subtensor.chain_client.subnets.metagraph.assert_not_awaited()
        mock_subtensor.chain_client.execute.assert_not_awaited()

    async def test_run_set_weights_timeout(
        self, set_weights_task: SetWeights, mock_subtensor: MagicMock
    ):
        unit = set_weights_task
        unit.last_set_weights_at = time.time() - 101 * BLOCK_DURATION

        async def execute_mock(*args, **kwargs):
            await asyncio.sleep(2)

            return ExtrinsicResult(success=True, message="Message")

        mock_subtensor.chain_client.execute = AsyncMock(side_effect=execute_mock)
        unit.timeout_seconds = 0.01

        with pytest.raises(asyncio.TimeoutError):
            await unit.run()

        mock_subtensor.return_value.__aexit__.assert_awaited_once()

    async def test_run_builds_fresh_subtensor_each_tick(
        self, set_weights_task: SetWeights, monkeypatch
    ):
        unit = set_weights_task

        constructed = []

        def subtensor_factory(network):
            connection = self._connection(
                lambda *args, **kwargs: ExtrinsicResult(success=True, message="ok")
            )
            constructed.append(connection)
            return connection

        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", subtensor_factory)

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

        wedged_connection = self._connection(wedge_error)
        healthy_connection = self._connection(
            lambda *args, **kwargs: ExtrinsicResult(success=True, message="ok")
        )

        connections = iter([wedged_connection, healthy_connection])
        constructed = []

        def subtensor_factory(network):
            connection = next(connections)
            constructed.append(connection)
            return connection

        monkeypatch.setattr("neurons.validator.tasks.set_weights.Subtensor", subtensor_factory)

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
