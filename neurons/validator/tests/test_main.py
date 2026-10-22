import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from neurons.validator.main import main
from neurons.validator.utils.config import ValidatorSettings


@patch("neurons.validator.main.DbVacuum", spec=True)
@patch("neurons.validator.main.DbCleaner", spec=True)
@patch("neurons.validator.main.SetWeights", spec=True)
@patch("neurons.validator.main.PostDkgScores", spec=True)
@patch("neurons.validator.main.ScoreDkgMiners", spec=True)
@patch("neurons.validator.main.PullDkgResults", spec=True)
@patch("neurons.validator.main.PullDkgRounds", spec=True)
@patch("neurons.validator.main.SyncMinersMetadata", spec=True)
class TestValidatorMain:
    @pytest.mark.parametrize(
        "config_env,db_path",
        [
            ("prod", "validator.db"),
            ("test", "validator_test.db"),
        ],
    )
    def test_main(
        self,
        mock_sync_miners,
        mock_pull_dkg_rounds,
        mock_pull_dkg_results,
        mock_score_dkg_miners,
        mock_post_dkg_scores,
        mock_set_weights,
        mock_db_cleaner,
        mock_db_vacuum,
        config_env,
        db_path,
    ):
        # Patch key dependencies inside the method
        with (
            patch("neurons.validator.main.override_loggers_level") as mock_override_loggers_level,
            patch("neurons.validator.main.set_bittensor_logger") as mock_set_bittensor_logger,
            patch("neurons.validator.main.get_config", spec=True) as get_config,
            patch("neurons.validator.main.Wallet", spec=True) as MockWallet,
            patch("neurons.validator.main.Subtensor", spec=True) as MockAsyncSubtensor,
            patch("neurons.validator.main.DkgClient", spec=True) as MockDkgClient,
            patch(
                "neurons.validator.main.DkgDatabaseOperations", spec=True
            ) as MockDkgDatabaseOperations,
            patch("neurons.validator.main.DatabaseClient", spec=True) as MockDatabaseClient,
            patch("neurons.validator.main.TasksScheduler") as MockTasksScheduler,
            patch("neurons.validator.main.measure_event_loop_lag") as mock_measure_event_loop_lag,
            patch("neurons.validator.main.logger", spec=True) as mock_logger,
        ):
            # Mock get_config
            logger_level = 99
            netuid = 1234
            network = "testnet"

            get_config.return_value = ValidatorSettings(
                netuid=netuid,
                subtensor_network=network,
                subtensor_chain_endpoint=None,
                wallet_name="wallet_name",
                wallet_hotkey="wallet_hotkey",
                wallet_path="~/.bittensor/wallets/",
                numinous_env=config_env,
                db_path=db_path,
                logging_level=logger_level,
            )

            # Mock Wallet
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "hk3"
            MockWallet.return_value = mock_wallet

            # Mock AsyncSubtensor async context manager ---
            mock_async_subtensor = AsyncMock()
            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk0", "hk1", "hk2", "hk3", "hk4"]
            mock_async_subtensor.subnets.metagraph = AsyncMock(return_value=mock_metagraph)

            MockAsyncSubtensor.return_value.__aenter__ = AsyncMock(
                return_value=mock_async_subtensor
            )
            MockAsyncSubtensor.return_value.__aexit__ = AsyncMock(return_value=False)

            # Mock Database Client
            mock_db_client = MockDatabaseClient.return_value
            mock_db_client.migrate = AsyncMock()

            # Mock TasksScheduler
            mock_scheduler = MockTasksScheduler.return_value
            mock_scheduler.start = AsyncMock(return_value=None)
            mock_scheduler.add = MagicMock(return_value=None)

            # Mock Logger
            mock_logger.start_session = MagicMock()

            # Run the validator
            asyncio.run(main())

            # Verify loggers set
            mock_override_loggers_level.assert_called_once_with(logger_level)
            mock_set_bittensor_logger.assert_called_once()

            # Verify start session called
            mock_logger.start_session.assert_called_once()

            # Verify get_config() was called
            get_config.assert_called_once()

            # Verify DatabaseClient args
            MockDatabaseClient.assert_called_once_with(db_path=db_path, logger=mock_logger)

            # Verify Wallet built from settings
            MockWallet.assert_called_once_with(
                name="wallet_name", hotkey="wallet_hotkey", path="~/.bittensor/wallets/"
            )

            # Verify the boot Subtensor built from the network name
            MockAsyncSubtensor.assert_called_once_with(network)

            # Verify the graph forecasting client and db operations
            MockDkgClient.assert_called_once_with(
                env=config_env, logger=mock_logger, bt_wallet=mock_wallet
            )
            MockDkgDatabaseOperations.assert_called_once_with(db_client=mock_db_client)

            # Verify the graph forecasting pull tasks
            mock_pull_dkg_rounds.assert_called_once_with(
                interval_seconds=600.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                dkg_client=MockDkgClient.return_value,
                logger=mock_logger,
            )
            mock_pull_dkg_results.assert_called_once_with(
                interval_seconds=900.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                dkg_client=MockDkgClient.return_value,
                logger=mock_logger,
            )
            mock_score_dkg_miners.assert_called_once_with(
                interval_seconds=300.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                logger=mock_logger,
            )
            mock_post_dkg_scores.assert_called_once_with(
                interval_seconds=300.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                dkg_client=MockDkgClient.return_value,
                logger=mock_logger,
            )
            mock_set_weights.assert_called_once_with(
                interval_seconds=379.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                logger=mock_logger,
                netuid=netuid,
                network=network,
                wallet=mock_wallet,
            )

            mock_sync_miners.assert_called_once_with(
                interval_seconds=300.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                netuid=netuid,
                network=network,
                logger=mock_logger,
            )
            mock_db_vacuum.assert_called_once_with(
                interval_seconds=300.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                logger=mock_logger,
                pages=500,
            )
            mock_db_cleaner.assert_called_once_with(
                interval_seconds=3600.0,
                dkg_operations=MockDkgDatabaseOperations.return_value,
                batch_size=4000,
                logger=mock_logger,
            )

            # Verify migrate() was called
            mock_db_client.migrate.assert_awaited_once()

            # Verify AsyncSubtensor context manager used
            MockAsyncSubtensor.return_value.__aenter__.assert_awaited_once()
            MockAsyncSubtensor.return_value.__aexit__.assert_awaited_once()

            # Verify scheduler was started
            mock_scheduler.start.assert_awaited_once()

            # Verify event loop lag is measured
            mock_measure_event_loop_lag.assert_awaited_once()

            # Verify the tasks added
            assert [call.kwargs["task"] for call in mock_scheduler.add.call_args_list] == [
                mock_sync_miners.return_value,
                mock_pull_dkg_rounds.return_value,
                mock_pull_dkg_results.return_value,
                mock_score_dkg_miners.return_value,
                mock_post_dkg_scores.return_value,
                mock_set_weights.return_value,
                mock_db_cleaner.return_value,
                mock_db_vacuum.return_value,
            ]

            # Verify logging
            mock_logger.info.assert_called_with(
                "Validator started",
                extra={
                    "validator_uid": 3,
                    "validator_hotkey": mock_wallet.hotkey.ss58_address,
                    "bt_network": network,
                    "bt_netuid": netuid,
                    "numinous_env": config_env,
                    "db_path": db_path,
                    "python": ANY,
                    "sqlite": ANY,
                },
            )
