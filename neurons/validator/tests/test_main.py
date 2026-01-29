import asyncio
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from neurons.validator.main import main


@patch("neurons.validator.main.RunAgents", spec=True)
@patch("neurons.validator.main.PullAgents", spec=True)
@patch("neurons.validator.main.SetWeights", spec=True)
@patch("neurons.validator.main.ExportScores", spec=True)
@patch("neurons.validator.main.Scoring", spec=True)
@patch("neurons.validator.main.ExportAgentRunLogs", spec=True)
@patch("neurons.validator.main.ExportAgentRuns", spec=True)
@patch("neurons.validator.main.ExportPredictions", spec=True)
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
        mock_export_predictions,
        mock_export_agent_runs,
        mock_export_agent_run_logs,
        mock_peer_scoring,
        mock_export_scores,
        mock_set_weights,
        mock_pull_agents,
        mock_run_agents,
        config_env,
        db_path,
    ):
        # Patch key dependencies inside the method
        with (
            patch("neurons.validator.main.assert_requirements") as mock_assert_requirements,
            patch("neurons.validator.main.override_loggers_level") as mock_override_loggers_level,
            patch("neurons.validator.main.set_bittensor_logger") as mock_set_bittensor_logger,
            patch(
                "neurons.validator.main.set_async_substrate_interface_logger"
            ) as mock_set_async_substrate_interface_logger,
            patch("neurons.validator.main.get_config", spec=True) as get_config,
            patch("neurons.validator.main.Wallet", spec=True) as MockWallet,
            patch("neurons.validator.main.SandboxManager", spec=True),
            patch("neurons.validator.main.AsyncSubtensor", spec=True) as MockAsyncSubtensor,
            patch("neurons.validator.main.NuminousClient", spec=True) as MockNuminousClient,
            patch("neurons.validator.main.DatabaseClient", spec=True) as MockDatabaseClient,
            patch("neurons.validator.main.TasksScheduler") as MockTasksScheduler,
            patch("neurons.validator.main.measure_event_loop_lag") as mock_measure_event_loop_lag,
            patch("neurons.validator.main.logger", spec=True) as mock_logger,
        ):
            # Mock get_config
            logger_level = 99
            gateway_url = "https://test.numinous.earth"
            validator_sync_hour = 1
            netuid = 1234
            network = "testnet"

            get_config.return_value = (
                {"netuid": netuid, "subtensor": {"network": network}},
                config_env,
                db_path,
                logger_level,
                gateway_url,
                validator_sync_hour,
            )

            # Mock Wallet
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "hk3"
            MockWallet.return_value = mock_wallet

            # Mock AsyncSubtensor async context manager ---
            mock_async_subtensor = AsyncMock()
            mock_metagraph = MagicMock()
            mock_metagraph.hotkeys = ["hk0", "hk1", "hk2", "hk3", "hk4"]
            mock_async_subtensor.metagraph = AsyncMock(return_value=mock_metagraph)

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

            # Verify assert_requirements() was called
            mock_assert_requirements.assert_called_once()

            # Verify loggers set
            mock_override_loggers_level.assert_called_once_with(logger_level)
            mock_set_bittensor_logger.assert_called_once()
            mock_set_async_substrate_interface_logger.assert_called_once()

            # Verify start session called
            mock_logger.start_session.assert_called_once()

            # Verify get_config() was called
            get_config.assert_called_once()

            # Verify DatabaseClient args
            MockDatabaseClient.assert_called_once_with(db_path=db_path, logger=mock_logger)

            # Verify NuminousClient args
            MockNuminousClient.assert_called_once_with(
                env=config_env, logger=mock_logger, bt_wallet=ANY
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

            # Verify tasks added count
            assert mock_scheduler.add.call_count == 14

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
