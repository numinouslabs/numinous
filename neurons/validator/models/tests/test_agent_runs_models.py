from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.agent_runs import (
    AgentRunExportedStatus,
    AgentRunsModel,
    AgentRunStatus,
    IsFinalStatus,
)


class TestAgentRunsModel:
    def test_create_minimal(self):
        # Minimal required fields
        model = AgentRunsModel(
            run_id="run_123",
            unique_event_id="event_456",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )

        assert model.run_id == "run_123"
        assert model.unique_event_id == "event_456"
        assert model.agent_version_id == "agent_v1"
        assert model.miner_uid == 42
        assert model.miner_hotkey == "5GTest..."
        assert model.status == AgentRunStatus.SUCCESS

        # Defaults
        assert model.exported is False
        assert model.is_final is True
        assert model.created_at is None
        assert model.updated_at is None

    def test_create_full_success(self):
        created = datetime(2024, 1, 1, 12, 0, 0)
        updated = datetime(2024, 1, 1, 12, 30, 0)

        model = AgentRunsModel(
            run_id="run_abc_123",
            unique_event_id="event_xyz_456",
            agent_version_id="agent_version_1",
            miner_uid=99,
            miner_hotkey="hotkey_xyz",
            status=AgentRunStatus.SUCCESS,
            exported=True,
            is_final=True,
            created_at=created,
            updated_at=updated,
        )

        assert model.run_id == "run_abc_123"
        assert model.unique_event_id == "event_xyz_456"
        assert model.agent_version_id == "agent_version_1"
        assert model.miner_uid == 99
        assert model.miner_hotkey == "hotkey_xyz"
        assert model.status == AgentRunStatus.SUCCESS
        assert model.exported is True
        assert model.is_final is True
        assert model.created_at == created
        assert model.updated_at == updated

    def test_create_with_error_status(self):
        model = AgentRunsModel(
            run_id="run_error_1",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=10,
            miner_hotkey="hotkey_10",
            status=AgentRunStatus.INTERNAL_AGENT_ERROR,
        )

        assert model.status == AgentRunStatus.INTERNAL_AGENT_ERROR

    def test_exported_int_to_bool(self):
        # exported as integer should convert to bool
        model = AgentRunsModel(
            run_id="run_1",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=1,
        )
        assert model.exported is True

        model2 = AgentRunsModel(
            run_id="run_2",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=0,
        )
        assert model2.exported is False

    def test_exported_bool_passthrough(self):
        # exported as boolean should pass through
        model = AgentRunsModel(
            run_id="run_1",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            exported=True,
        )
        assert model.exported is True

    def test_is_final_int_to_bool(self):
        # is_final as integer should convert to bool
        model = AgentRunsModel(
            run_id="run_1",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SUCCESS,
            is_final=1,
        )
        assert model.is_final is True

        model2 = AgentRunsModel(
            run_id="run_2",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=1,
            miner_hotkey="hotkey_1",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=0,
        )
        assert model2.is_final is False

    def test_all_status_types(self):
        # Test all status enum values
        statuses_to_test = [
            AgentRunStatus.SUCCESS,
            AgentRunStatus.INTERNAL_AGENT_ERROR,
            AgentRunStatus.INVALID_SANDBOX_OUTPUT,
            AgentRunStatus.SANDBOX_TIMEOUT,
        ]

        for status in statuses_to_test:
            model = AgentRunsModel(
                run_id=f"run_{status.value}",
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status=status,
            )
            assert model.status == status

    def test_invalid_status_type(self):
        # status must be AgentRunStatus enum
        with pytest.raises(ValidationError):
            AgentRunsModel(
                run_id="run_1",
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid=1,
                miner_hotkey="hotkey_1",
                status="invalid_status",
            )

    def test_invalid_miner_uid_type(self):
        # miner_uid must be integer
        with pytest.raises(ValidationError):
            AgentRunsModel(
                run_id="run_1",
                unique_event_id="event_1",
                agent_version_id="agent_v1",
                miner_uid="not_an_int",
                miner_hotkey="hotkey_1",
                status=AgentRunStatus.SUCCESS,
            )

    def test_primary_key_property(self):
        model = AgentRunsModel(
            run_id="run_123",
            unique_event_id="event_456",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="5GTest...",
            status=AgentRunStatus.SUCCESS,
        )
        assert model.primary_key == ["run_id"]

    def test_retry_scenario(self):
        # Simulate retry scenario: first two attempts not final, third is final
        run1 = AgentRunsModel(
            run_id="run_attempt_1",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )
        assert run1.is_final is False
        assert run1.status == AgentRunStatus.SANDBOX_TIMEOUT

        run2 = AgentRunsModel(
            run_id="run_attempt_2",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SANDBOX_TIMEOUT,
            is_final=False,
        )
        assert run2.is_final is False

        run3 = AgentRunsModel(
            run_id="run_attempt_3",
            unique_event_id="event_1",
            agent_version_id="agent_v1",
            miner_uid=42,
            miner_hotkey="hotkey_42",
            status=AgentRunStatus.SUCCESS,
            is_final=True,
        )
        assert run3.is_final is True
        assert run3.status == AgentRunStatus.SUCCESS


class TestAgentRunStatus:
    def test_status_enum_values(self):
        assert AgentRunStatus.SUCCESS.value == "success"
        assert AgentRunStatus.INTERNAL_AGENT_ERROR.value == "internal_agent_error"
        assert AgentRunStatus.INVALID_SANDBOX_OUTPUT.value == "invalid_sandbox_output"
        assert AgentRunStatus.SANDBOX_TIMEOUT.value == "sandbox_timeout"


class TestAgentRunExportedStatus:
    def test_exported_status_enum_values(self):
        assert AgentRunExportedStatus.NOT_EXPORTED == 0
        assert AgentRunExportedStatus.EXPORTED == 1


class TestIsFinalStatus:
    def test_is_final_status_enum_values(self):
        assert IsFinalStatus.NOT_FINAL == 0
        assert IsFinalStatus.IS_FINAL == 1
