from datetime import datetime

import pytest
from pydantic import ValidationError

from neurons.validator.models.agent_run_logs import (
    AGENT_RUN_LOGS_FIELDS,
    AgentRunLogExportedStatus,
    AgentRunLogsModel,
)


class TestAgentRunLogsModel:
    def test_create_minimal(self):
        # Minimal required fields
        model = AgentRunLogsModel(
            run_id="run_123",
            log_content="Agent execution log content here",
        )

        assert model.run_id == "run_123"
        assert model.log_content == "Agent execution log content here"

        # Defaults
        assert model.exported is False
        assert model.created_at is None
        assert model.updated_at is None

    def test_create_full_success(self):
        created = datetime(2024, 1, 1, 12, 0, 0)
        updated = datetime(2024, 1, 1, 12, 30, 0)

        model = AgentRunLogsModel(
            run_id="run_abc_123",
            log_content="Full log content with details",
            exported=True,
            created_at=created,
            updated_at=updated,
        )

        assert model.run_id == "run_abc_123"
        assert model.log_content == "Full log content with details"
        assert model.exported is True
        assert model.created_at == created
        assert model.updated_at == updated

    def test_create_with_large_log_content(self):
        # Test with large log content (25KB)
        large_log = "x" * 25000
        model = AgentRunLogsModel(
            run_id="run_large",
            log_content=large_log,
        )

        assert len(model.log_content) == 25000
        assert model.log_content == large_log

    def test_exported_int_to_bool(self):
        # exported as integer should convert to bool
        model = AgentRunLogsModel(
            run_id="run_1",
            log_content="Log content",
            exported=1,
        )
        assert model.exported is True

        model2 = AgentRunLogsModel(
            run_id="run_2",
            log_content="Log content 2",
            exported=0,
        )
        assert model2.exported is False

    def test_exported_bool_passthrough(self):
        # exported as boolean should pass through
        model = AgentRunLogsModel(
            run_id="run_1",
            log_content="Log content",
            exported=True,
        )
        assert model.exported is True

        model2 = AgentRunLogsModel(
            run_id="run_2",
            log_content="Log content 2",
            exported=False,
        )
        assert model2.exported is False

    def test_invalid_run_id_type(self):
        # run_id must be string
        with pytest.raises(ValidationError):
            AgentRunLogsModel(
                run_id=123,
                log_content="Log content",
            )

    def test_invalid_log_content_type(self):
        # log_content must be string
        with pytest.raises(ValidationError):
            AgentRunLogsModel(
                run_id="run_1",
                log_content=123,
            )

    def test_missing_required_run_id(self):
        # run_id is required
        with pytest.raises(ValidationError):
            AgentRunLogsModel(
                log_content="Log content",
            )

    def test_missing_required_log_content(self):
        # log_content is required
        with pytest.raises(ValidationError):
            AgentRunLogsModel(
                run_id="run_1",
            )

    def test_primary_key_property(self):
        model = AgentRunLogsModel(
            run_id="run_123",
            log_content="Log content",
        )
        assert model.primary_key == ["run_id"]

    def test_empty_log_content(self):
        # Empty log content should be allowed (though unusual)
        model = AgentRunLogsModel(
            run_id="run_empty",
            log_content="",
        )
        assert model.log_content == ""

    def test_log_content_with_special_characters(self):
        # Test log content with special characters
        special_log = """Line 1: Normal text
Line 2: Special chars: !@#$%^&*()
Line 3: Unicode: 你好世界 🚀
Line 4: Newlines and tabs\t\n"""

        model = AgentRunLogsModel(
            run_id="run_special",
            log_content=special_log,
        )
        assert model.log_content == special_log

    def test_log_content_with_error_traceback(self):
        # Simulate typical error log with traceback
        error_log = """Traceback (most recent call last):
  File "agent.py", line 42, in agent_main
    result = process_event(event_data)
  File "agent.py", line 10, in process_event
    return data['missing_key']
KeyError: 'missing_key'
"""
        model = AgentRunLogsModel(
            run_id="run_error",
            log_content=error_log,
        )
        assert "KeyError" in model.log_content
        assert "Traceback" in model.log_content

    def test_model_serialization(self):
        # Test that model can be serialized to dict
        model = AgentRunLogsModel(
            run_id="run_serialize",
            log_content="Serialization test",
            exported=True,
        )

        data = model.model_dump()
        assert data["run_id"] == "run_serialize"
        assert data["log_content"] == "Serialization test"
        assert data["exported"] is True

    def test_model_deserialization(self):
        # Test that model can be deserialized from dict
        data = {
            "run_id": "run_deserialize",
            "log_content": "Deserialization test",
            "exported": False,
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "updated_at": datetime(2024, 1, 1, 12, 30, 0),
        }

        model = AgentRunLogsModel(**data)
        assert model.run_id == "run_deserialize"
        assert model.log_content == "Deserialization test"
        assert model.exported is False


class TestAgentRunLogExportedStatus:
    def test_exported_status_enum_values(self):
        assert AgentRunLogExportedStatus.NOT_EXPORTED == 0
        assert AgentRunLogExportedStatus.EXPORTED == 1


class TestAgentRunLogsFields:
    def test_fields_constant_exists(self):
        # Verify AGENT_RUN_LOGS_FIELDS constant is defined
        assert AGENT_RUN_LOGS_FIELDS is not None

    def test_fields_contains_all_model_fields(self):
        # Verify all model fields are in the constant
        expected_fields = {"run_id", "log_content", "exported", "created_at", "updated_at"}
        assert set(AGENT_RUN_LOGS_FIELDS) == expected_fields
