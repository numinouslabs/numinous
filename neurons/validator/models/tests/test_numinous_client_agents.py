import base64
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from neurons.validator.models.numinous_client import (
    AgentRunSubmission,
    GetAgentsQueryParams,
    GetAgentsResponse,
    MinerAgentWithCode,
    PostAgentRunsRequestBody,
)


class TestMinerAgentWithCode:
    def test_create_valid_agent(self):
        version_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        code = base64.b64encode(b"def agent_main():\n    return 0.5").decode("utf-8")

        agent = MinerAgentWithCode(
            version_id=version_id,
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            miner_uid=42,
            agent_name="TestAgent",
            version_number=1,
            created_at=created_at,
            code=code,
        )

        assert agent.version_id == version_id
        assert agent.miner_hotkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        assert agent.miner_uid == 42
        assert agent.agent_name == "TestAgent"
        assert agent.version_number == 1
        assert agent.created_at == created_at
        assert agent.code == code

    def test_create_from_json(self):
        json_data = {
            "version_id": "123e4567-e89b-12d3-a456-426614174000",
            "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "miner_uid": 42,
            "agent_name": "TestAgent",
            "version_number": 1,
            "created_at": "2024-01-15T10:30:00Z",
            "code": "ZGVmIGFnZW50X21haW4oKToKICAgIHJldHVybiAwLjU=",
        }

        agent = MinerAgentWithCode.model_validate(json_data)

        assert agent.version_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert agent.miner_uid == 42
        assert agent.agent_name == "TestAgent"
        assert agent.version_number == 1
        assert isinstance(agent.created_at, datetime)

    def test_invalid_version_id(self):
        with pytest.raises(ValidationError):
            MinerAgentWithCode(
                version_id="not-a-uuid",
                miner_hotkey="hotkey",
                miner_uid=1,
                agent_name="Agent",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code="code",
            )

    def test_invalid_miner_uid_type(self):
        with pytest.raises(ValidationError):
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_hotkey="hotkey",
                miner_uid="not-an-int",
                agent_name="Agent",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code="code",
            )

    def test_decode_code(self):
        code_text = "def agent_main():\n    return 0.5"
        code_base64 = base64.b64encode(code_text.encode("utf-8")).decode("utf-8")

        agent = MinerAgentWithCode(
            version_id=uuid4(),
            miner_hotkey="hotkey",
            miner_uid=1,
            agent_name="Agent",
            version_number=1,
            created_at=datetime.now(timezone.utc),
            code=code_base64,
        )

        decoded = base64.b64decode(agent.code).decode("utf-8")
        assert decoded == code_text


class TestGetAgentsQueryParams:
    def test_default_values(self):
        params = GetAgentsQueryParams()

        assert params.offset == 0
        assert params.limit == 50

    def test_custom_values(self):
        params = GetAgentsQueryParams(offset=10, limit=20)

        assert params.offset == 10
        assert params.limit == 20

    def test_offset_validation_negative(self):
        with pytest.raises(ValidationError):
            GetAgentsQueryParams(offset=-1)

    def test_limit_validation_too_low(self):
        with pytest.raises(ValidationError):
            GetAgentsQueryParams(limit=0)

    def test_limit_validation_too_high(self):
        with pytest.raises(ValidationError):
            GetAgentsQueryParams(limit=101)

    def test_limit_boundary_values(self):
        params_min = GetAgentsQueryParams(limit=1)
        assert params_min.limit == 1

        params_max = GetAgentsQueryParams(limit=100)
        assert params_max.limit == 100


class TestGetAgentsResponse:
    def test_empty_response(self):
        response = GetAgentsResponse(count=0, items=[])

        assert response.count == 0
        assert len(response.items) == 0

    def test_response_with_agents(self):
        agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_hotkey=f"hotkey{i}",
                miner_uid=i,
                agent_name=f"Agent{i}",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code="code",
            )
            for i in range(3)
        ]

        response = GetAgentsResponse(count=100, items=agents)

        assert response.count == 100
        assert len(response.items) == 3

    def test_count_not_equal_items_length(self):
        agents = [
            MinerAgentWithCode(
                version_id=uuid4(),
                miner_hotkey="hotkey",
                miner_uid=1,
                agent_name="Agent",
                version_number=1,
                created_at=datetime.now(timezone.utc),
                code="code",
            )
        ]

        response = GetAgentsResponse(count=150, items=agents)

        assert response.count == 150
        assert len(response.items) == 1

    def test_from_json(self):
        json_data = {
            "count": 2,
            "items": [
                {
                    "version_id": "123e4567-e89b-12d3-a456-426614174000",
                    "miner_hotkey": "hotkey1",
                    "miner_uid": 1,
                    "agent_name": "Agent1",
                    "version_number": 1,
                    "created_at": "2024-01-15T10:30:00Z",
                    "code": "ZGVmIGFnZW50X21haW4oKToKICAgIHJldHVybiAwLjU=",
                },
                {
                    "version_id": "223e4567-e89b-12d3-a456-426614174000",
                    "miner_hotkey": "hotkey2",
                    "miner_uid": 2,
                    "agent_name": "Agent2",
                    "version_number": 2,
                    "created_at": "2024-01-15T11:30:00Z",
                    "code": "ZGVmIGFnZW50X21haW4oKToKICAgIHJldHVybiAwLjU=",
                },
            ],
        }

        response = GetAgentsResponse.model_validate(json_data)

        assert response.count == 2
        assert len(response.items) == 2
        assert response.items[0].miner_uid == 1
        assert response.items[1].miner_uid == 2


class TestAgentRunSubmission:
    def test_create_valid_run(self):
        run = AgentRunSubmission(
            run_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
            miner_uid=10,
            miner_hotkey="miner_hotkey_1",
            vali_uid=5,
            vali_hotkey="validator_hotkey",
            status="SUCCESS",
            event_id="event_123",
            version_id=UUID("223e4567-e89b-12d3-a456-426614174001"),
            is_final=True,
        )

        assert run.run_id == UUID("123e4567-e89b-12d3-a456-426614174000")
        assert run.miner_uid == 10
        assert run.miner_hotkey == "miner_hotkey_1"
        assert run.vali_uid == 5
        assert run.vali_hotkey == "validator_hotkey"
        assert run.status == "SUCCESS"
        assert run.event_id == "event_123"
        assert run.version_id == UUID("223e4567-e89b-12d3-a456-426614174001")
        assert run.is_final is True

    def test_create_from_json(self):
        json_data = {
            "run_id": "323e4567-e89b-12d3-a456-426614174002",
            "miner_uid": 20,
            "miner_hotkey": "miner_hotkey_2",
            "vali_uid": 5,
            "vali_hotkey": "validator_hotkey",
            "status": "SANDBOX_TIMEOUT",
            "event_id": "event_456",
            "version_id": "423e4567-e89b-12d3-a456-426614174003",
            "is_final": False,
        }

        run = AgentRunSubmission.model_validate(json_data)

        assert run.run_id == UUID("323e4567-e89b-12d3-a456-426614174002")
        assert run.miner_uid == 20
        assert run.status == "SANDBOX_TIMEOUT"
        assert run.is_final is False

    def test_invalid_run_id(self):
        with pytest.raises(ValidationError):
            AgentRunSubmission(
                run_id="not-a-uuid",
                miner_uid=10,
                miner_hotkey="miner_hotkey",
                vali_uid=5,
                vali_hotkey="validator_hotkey",
                status="SUCCESS",
                event_id="event_123",
                version_id=UUID("223e4567-e89b-12d3-a456-426614174001"),
                is_final=True,
            )

    def test_invalid_version_id(self):
        with pytest.raises(ValidationError):
            AgentRunSubmission(
                run_id=UUID("123e4567-e89b-12d3-a456-426614174000"),
                miner_uid=10,
                miner_hotkey="miner_hotkey",
                vali_uid=5,
                vali_hotkey="validator_hotkey",
                status="SUCCESS",
                event_id="event_123",
                version_id="not-a-uuid",
                is_final=True,
            )


class TestPostAgentRunsRequestBody:
    def test_empty_runs_list(self):
        body = PostAgentRunsRequestBody(runs=[])

        assert body.runs == []

    def test_single_run(self):
        run = AgentRunSubmission(
            run_id=UUID("523e4567-e89b-12d3-a456-426614174004"),
            miner_uid=30,
            miner_hotkey="miner_hotkey_3",
            vali_uid=5,
            vali_hotkey="validator_hotkey",
            status="INTERNAL_AGENT_ERROR",
            event_id="event_789",
            version_id=UUID("623e4567-e89b-12d3-a456-426614174005"),
            is_final=True,
        )

        body = PostAgentRunsRequestBody(runs=[run])

        assert len(body.runs) == 1
        assert body.runs[0].run_id == UUID("523e4567-e89b-12d3-a456-426614174004")
        assert body.runs[0].status == "INTERNAL_AGENT_ERROR"

    def test_multiple_runs(self):
        runs = [
            AgentRunSubmission(
                run_id=UUID("723e4567-e89b-12d3-a456-426614174006"),
                miner_uid=10,
                miner_hotkey="miner_1",
                vali_uid=5,
                vali_hotkey="validator_hotkey",
                status="SUCCESS",
                event_id="event_1",
                version_id=UUID("823e4567-e89b-12d3-a456-426614174007"),
                is_final=True,
            ),
            AgentRunSubmission(
                run_id=UUID("923e4567-e89b-12d3-a456-426614174008"),
                miner_uid=20,
                miner_hotkey="miner_2",
                vali_uid=5,
                vali_hotkey="validator_hotkey",
                status="SANDBOX_TIMEOUT",
                event_id="event_2",
                version_id=UUID("a23e4567-e89b-12d3-a456-426614174009"),
                is_final=False,
            ),
        ]

        body = PostAgentRunsRequestBody(runs=runs)

        assert len(body.runs) == 2
        assert body.runs[0].miner_uid == 10
        assert body.runs[1].miner_uid == 20

    def test_from_json(self):
        json_data = {
            "runs": [
                {
                    "run_id": "b23e4567-e89b-12d3-a456-42661417400a",
                    "miner_uid": 40,
                    "miner_hotkey": "miner_hotkey_4",
                    "vali_uid": 5,
                    "vali_hotkey": "validator_hotkey",
                    "status": "SUCCESS",
                    "event_id": "event_abc",
                    "version_id": "c23e4567-e89b-12d3-a456-42661417400b",
                    "is_final": True,
                }
            ]
        }

        body = PostAgentRunsRequestBody.model_validate(json_data)

        assert len(body.runs) == 1
        assert body.runs[0].miner_uid == 40
        assert body.runs[0].event_id == "event_abc"
