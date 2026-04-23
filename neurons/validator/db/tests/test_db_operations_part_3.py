import json
from datetime import datetime, timedelta, timezone

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.agent_runs import AgentRunsModel, AgentRunStatus
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.miner_agent import MinerAgentsModel
from neurons.validator.models.reasoning import ReasoningForExport
from neurons.validator.models.score import ScoresModel
from neurons.validator.models.sources import (
    ImpactBucket,
    PersistenceBucket,
    SourceDirection,
    SourceItem,
    SourcesForExport,
)


class TestDbOperationsPart3(TestDbOperationsBase):
    async def test_get_events_last_deleted_at(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="market_type",
                event_type="type",
                description="Recent event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
        ]

        current_time = datetime.now(timezone.utc)
        future_time = current_time + timedelta(days=1)

        await db_operations.upsert_events(events=events)

        # Delete events 1 and 3
        await db_operations.delete_event(event_id="event1", deleted_at=current_time)
        await db_operations.delete_event(event_id="event3", deleted_at=future_time)

        result = await db_operations.get_events_last_deleted_at()

        assert result == future_time.isoformat().replace("T", " ")

    async def test_get_events_last_deleted_at_no_events(self, db_operations: DatabaseOperations):
        result = await db_operations.get_events_last_deleted_at()

        assert result is None

    async def test_delete_events_hard_delete(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        recent_deleted_datetime = datetime.now(timezone.utc) - timedelta(days=10)
        old_deleted_datetime = datetime.now(timezone.utc) - timedelta(days=15)

        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="market_type",
                event_type="type",
                description="Pending event",
                outcome="1",
                status=EventStatus.PENDING,
                metadata='{"key": "value"}',
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="market_type",
                event_type="type",
                description="Recent deleted event",
                outcome="1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                deleted_at=recent_deleted_datetime,
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="market_type",
                event_type="type",
                description="Old deleted event",
                outcome="1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                deleted_at=old_deleted_datetime,
            ),
            EventsModel(
                unique_event_id="unique4",
                event_id="event4",
                market_type="market_type",
                event_type="type",
                description="Old deleted event",
                outcome="1",
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                deleted_at=old_deleted_datetime,
            ),
        ]

        await db_operations.upsert_events(events=events)

        # Test delete events - batch size 1
        deleted = await db_operations.delete_events_hard_delete(batch_size=1)
        assert deleted == [(3,)]

        # Test delete remaining events
        deleted = await db_operations.delete_events_hard_delete(batch_size=10)
        assert deleted == [(4,)]

        # Verify remaining events
        remaining = await db_client.many("SELECT event_id FROM events")

        assert remaining == [
            ("event1",),
            ("event2",),
        ]

    async def test_get_predictions_ranked(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="event1",
                event_id="event1",
                market_type="market_type1",
                event_type="type1",
                description="First event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value1"}',
                resolved_at="2024-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="event2",
                event_id="event2",
                market_type="market_type2",
                event_type="type2",
                description="Second event",
                outcome="0",
                status=EventStatus.SETTLED,
                metadata='{"key": "value2"}',
                resolved_at="2024-12-31T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        # Create test scores for each event
        scores = []

        for event in events:
            for i in range(3):
                # 3 miners per event
                miner_uid = i + 1
                scores.append(
                    ScoresModel(
                        event_id=event.event_id,
                        miner_uid=miner_uid,
                        miner_hotkey=f"hk_{miner_uid}",
                        track="MAIN",
                        prediction=miner_uid / 10,
                        event_score=0.5,
                        spec_version=1,
                    )
                )

        await db_operations.insert_scores(scores)

        # Test with moving window of 1 (should return top 1 events)
        result_small_window = await db_operations.get_predictions_ranked(moving_window=1)

        # Should return 3 rows (1 events × 3 miners)
        assert result_small_window == [
            (
                "event2",
                1,
                "0",
                1,
                "hk_1",
                0.1,
            ),
            (
                "event2",
                1,
                "0",
                2,
                "hk_2",
                0.2,
            ),
            (
                "event2",
                1,
                "0",
                3,
                "hk_3",
                0.3,
            ),
        ]

        # Test with moving window of 100
        result_large_window = await db_operations.get_predictions_ranked(moving_window=100)

        # Should return 6 rows (2 events × 3 miners)
        assert result_large_window == [
            (
                "event2",
                1,
                "0",
                1,
                "hk_1",
                0.1,
            ),
            (
                "event2",
                1,
                "0",
                2,
                "hk_2",
                0.2,
            ),
            (
                "event2",
                1,
                "0",
                3,
                "hk_3",
                0.3,
            ),
            (
                "event1",
                2,
                "1",
                1,
                "hk_1",
                0.1,
            ),
            (
                "event1",
                2,
                "1",
                2,
                "hk_2",
                0.2,
            ),
            (
                "event1",
                2,
                "1",
                3,
                "hk_3",
                0.3,
            ),
        ]

    async def test_get_predictions_ranked_no_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        result = await db_operations.get_predictions_ranked(moving_window=2)

        assert len(result) == 0

    async def test_get_events(self, db_operations: DatabaseOperations):
        events = [
            EventsModel(
                unique_event_id="unique1",
                event_id="event1",
                market_type="truncated_market1",
                event_type="market1",
                description="desc1",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value1"}',
                created_at="2000-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique2",
                event_id="event2",
                market_type="truncated_market2",
                event_type="market2",
                description="desc2",
                outcome="0",
                status=EventStatus.PENDING,
                metadata='{"key": "value2"}',
                created_at="2012-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
            EventsModel(
                unique_event_id="unique3",
                event_id="event3",
                market_type="truncated_market3",
                event_type="market3",
                description="desc3",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value3"}',
                created_at="2015-12-02T14:30:00+00:00",
                cutoff="2000-12-30T14:30:00+00:00",
            ),
        ]

        await db_operations.upsert_events(events=events)

        unique_event_ids = ["unique1", "unique3"]
        result = await db_operations.get_events(unique_event_ids=unique_event_ids)

        # Verify results
        assert len(result) == 2
        assert result[0].event_id == events[0].event_id
        assert result[1].event_id == events[2].event_id

    async def test_get_events_empty_events_ids_list(self, db_operations: DatabaseOperations):
        unique_event_ids = []

        result = await db_operations.get_events(unique_event_ids=unique_event_ids)

        assert len(result) == 0

    async def _setup_agent_run(
        self,
        db_operations: DatabaseOperations,
        run_id: str,
        unique_event_id: str,
        miner_uid: int,
        miner_hotkey: str,
    ) -> None:
        event = EventsModel(
            unique_event_id=unique_event_id,
            event_id=f"event_{unique_event_id}",
            market_type="market_type",
            event_type="type",
            description="Test event",
            outcome=None,
            status=EventStatus.PENDING,
            metadata="{}",
            created_at="2024-01-01T00:00:00+00:00",
            cutoff="2024-12-31T23:59:59+00:00",
        )
        await db_operations.upsert_events([event])

        version_id = f"v-{run_id}"
        agent = MinerAgentsModel(
            version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            agent_name="TestAgent",
            version_number=1,
            file_path=f"/data/agents/{miner_uid}/test.py",
            pulled_at=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            created_at=datetime(2024, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        )
        await db_operations.upsert_miner_agents([agent])

        run = AgentRunsModel(
            run_id=run_id,
            unique_event_id=unique_event_id,
            agent_version_id=version_id,
            miner_uid=miner_uid,
            miner_hotkey=miner_hotkey,
            track="MAIN",
            status=AgentRunStatus.SUCCESS,
            exported=False,
            is_final=True,
        )
        await db_operations.upsert_agent_runs([run])

    async def test_insert_reasoning(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_1", "Test reasoning text")

        rows = await db_client.many("SELECT run_id, reasoning, exported FROM reasoning")

        assert len(rows) == 1
        assert rows[0] == ("run_1", "Test reasoning text", 0)

    async def test_insert_reasoning_upsert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_1", "Original reasoning")
        await db_operations.insert_reasoning("run_1", "Updated reasoning")

        rows = await db_client.many("SELECT run_id, reasoning FROM reasoning")

        assert len(rows) == 1
        assert rows[0] == ("run_1", "Updated reasoning")

    async def test_get_reasonings_for_export(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_reasoning("run_1", "Reasoning 1")
        await db_operations.insert_reasoning("run_2", "[NO_REASONING - SUCCESS]")

        result = await db_operations.get_reasonings_for_export(limit=100)

        assert len(result) == 2
        assert all(isinstance(r, ReasoningForExport) for r in result)

        assert result[0].run_id == "run_1"
        assert result[0].reasoning == "Reasoning 1"
        assert result[0].event_id == "event_1"
        assert result[0].miner_uid == 10
        assert result[0].miner_hotkey == "hotkey_1"
        assert result[0].track == "MAIN"

        assert result[1].run_id == "run_2"
        assert result[1].reasoning == "[NO_REASONING - SUCCESS]"
        assert result[1].event_id == "event_2"

    async def test_get_reasonings_for_export_empty(self, db_operations: DatabaseOperations):
        result = await db_operations.get_reasonings_for_export(limit=100)
        assert len(result) == 0

    async def test_get_reasonings_for_export_skips_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_reasoning("run_1", "Reasoning 1")
        await db_operations.insert_reasoning("run_2", "Reasoning 2")

        await db_operations.mark_reasonings_as_exported(run_ids=["run_1"])

        result = await db_operations.get_reasonings_for_export(limit=100)

        assert len(result) == 1
        assert result[0].run_id == "run_2"

    async def test_mark_reasonings_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_1", "Reasoning 1")
        await db_operations.insert_reasoning("run_2", "Reasoning 2")
        await db_operations.insert_reasoning("run_3", "Reasoning 3")

        await db_operations.mark_reasonings_as_exported(run_ids=["run_1", "run_3"])

        rows = await db_client.many("SELECT run_id, exported FROM reasoning ORDER BY run_id")

        assert rows == [("run_1", 1), ("run_2", 0), ("run_3", 1)]

    async def test_mark_reasonings_as_exported_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_1", "Reasoning 1")

        await db_operations.mark_reasonings_as_exported(run_ids=[])

        rows = await db_client.many("SELECT exported FROM reasoning")
        assert rows == [(0,)]

    async def test_delete_reasonings(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_old", "Old reasoning")
        await db_operations.insert_reasoning("run_new", "New reasoning")

        await db_operations.mark_reasonings_as_exported(run_ids=["run_old", "run_new"])

        # Backdate the old one
        await db_client.update(
            """
                UPDATE reasoning
                SET created_at = datetime(CURRENT_TIMESTAMP, '-8 day')
                WHERE run_id = ?
            """,
            ["run_old"],
        )

        deleted = await db_operations.delete_reasonings(batch_size=100)

        assert len(deleted) == 1

        rows = await db_client.many("SELECT run_id FROM reasoning")
        assert rows == [("run_new",)]

    async def test_delete_reasonings_skips_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_reasoning("run_1", "Reasoning 1")

        # Backdate but don't export
        await db_client.update(
            """
                UPDATE reasoning
                SET created_at = datetime(CURRENT_TIMESTAMP, '-8 day')
                WHERE run_id = ?
            """,
            ["run_1"],
        )

        deleted = await db_operations.delete_reasonings(batch_size=100)

        assert len(deleted) == 0

        rows = await db_client.many("SELECT run_id FROM reasoning")
        assert rows == [("run_1",)]

    def _build_source(self, url: str = "https://example.com") -> SourceItem:
        return SourceItem(
            url=url,
            source_type="news",
            direction=SourceDirection.UP,
            source_timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            impact_bucket=ImpactBucket.HIGH,
            persistence_bucket=PersistenceBucket.MEDIUM,
            reasoning="supports outcome",
        )

    async def test_insert_sources(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        sources = [self._build_source("https://a.com"), self._build_source("https://b.com")]

        await db_operations.insert_sources("run_1", sources)

        rows = await db_client.many("SELECT run_id, sources, exported FROM sources")

        assert len(rows) == 1
        assert rows[0][0] == "run_1"
        assert rows[0][2] == 0

        parsed = json.loads(rows[0][1])
        assert len(parsed) == 2
        assert parsed[0]["url"] == "https://a.com"
        assert parsed[1]["url"] == "https://b.com"

    async def test_insert_sources_upsert(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_sources("run_1", [self._build_source("https://a.com")])
        await db_operations.insert_sources("run_1", [self._build_source("https://b.com")])

        rows = await db_client.many("SELECT run_id, sources FROM sources")

        assert len(rows) == 1
        parsed = json.loads(rows[0][1])
        assert len(parsed) == 1
        assert parsed[0]["url"] == "https://b.com"

    async def test_get_sources_for_export(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_sources(
            "run_1", [self._build_source("https://a.com"), self._build_source("https://b.com")]
        )
        await db_operations.insert_sources("run_2", [self._build_source("https://c.com")])

        result = await db_operations.get_sources_for_export(limit=100)

        assert len(result) == 2
        assert all(isinstance(r, SourcesForExport) for r in result)

        assert result[0].run_id == "run_1"
        assert len(result[0].sources) == 2
        assert result[0].sources[0].url == "https://a.com"
        assert result[0].event_id == "event_1"
        assert result[0].miner_uid == 10
        assert result[0].miner_hotkey == "hotkey_1"
        assert result[0].track == "MAIN"

        assert result[1].run_id == "run_2"
        assert len(result[1].sources) == 1
        assert result[1].sources[0].url == "https://c.com"
        assert result[1].event_id == "event_2"

    async def test_get_sources_for_export_empty(self, db_operations: DatabaseOperations):
        result = await db_operations.get_sources_for_export(limit=100)
        assert len(result) == 0

    async def test_get_sources_for_export_skips_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await self._setup_agent_run(db_operations, "run_1", "event_1", 10, "hotkey_1")
        await self._setup_agent_run(db_operations, "run_2", "event_2", 20, "hotkey_2")

        await db_operations.insert_sources("run_1", [self._build_source()])
        await db_operations.insert_sources("run_2", [self._build_source()])

        await db_operations.mark_sources_as_exported(run_ids=["run_1"])

        result = await db_operations.get_sources_for_export(limit=100)

        assert len(result) == 1
        assert result[0].run_id == "run_2"

    async def test_mark_sources_as_exported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_sources("run_1", [self._build_source()])
        await db_operations.insert_sources("run_2", [self._build_source()])
        await db_operations.insert_sources("run_3", [self._build_source()])

        await db_operations.mark_sources_as_exported(run_ids=["run_1", "run_3"])

        rows = await db_client.many("SELECT run_id, exported FROM sources ORDER BY run_id")

        assert rows == [("run_1", 1), ("run_2", 0), ("run_3", 1)]

    async def test_mark_sources_as_exported_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_sources("run_1", [self._build_source()])

        await db_operations.mark_sources_as_exported(run_ids=[])

        rows = await db_client.many("SELECT exported FROM sources")
        assert rows == [(0,)]

    async def test_delete_sources(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_sources("run_old", [self._build_source()])
        await db_operations.insert_sources("run_new", [self._build_source()])

        await db_operations.mark_sources_as_exported(run_ids=["run_old", "run_new"])

        # Backdate the old one
        await db_client.update(
            """
                UPDATE sources
                SET created_at = datetime(CURRENT_TIMESTAMP, '-8 day')
                WHERE run_id = ?
            """,
            ["run_old"],
        )

        deleted = await db_operations.delete_sources(batch_size=100)

        assert len(deleted) == 1

        rows = await db_client.many("SELECT run_id FROM sources")
        assert rows == [("run_new",)]

    async def test_delete_sources_skips_unexported(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        await db_operations.insert_sources("run_1", [self._build_source()])

        # Backdate but don't export
        await db_client.update(
            """
                UPDATE sources
                SET created_at = datetime(CURRENT_TIMESTAMP, '-8 day')
                WHERE run_id = ?
            """,
            ["run_1"],
        )

        deleted = await db_operations.delete_sources(batch_size=100)

        assert len(deleted) == 0

        rows = await db_client.many("SELECT run_id FROM sources")
        assert rows == [("run_1",)]
