import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.operations import DatabaseOperations
from neurons.validator.db.tests.test_utils import TestDbOperationsBase
from neurons.validator.models.event import EventsModel, EventStatus
from neurons.validator.models.reasoning import ReasoningModel
from neurons.validator.models.score import ScoresModel


class TestDbOperationsPart3(TestDbOperationsBase):
    async def test_upsert_reasonings(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test basic upsert of reasonings"""
        reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="event2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=True,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        # Verify the reasonings were inserted
        rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
                ORDER BY
                    ROWID ASC
            """
        )

        assert len(rows) == 2
        assert rows[0] == (
            "event1",
            1,
            "hotkey1",
            "Test reasoning 1",
            0,
            ANY,
            ANY,
        )
        assert rows[1] == ("event2", 2, "hotkey2", "Test reasoning 2", 1, ANY, ANY)

    async def test_upsert_reasonings_update_existing(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test updating existing reasonings"""
        # First insert
        initial_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Initial reasoning",
                exported=False,
            )
        ]

        await db_operations.upsert_reasonings(initial_reasonings)

        initial_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(initial_rows) == 1
        assert initial_rows[0] == ("event1", 1, "hotkey1", "Initial reasoning", 0, ANY, ANY)

        # Update the same reasoning
        updated_reasonings = [
            ReasoningModel(
                event_id="event1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Updated reasoning",
                exported=True,
            )
        ]

        # Delay upsert for updated_at to be different
        await asyncio.sleep(1)

        await db_operations.upsert_reasonings(updated_reasonings)

        # Verify the reasoning was updated
        final_rows = await db_client.many(
            """
                SELECT
                    event_id,
                    miner_uid,
                    miner_hotkey,
                    reasoning,
                    exported,
                    created_at,
                    updated_at
                FROM
                    reasoning
            """
        )

        assert len(final_rows) == 1
        # Exported is not updated
        assert final_rows[0] == ("event1", 1, "hotkey1", "Updated reasoning", 0, ANY, ANY)

        # Verify created_at and updated_at timestamps
        assert initial_rows[0][5] == final_rows[0][5]
        assert initial_rows[0][6] < final_rows[0][6]

    async def test_upsert_reasonings_empty_list(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        """Test upserting an empty list of reasonings"""
        # Attempt to upsert an empty list
        await db_operations.upsert_reasonings([])

        # Verify no reasonings were inserted
        rows = await db_client.many(
            """
                SELECT COUNT(*) FROM reasoning
            """
        )

        assert rows[0][0] == 0

    async def test_delete_reasonings_orphan(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        reasonings = [
            ReasoningModel(
                event_id="non_existent_event_1",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                # Not exported
                exported=False,
            ),
            ReasoningModel(
                event_id="non_existent_event_2",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=True,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 2

        # Delete orphan reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,), (2,)]

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 0

    async def test_delete_reasonings_resolved_old(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        old_date = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()

        new_date = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        events = [
            EventsModel(
                unique_event_id="old_event",
                event_id="old_event",
                market_type="market_type",
                event_type="type",
                description="Old event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=old_date,
                processed=True,
            ),
            EventsModel(
                unique_event_id="new_event",
                event_id="new_event",
                market_type="market_type",
                event_type="type",
                description="New event",
                outcome="1",
                status=EventStatus.SETTLED,
                metadata='{"key": "value"}',
                resolved_at=new_date,
                processed=True,
            ),
        ]

        await db_operations.upsert_events(events)

        reasonings = [
            ReasoningModel(
                event_id="old_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="new_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=False,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 2

        # Delete old reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,)]

        # Verify reasoning for new_event remains
        result = await db_client.one("SELECT event_id FROM reasoning")
        assert result[0] == "new_event"

    async def test_delete_reasonings_discarded_and_deleted_events(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        events = [
            EventsModel(
                unique_event_id="discarded_event",
                event_id="discarded_event",
                market_type="market_type",
                event_type="type",
                description="Discarded event",
                outcome=None,
                status=EventStatus.DISCARDED,
                metadata='{"key": "value"}',
                resolved_at=None,
            ),
            EventsModel(
                unique_event_id="deleted_event",
                event_id="deleted_event",
                market_type="market_type",
                event_type="type",
                description="Deleted event",
                outcome=None,
                status=EventStatus.DELETED,
                metadata='{"key": "value"}',
                resolved_at=None,
            ),
        ]

        await db_operations.upsert_events(events=events)

        # Insert reasonings for the discarded event
        reasonings = [
            ReasoningModel(
                event_id="discarded_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
            ReasoningModel(
                event_id="discarded_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 2",
                exported=False,
            ),
            ReasoningModel(
                event_id="deleted_event",
                miner_uid=2,
                miner_hotkey="hotkey2",
                reasoning="Test reasoning 3",
                exported=False,
            ),
        ]

        await db_operations.upsert_reasonings(reasonings=reasonings)

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 3

        # Delete discarded reasonings
        deleted = await db_operations.delete_reasonings(batch_size=100)

        # Deleted in order
        assert deleted == [(1,), (2,), (3,)]

        result = await db_client.one("SELECT COUNT(*) FROM reasoning")
        assert result[0] == 0

    async def test_delete_reasonings_no_results(
        self, db_operations: DatabaseOperations, db_client: DatabaseClient
    ):
        resolved_at_recently = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

        event = EventsModel(
            unique_event_id="recent_event",
            event_id="recent_event",
            market_type="market_type",
            event_type="type",
            description="Recent event",
            outcome="1",
            status=EventStatus.PENDING,
            metadata='{"key": "value"}',
            resolved_at=resolved_at_recently,
            processed=True,
        )
        await db_operations.upsert_events([event])

        reasonings = [
            ReasoningModel(
                event_id="recent_event",
                miner_uid=1,
                miner_hotkey="hotkey1",
                reasoning="Test reasoning 1",
                exported=False,
            ),
        ]
        await db_operations.upsert_reasonings(reasonings)

        # Verify reasonings were inserted
        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 1

        # Try to delete reasonings (should not delete any as they're too recent)
        deleted = await db_operations.delete_reasonings(batch_size=100)

        assert len(deleted) == 0

        # Verify reasonings were not deleted
        result = await db_client.one("SELECT COUNT(*) FROM reasoning")

        assert result[0] == 1

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
