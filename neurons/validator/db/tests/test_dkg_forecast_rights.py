import json
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pytest

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgForecastHashModel,
    DkgForecastModel,
    DkgForecastRightsModel,
    DkgMinerModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgStatementResultModel,
)

MINER = (1, "5MinerHotkey")
OTHER_MINER = (2, "5OtherMinerHotkey")
SAME_HOTKEY_REGISTERED_AGAIN = (99, "5MinerHotkey")
QUESTIONS = ["question-1", "question-2", "question-3"]
RECEIVED = datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc)
JUDGED = datetime(2026, 10, 20, 7, 30, tzinfo=timezone.utc)
WINDOW_START = datetime(2026, 10, 15, tzinfo=timezone.utc)
MOMENT = datetime(2026, 10, 25, 15, 0, tzinfo=timezone.utc)


class Right(NamedTuple):
    position: int
    event_id: str
    judged_at: datetime = JUDGED


async def add_forecast(
    dkg_operations: DkgDatabaseOperations,
    miner: tuple[int, str],
    question_id: str,
    statements: int,
    rights: list[Right],
    received_at: datetime = RECEIVED,
    failed_check: str | None = None,
) -> None:
    miner_uid, miner_hotkey = miner
    await dkg_operations.save_forecast_hashes(
        [
            DkgForecastHashModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                sealed_sha256=f"sha256-of-{miner_uid}-{question_id}",
                received_at=received_at,
            )
        ]
    )
    opened = {
        "version": 2,
        "statements": [
            {"head": "Apple", "relation": "INVESTS_IN", "tail": f"Company {position}"}
            for position in range(statements)
        ],
    }
    right_at = {right.position: right for right in rights}
    await dkg_operations.save_forecasts(
        [
            DkgForecastModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                opened=json.dumps(opened),
                invalid_reason=None,
                failed_check=failed_check,
            )
        ],
        [
            DkgStatementResultModel(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                question_id=question_id,
                statement_position=position,
                outcome=position in right_at,
                reason="judged",
                event_id=right_at[position].event_id if position in right_at else None,
                reported_at=None,
                judged_at=right_at[position].judged_at if position in right_at else JUDGED,
            )
            for position in range(statements)
        ],
    )


def counted(miner: tuple[int, str], rights: int, statements: int) -> tuple[int, str, int, int]:
    return (*miner, rights, statements)


async def forecast_rights(dkg_operations: DkgDatabaseOperations) -> list[tuple[int, str, int, int]]:
    return sorted(
        (forecast.miner_uid, forecast.miner_hotkey, forecast.rights, forecast.statements)
        for forecast in await dkg_operations.get_forecast_rights(
            judged_after=WINDOW_START, judged_before=MOMENT
        )
    )


class TestCurrentMiners:
    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        return DkgDatabaseOperations(db_client=db_client)

    async def test_the_current_miners_are_the_ones_of_the_latest_sync(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await db_client.insert_many(
            "INSERT INTO miners (miner_uid, miner_hotkey, blocktime) VALUES (?, ?, ?)",
            [
                ("1", "5MinerHotkey", 200),
                ("2", "5OtherMinerHotkey", 200),
                ("1", "5Deregistered", 150),
            ],
        )

        miners = await dkg_operations.get_current_miners()

        assert sorted(miners, key=lambda miner: miner.miner_uid) == [
            DkgMinerModel(miner_uid=1, miner_hotkey="5MinerHotkey"),
            DkgMinerModel(miner_uid=2, miner_hotkey="5OtherMinerHotkey"),
        ]

    async def test_no_current_miners_before_the_first_sync(
        self, dkg_operations: DkgDatabaseOperations
    ):
        assert await dkg_operations.get_current_miners() == []


class TestForecastRights:
    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        dkg_operations = DkgDatabaseOperations(db_client=db_client)
        await dkg_operations.save_round(
            DkgRoundModel(
                round_id="round-1",
                deadline_at=datetime(2026, 10, 13, tzinfo=timezone.utc),
                horizon_end_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
            ),
            [
                DkgQuestionModel(
                    question_id=question_id,
                    round_id="round-1",
                    entity_id="entity-apple",
                    entity="Apple",
                    counterparty_id=None,
                    counterparty=None,
                    relationship=None,
                )
                for question_id in QUESTIONS
            ],
        )
        return dkg_operations

    async def test_a_right_counts_with_every_statement_of_its_forecast(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", 3, [Right(0, "event-1")])

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 3)]
        assert await dkg_operations.get_forecast_rights(
            judged_after=WINDOW_START, judged_before=MOMENT
        ) == [
            DkgForecastRightsModel(miner_uid=1, miner_hotkey="5MinerHotkey", rights=1, statements=3)
        ]

    async def test_a_forecast_with_nothing_right_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", 2, [])

        assert await forecast_rights(dkg_operations) == []

    async def test_rights_through_different_events_all_count(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 3, [Right(0, "event-1"), Right(2, "event-2")]
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 2, 3)]

    async def test_the_same_event_twice_in_a_forecast_counts_once_and_leaves_the_statements(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 3, [Right(0, "event-1"), Right(1, "event-1")]
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_the_same_event_in_another_question_counts_in_the_forecast_received_first(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")], received_at=RECEIVED
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            3,
            [Right(0, "event-1"), Right(1, "event-2")],
            received_at=RECEIVED + timedelta(hours=1),
        )

        assert await forecast_rights(dkg_operations) == [
            counted(MINER, 1, 1),
            counted(MINER, 1, 2),
        ]

    async def test_a_later_forecast_right_only_through_a_counted_event_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")], received_at=RECEIVED
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [Right(0, "event-1")],
            received_at=RECEIVED + timedelta(days=1),
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 1)]

    async def test_the_time_received_decides_not_the_question(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1")],
            received_at=RECEIVED + timedelta(hours=1),
        )
        await add_forecast(
            dkg_operations, MINER, "question-2", 2, [Right(0, "event-1")], received_at=RECEIVED
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_forecasts_received_at_the_same_time_go_by_question(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-2", 1, [Right(0, "event-1")])
        await add_forecast(dkg_operations, MINER, "question-1", 2, [Right(0, "event-1")])

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_each_miner_keeps_its_own_right_through_the_same_event(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")])
        await add_forecast(dkg_operations, OTHER_MINER, "question-1", 2, [Right(0, "event-1")])

        assert await forecast_rights(dkg_operations) == [
            counted(MINER, 1, 1),
            counted(OTHER_MINER, 1, 2),
        ]

    async def test_the_same_hotkey_under_another_uid_is_another_miner(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")])
        await add_forecast(
            dkg_operations, SAME_HOTKEY_REGISTERED_AGAIN, "question-1", 2, [Right(0, "event-1")]
        )

        assert await forecast_rights(dkg_operations) == [
            counted(MINER, 1, 1),
            counted(SAME_HOTKEY_REGISTERED_AGAIN, 1, 2),
        ]

    async def test_a_forecast_that_failed_a_check_does_not_count(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1")],
            failed_check="the miner signature does not verify",
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_a_forecast_that_failed_a_check_does_not_take_the_event_from_a_later_one(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1")],
            received_at=RECEIVED,
            failed_check="the sealed forecast does not match its hash",
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [Right(0, "event-1")],
            received_at=RECEIVED + timedelta(hours=1),
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_a_forecast_whose_rights_were_judged_before_the_window_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1", judged_at=WINDOW_START - timedelta(days=1))],
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_a_right_judged_exactly_at_the_window_start_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1", judged_at=WINDOW_START)]
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_only_the_rights_judged_in_the_window_count_over_every_statement(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            3,
            [
                Right(0, "event-1", judged_at=WINDOW_START - timedelta(days=10)),
                Right(1, "event-2", judged_at=WINDOW_START + timedelta(days=1)),
            ],
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 3)]

    async def test_a_right_judged_one_second_into_the_window_counts(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1", judged_at=WINDOW_START + timedelta(seconds=1))],
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 1)]

    async def test_of_two_rights_either_side_of_the_window_start_only_the_inside_one_counts(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            2,
            [
                Right(0, "event-1", judged_at=WINDOW_START),
                Right(1, "event-2", judged_at=WINDOW_START + timedelta(seconds=1)),
            ],
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_every_right_of_a_forecast_in_the_window_counts(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            2,
            [
                Right(0, "event-1", judged_at=WINDOW_START + timedelta(days=1)),
                Right(1, "event-2", judged_at=MOMENT - timedelta(minutes=1)),
            ],
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 2, 2)]

    async def test_an_event_paid_by_a_right_that_left_the_window_is_not_paid_again(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1", judged_at=WINDOW_START - timedelta(days=5))],
            received_at=RECEIVED,
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [Right(0, "event-1")],
            received_at=RECEIVED + timedelta(hours=1),
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_a_repeat_judged_before_the_window_still_leaves_the_statements(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")], received_at=RECEIVED
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            3,
            [
                Right(0, "event-1", judged_at=WINDOW_START - timedelta(days=3)),
                Right(1, "event-2"),
            ],
            received_at=RECEIVED + timedelta(hours=1),
        )

        assert await forecast_rights(dkg_operations) == [
            counted(MINER, 1, 1),
            counted(MINER, 1, 2),
        ]

    async def test_each_forecast_keeps_its_own_window(self, dkg_operations: DkgDatabaseOperations):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            2,
            [Right(0, "event-1", judged_at=WINDOW_START - timedelta(days=1))],
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [Right(0, "event-2", judged_at=WINDOW_START + timedelta(days=1))],
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]

    async def test_a_repeated_event_does_not_keep_a_forecast_in_the_window(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1")], received_at=RECEIVED
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [
                Right(0, "event-2", judged_at=WINDOW_START - timedelta(days=10)),
                Right(1, "event-1"),
            ],
            received_at=RECEIVED + timedelta(hours=1),
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 1)]

    async def test_a_right_judged_after_the_moment_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1", judged_at=MOMENT + timedelta(minutes=45))],
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_a_right_judged_exactly_at_the_moment_is_left_out(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations, MINER, "question-1", 1, [Right(0, "event-1", judged_at=MOMENT)]
        )

        assert await forecast_rights(dkg_operations) == []

    async def test_a_right_judged_after_the_moment_does_not_take_its_event_from_an_earlier_one(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await add_forecast(
            dkg_operations,
            MINER,
            "question-1",
            1,
            [Right(0, "event-1", judged_at=MOMENT + timedelta(days=1))],
            received_at=RECEIVED,
        )
        await add_forecast(
            dkg_operations,
            MINER,
            "question-2",
            2,
            [Right(0, "event-1")],
            received_at=RECEIVED + timedelta(days=2),
        )

        assert await forecast_rights(dkg_operations) == [counted(MINER, 1, 2)]
