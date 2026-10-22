from datetime import datetime, timedelta, timezone

import pytest
from aiosqlite import IntegrityError

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import (
    DkgForecastHashModel,
    DkgForecastModel,
    DkgMinerModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgScoreModel,
    DkgStatementResultModel,
)

MINER = "5MinerHotkey"
OTHER_MINER = "5OtherMinerHotkey"
UID_OF = {MINER: 1, OTHER_MINER: 2}
FIRST_RUN = datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
SECOND_RUN = datetime(2026, 10, 16, 16, 0, tzinfo=timezone.utc)
SPEC_VERSION = 3013


def make_round(round_id: str, deadline_day: int) -> DkgRoundModel:
    return DkgRoundModel(
        round_id=round_id,
        deadline_at=datetime(2026, 10, deadline_day, 0, 0, tzinfo=timezone.utc),
        horizon_end_at=datetime(2026, 11, deadline_day, 0, 0, tzinfo=timezone.utc),
    )


def make_question(question_id: str, round_id: str) -> DkgQuestionModel:
    return DkgQuestionModel(
        question_id=question_id,
        round_id=round_id,
        entity_id="entity-apple",
        entity="Apple",
        counterparty_id=None,
        counterparty=None,
        relationship=None,
    )


def make_hash(miner_hotkey: str, question_id: str) -> DkgForecastHashModel:
    return DkgForecastHashModel(
        miner_uid=UID_OF[miner_hotkey],
        miner_hotkey=miner_hotkey,
        question_id=question_id,
        sealed_sha256=f"sha256-of-{miner_hotkey}-{question_id}",
        received_at=datetime(2026, 10, 12, 17, 30, tzinfo=timezone.utc),
    )


def make_opened_forecast(
    miner_hotkey: str, question_id: str, failed_check: str | None = None
) -> DkgForecastModel:
    return DkgForecastModel(
        miner_uid=UID_OF[miner_hotkey],
        miner_hotkey=miner_hotkey,
        question_id=question_id,
        opened='{"version": 2, "statements": []}',
        invalid_reason=None,
        failed_check=failed_check,
    )


def make_result(
    miner_hotkey: str, question_id: str, statement_position: int, outcome: bool
) -> DkgStatementResultModel:
    return DkgStatementResultModel(
        miner_uid=UID_OF[miner_hotkey],
        miner_hotkey=miner_hotkey,
        question_id=question_id,
        statement_position=statement_position,
        outcome=outcome,
        reason="a reported development is the act the forecast states",
        event_id="event-1" if outcome else None,
        reported_at=datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc) if outcome else None,
        judged_at=datetime(2026, 10, 21, 7, 30, tzinfo=timezone.utc),
    )


def make_score(miner_hotkey: str, scored_at: datetime, score: float) -> DkgScoreModel:
    return DkgScoreModel(
        miner_uid=UID_OF[miner_hotkey],
        miner_hotkey=miner_hotkey,
        scored_at=scored_at,
        score=score,
        spec_version=SPEC_VERSION,
    )


async def count(db_client: DatabaseClient, table: str) -> int:
    row = await db_client.one(f"SELECT COUNT(*) FROM {table}")
    return row[0]


class TestDkgDatabaseOperations:
    @pytest.fixture
    async def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        return DkgDatabaseOperations(db_client=db_client)

    @pytest.fixture
    async def pulled_round(self, dkg_operations: DkgDatabaseOperations) -> None:
        await dkg_operations.save_round(
            make_round("round-1", 13),
            [make_question("question-1", "round-1"), make_question("question-2", "round-1")],
        )
        await dkg_operations.save_forecast_hashes(
            [make_hash(MINER, "question-1"), make_hash(OTHER_MINER, "question-2")]
        )
        await dkg_operations.mark_round_hashes_pulled("round-1")

    def test_init_rejects_anything_but_a_db_client(self):
        with pytest.raises(TypeError):
            DkgDatabaseOperations(db_client="not a client")

    async def test_last_complete_round_deadline_on_an_empty_db(
        self, dkg_operations: DkgDatabaseOperations
    ):
        assert await dkg_operations.get_last_complete_round_deadline() is None

    async def test_a_round_without_its_hashes_does_not_count(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.save_round(make_round("round-1", 13), [])

        assert await dkg_operations.get_last_complete_round_deadline() is None

    async def test_last_complete_round_deadline_skips_later_incomplete_rounds(
        self, dkg_operations: DkgDatabaseOperations
    ):
        for round_id, deadline_day in [("round-1", 11), ("round-2", 12), ("round-3", 13)]:
            await dkg_operations.save_round(make_round(round_id, deadline_day), [])
        await dkg_operations.mark_round_hashes_pulled("round-1")
        await dkg_operations.mark_round_hashes_pulled("round-2")

        assert await dkg_operations.get_last_complete_round_deadline() == datetime(
            2026, 10, 12, 0, 0, tzinfo=timezone.utc
        )

    async def test_save_round_ignores_repeats(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        questions = [make_question("question-1", "round-1"), make_question("question-2", "round-1")]

        await dkg_operations.save_round(make_round("round-1", 13), questions)
        await dkg_operations.save_round(make_round("round-1", 13), questions)

        assert await count(db_client, "dkg_rounds") == 1
        assert await count(db_client, "dkg_questions") == 2

    async def test_save_forecast_hashes_ignores_repeats(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient, pulled_round: None
    ):
        await dkg_operations.save_forecast_hashes([make_hash(MINER, "question-1")])

        assert await count(db_client, "dkg_forecast_hashes") == 2

    async def test_a_hash_needs_a_known_question(self, dkg_operations: DkgDatabaseOperations):
        with pytest.raises(IntegrityError):
            await dkg_operations.save_forecast_hashes([make_hash(MINER, "unknown-question")])

    async def test_get_forecast_hashes_returns_only_the_asked_questions(
        self, dkg_operations: DkgDatabaseOperations, pulled_round: None
    ):
        hashes = await dkg_operations.get_forecast_hashes(["question-1"])

        assert hashes == [make_hash(MINER, "question-1")]

    async def test_get_forecast_hashes_of_no_questions(
        self, dkg_operations: DkgDatabaseOperations, pulled_round: None
    ):
        assert await dkg_operations.get_forecast_hashes([]) == []

    async def test_save_forecasts_overwrites_repeats(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient, pulled_round: None
    ):
        await dkg_operations.save_forecasts(
            [make_opened_forecast(MINER, "question-1")],
            [make_result(MINER, "question-1", 0, outcome=False)],
        )
        await dkg_operations.save_forecasts(
            [
                make_opened_forecast(
                    MINER, "question-1", failed_check="the miner signature does not verify"
                )
            ],
            [
                make_result(MINER, "question-1", 0, outcome=True),
                make_result(MINER, "question-1", 1, outcome=False),
            ],
        )

        forecast = await db_client.one("SELECT failed_check FROM dkg_forecasts")
        results = await db_client.many(
            "SELECT statement_position, outcome, event_id FROM dkg_statement_results ORDER BY statement_position"
        )
        assert await count(db_client, "dkg_forecasts") == 1
        assert forecast[0] == "the miner signature does not verify"
        assert [tuple(row) for row in results] == [(0, 1, "event-1"), (1, 0, None)]

    async def test_an_invalid_forecast_is_stored_with_its_reason(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient, pulled_round: None
    ):
        await dkg_operations.save_forecasts(
            [
                DkgForecastModel(
                    miner_uid=UID_OF[MINER],
                    miner_hotkey=MINER,
                    question_id="question-1",
                    opened=None,
                    invalid_reason="the seal does not open",
                    failed_check=None,
                )
            ],
            [],
        )

        row = await db_client.one("SELECT opened, invalid_reason FROM dkg_forecasts")
        assert tuple(row) == (None, "the seal does not open")

    async def test_a_forecast_needs_its_hash(
        self, dkg_operations: DkgDatabaseOperations, pulled_round: None
    ):
        with pytest.raises(IntegrityError):
            await dkg_operations.save_forecasts([make_opened_forecast(MINER, "question-2")], [])

    @pytest.mark.parametrize(
        "opened, invalid_reason",
        [(None, None), ('{"version": 2, "statements": []}', "the seal does not open")],
    )
    async def test_a_forecast_is_either_opened_or_invalid(
        self,
        dkg_operations: DkgDatabaseOperations,
        pulled_round: None,
        opened: str | None,
        invalid_reason: str | None,
    ):
        forecast = DkgForecastModel(
            miner_uid=UID_OF[MINER],
            miner_hotkey=MINER,
            question_id="question-1",
            opened=opened,
            invalid_reason=invalid_reason,
            failed_check=None,
        )

        with pytest.raises(IntegrityError):
            await dkg_operations.save_forecasts([forecast], [])

    async def test_the_same_hotkey_under_another_uid_is_another_forecast(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient, pulled_round: None
    ):
        registered_again = 99
        await dkg_operations.save_forecast_hashes(
            [make_hash(MINER, "question-1").model_copy(update={"miner_uid": registered_again})]
        )
        await dkg_operations.save_forecasts(
            [
                make_opened_forecast(MINER, "question-1"),
                make_opened_forecast(MINER, "question-1").model_copy(
                    update={"miner_uid": registered_again}
                ),
            ],
            [
                make_result(MINER, "question-1", 0, outcome=True),
                make_result(MINER, "question-1", 0, outcome=False).model_copy(
                    update={"miner_uid": registered_again}
                ),
            ],
        )

        results = await db_client.many(
            "SELECT miner_uid, outcome FROM dkg_statement_results WHERE miner_hotkey = ? ORDER BY miner_uid",
            [MINER],
        )
        assert await count(db_client, "dkg_forecasts") == 2
        assert [tuple(row) for row in results] == [(UID_OF[MINER], 1), (registered_again, 0)]

    async def test_a_forecast_needs_the_hash_of_its_uid(
        self, dkg_operations: DkgDatabaseOperations, pulled_round: None
    ):
        forecast = make_opened_forecast(MINER, "question-1").model_copy(update={"miner_uid": 99})

        with pytest.raises(IntegrityError):
            await dkg_operations.save_forecasts([forecast], [])

    async def test_a_result_needs_its_forecast(
        self, dkg_operations: DkgDatabaseOperations, pulled_round: None
    ):
        with pytest.raises(IntegrityError):
            await dkg_operations.save_forecasts(
                [], [make_result(MINER, "question-1", 0, outcome=True)]
            )

    async def test_last_results_pull_on_an_empty_db(self, dkg_operations: DkgDatabaseOperations):
        assert await dkg_operations.get_last_results_pull() is None

    async def test_last_results_pull_is_the_latest(self, dkg_operations: DkgDatabaseOperations):
        latest = datetime(2026, 10, 14, 15, 0, 30, 123456, tzinfo=timezone.utc)
        for started_at in [datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc), latest]:
            await dkg_operations.record_results_pull(started_at)

        assert await dkg_operations.get_last_results_pull() == latest

    async def test_save_scores_keeps_one_row_per_miner_and_run(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await dkg_operations.save_scores(
            [
                make_score(MINER, FIRST_RUN, 1.41),
                make_score(OTHER_MINER, FIRST_RUN, 0),
            ]
        )
        await dkg_operations.save_scores([make_score(MINER, SECOND_RUN, 2.12)])

        rows = await db_client.many(
            """
                SELECT miner_uid, miner_hotkey, scored_at, score, spec_version, posted_at
                FROM dkg_scores ORDER BY scored_at, miner_uid
            """
        )
        assert [tuple(row) for row in rows] == [
            (1, MINER, "2026-10-16T15:00:00Z", 1.41, 3013, None),
            (2, OTHER_MINER, "2026-10-16T15:00:00Z", 0.0, 3013, None),
            (1, MINER, "2026-10-16T16:00:00Z", 2.12, 3013, None),
        ]

    async def test_a_miner_is_scored_once_per_run(self, dkg_operations: DkgDatabaseOperations):
        score = make_score(MINER, FIRST_RUN, 1.41)
        await dkg_operations.save_scores([score])

        with pytest.raises(IntegrityError):
            await dkg_operations.save_scores([score])

    async def test_save_scores_of_no_miners(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await dkg_operations.save_scores([])

        assert await count(db_client, "dkg_scores") == 0

    async def test_last_scored_at_on_an_empty_db(self, dkg_operations: DkgDatabaseOperations):
        assert await dkg_operations.get_last_scored_at() is None

    async def test_last_scored_at_is_the_latest_moment(self, dkg_operations: DkgDatabaseOperations):
        for scored_at in [SECOND_RUN, FIRST_RUN]:
            await dkg_operations.save_scores([make_score(MINER, scored_at, 1.0)])

        assert await dkg_operations.get_last_scored_at() == SECOND_RUN

    async def test_latest_scores_on_an_empty_db(self, dkg_operations: DkgDatabaseOperations):
        assert await dkg_operations.get_latest_scores(scored_until=SECOND_RUN) == []

    @pytest.mark.parametrize(
        "scored_until, expected_run",
        [
            (SECOND_RUN + timedelta(minutes=5), SECOND_RUN),
            (SECOND_RUN, SECOND_RUN),
            (SECOND_RUN - timedelta(seconds=1), FIRST_RUN),
            (FIRST_RUN - timedelta(seconds=1), None),
        ],
    )
    async def test_latest_scores_are_the_newest_run_scored_until_the_moment(
        self,
        dkg_operations: DkgDatabaseOperations,
        scored_until: datetime,
        expected_run: datetime | None,
    ):
        for scored_at, score in [(FIRST_RUN, 1.0), (SECOND_RUN, 2.0)]:
            await dkg_operations.save_scores(
                [make_score(MINER, scored_at, score), make_score(OTHER_MINER, scored_at, 0.0)]
            )

        scores = await dkg_operations.get_latest_scores(scored_until=scored_until)

        assert {score.scored_at for score in scores} == ({expected_run} if expected_run else set())
        assert len(scores) == (2 if expected_run else 0)

    async def test_unposted_scores_on_an_empty_db(self, dkg_operations: DkgDatabaseOperations):
        assert await dkg_operations.get_unposted_scores() == []

    async def test_unposted_scores_come_oldest_run_first_then_by_uid(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.save_scores(
            [make_score(OTHER_MINER, SECOND_RUN, 0.0), make_score(MINER, SECOND_RUN, 2.0)]
        )
        await dkg_operations.save_scores(
            [make_score(OTHER_MINER, FIRST_RUN, 0.5), make_score(MINER, FIRST_RUN, 1.0)]
        )

        scores = await dkg_operations.get_unposted_scores()

        assert [(score.scored_at, score.miner_uid, score.score) for score in scores] == [
            (FIRST_RUN, 1, 1.0),
            (FIRST_RUN, 2, 0.5),
            (SECOND_RUN, 1, 2.0),
            (SECOND_RUN, 2, 0.0),
        ]

    async def test_unposted_scores_keep_the_version_that_scored_them(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.save_scores(
            [make_score(MINER, FIRST_RUN, 1.0).model_copy(update={"spec_version": 3012})]
        )
        await dkg_operations.save_scores([make_score(MINER, SECOND_RUN, 1.0)])

        scores = await dkg_operations.get_unposted_scores()

        assert [score.spec_version for score in scores] == [3012, 3013]

    async def test_mark_scores_posted_marks_only_that_run(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        for scored_at in [FIRST_RUN, SECOND_RUN]:
            await dkg_operations.save_scores(
                [make_score(MINER, scored_at, 1.0), make_score(OTHER_MINER, scored_at, 0.0)]
            )

        await dkg_operations.mark_scores_posted(FIRST_RUN)

        assert {score.scored_at for score in await dkg_operations.get_unposted_scores()} == {
            SECOND_RUN
        }
        rows = await db_client.many(
            "SELECT scored_at FROM dkg_scores WHERE posted_at IS NOT NULL ORDER BY miner_uid"
        )
        assert [row[0] for row in rows] == ["2026-10-16T15:00:00Z", "2026-10-16T15:00:00Z"]

    async def test_mark_scores_posted_finds_the_run_written_in_another_offset(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.save_scores([make_score(MINER, FIRST_RUN, 1.0)])

        await dkg_operations.mark_scores_posted(FIRST_RUN.astimezone(timezone(timedelta(hours=-3))))

        assert await dkg_operations.get_unposted_scores() == []

    async def test_mark_scores_posted_of_an_unknown_run_changes_nothing(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.save_scores([make_score(MINER, FIRST_RUN, 1.0)])

        await dkg_operations.mark_scores_posted(SECOND_RUN)

        assert len(await dkg_operations.get_unposted_scores()) == 1

    async def test_a_posted_run_still_sets_the_weights(self, dkg_operations: DkgDatabaseOperations):
        await dkg_operations.save_scores([make_score(MINER, FIRST_RUN, 1.0)])
        await dkg_operations.mark_scores_posted(FIRST_RUN)

        scores = await dkg_operations.get_latest_scores(scored_until=FIRST_RUN)

        assert [(score.miner_uid, score.score, score.spec_version) for score in scores] == [
            (1, 1.0, 3013)
        ]

    @pytest.fixture
    async def expired_and_kept_rounds(self, dkg_operations: DkgDatabaseOperations) -> None:
        await dkg_operations.save_round(
            make_round("expired-round", 10),
            [
                make_question("expired-question-1", "expired-round"),
                make_question("expired-question-2", "expired-round"),
            ],
        )
        await dkg_operations.save_round(
            make_round("kept-round", 13),
            [
                make_question("kept-question-1", "kept-round"),
                make_question("kept-question-2", "kept-round"),
            ],
        )
        await dkg_operations.save_forecast_hashes(
            [
                make_hash(MINER, "expired-question-1"),
                make_hash(MINER, "expired-question-2"),
                make_hash(OTHER_MINER, "expired-question-1"),
                make_hash(MINER, "kept-question-1"),
            ]
        )
        invalid_forecast = DkgForecastModel(
            miner_uid=UID_OF[MINER],
            miner_hotkey=MINER,
            question_id="expired-question-2",
            opened=None,
            invalid_reason="the seal does not open",
            failed_check=None,
        )
        await dkg_operations.save_forecasts(
            [
                make_opened_forecast(MINER, "expired-question-1"),
                invalid_forecast,
                make_opened_forecast(MINER, "kept-question-1"),
            ],
            [
                make_result(MINER, "expired-question-1", 0, True),
                make_result(MINER, "expired-question-1", 1, False),
                make_result(MINER, "kept-question-1", 0, True),
            ],
        )

    async def rows_of_each_round(self, db_client: DatabaseClient) -> dict[str, list[int]]:
        rows_of_each_round = {}
        for round_id in ["expired-round", "kept-round"]:
            questions = f"SELECT question_id FROM dkg_questions WHERE round_id = '{round_id}'"
            rows_of_each_round[round_id] = [
                (await db_client.one(query))[0]
                for query in [
                    f"SELECT COUNT(*) FROM dkg_rounds WHERE round_id = '{round_id}'",
                    f"SELECT COUNT(*) FROM ({questions})",
                    f"SELECT COUNT(*) FROM dkg_forecast_hashes WHERE question_id IN ({questions})",
                    f"SELECT COUNT(*) FROM dkg_forecasts WHERE question_id IN ({questions})",
                    f"SELECT COUNT(*) FROM dkg_statement_results WHERE question_id IN ({questions})",
                ]
            ]
        return rows_of_each_round

    @pytest.mark.usefixtures("expired_and_kept_rounds")
    async def test_an_expired_round_goes_with_everything_under_it(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        assert await self.rows_of_each_round(db_client) == {
            "expired-round": [1, 2, 3, 2, 2],
            "kept-round": [1, 2, 1, 1, 1],
        }

        deleted = await dkg_operations.delete_expired(
            expired_until=datetime(2026, 10, 10, tzinfo=timezone.utc), batch_size=4000
        )

        assert deleted == 10
        assert await self.rows_of_each_round(db_client) == {
            "expired-round": [0, 0, 0, 0, 0],
            "kept-round": [1, 2, 1, 1, 1],
        }

    @pytest.mark.usefixtures("expired_and_kept_rounds")
    async def test_a_round_just_after_the_cutoff_is_kept(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        deleted = await dkg_operations.delete_expired(
            expired_until=datetime(2026, 10, 9, 23, 59, 59, tzinfo=timezone.utc), batch_size=4000
        )

        assert deleted == 0
        assert (await self.rows_of_each_round(db_client))["expired-round"] == [1, 2, 3, 2, 2]

    @pytest.mark.usefixtures("expired_and_kept_rounds")
    async def test_the_cutoff_can_be_written_in_another_offset(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await dkg_operations.delete_expired(
            expired_until=datetime(2026, 10, 9, 21, 0, tzinfo=timezone(timedelta(hours=-3))),
            batch_size=4000,
        )

        assert (await self.rows_of_each_round(db_client))["expired-round"] == [0, 0, 0, 0, 0]

    @pytest.mark.usefixtures("expired_and_kept_rounds")
    async def test_small_batches_delete_children_before_parents(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        expired_until = datetime(2026, 10, 10, tzinfo=timezone.utc)
        passes, deleted_in_total = 0, 0
        while deleted := await dkg_operations.delete_expired(
            expired_until=expired_until, batch_size=1
        ):
            passes += 1
            deleted_in_total += deleted

        assert passes > 1
        assert deleted_in_total == 10
        assert await self.rows_of_each_round(db_client) == {
            "expired-round": [0, 0, 0, 0, 0],
            "kept-round": [1, 2, 1, 1, 1],
        }
        assert await db_client.many("PRAGMA foreign_key_check") == []

    async def test_scores_up_to_the_cutoff_are_deleted_without_an_expired_round(
        self, dkg_operations: DkgDatabaseOperations
    ):
        for scored_at in [FIRST_RUN, SECOND_RUN]:
            await dkg_operations.save_scores([make_score(MINER, scored_at, 1.0)])

        deleted = await dkg_operations.delete_expired(expired_until=FIRST_RUN, batch_size=4000)

        assert deleted == 1
        assert await dkg_operations.get_last_scored_at() == SECOND_RUN
        assert len(await dkg_operations.get_unposted_scores()) == 1

    async def test_old_pulls_are_deleted_but_the_latest_is_always_kept(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        pulls = [datetime(2026, 10, day, 12, 0, tzinfo=timezone.utc) for day in [14, 15, 16]]
        for started_at in pulls:
            await dkg_operations.record_results_pull(started_at)

        deleted = await dkg_operations.delete_expired(
            expired_until=datetime(2026, 12, 1, tzinfo=timezone.utc), batch_size=4000
        )

        assert deleted == 2
        assert await count(db_client, "dkg_results_pulls") == 1
        assert await dkg_operations.get_last_results_pull() == pulls[-1]

    async def test_miners_count_on_an_empty_db(self, dkg_operations: DkgDatabaseOperations):
        assert await dkg_operations.get_miners_count() == 0

    async def test_upsert_miners_inserts_new_miners(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await dkg_operations.upsert_miners(
            [
                ("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 100, "ip1", 100),
                ("2", OTHER_MINER, "ip2", "2026-10-02T14:30:00+00:00", 100, "ip2", 100),
            ]
        )

        rows = await db_client.many(
            """
                SELECT miner_uid, miner_hotkey, node_ip, registered_date, blocktime, blocklisted
                FROM miners ORDER BY miner_uid
            """
        )
        assert [tuple(row) for row in rows] == [
            ("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 100, 0),
            ("2", OTHER_MINER, "ip2", "2026-10-02T14:30:00+00:00", 100, 0),
        ]
        assert await dkg_operations.get_miners_count() == 2

    async def test_upsert_miners_updates_the_ip_and_blocktime_of_a_known_miner(
        self, dkg_operations: DkgDatabaseOperations, db_client: DatabaseClient
    ):
        await dkg_operations.upsert_miners(
            [("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 100, "ip1", 100)]
        )

        await dkg_operations.upsert_miners(
            [("1", MINER, "ip9", "2026-10-05T00:00:00+00:00", 200, "ip9", 200)]
        )

        rows = await db_client.many("SELECT node_ip, registered_date, blocktime FROM miners")
        assert [tuple(row) for row in rows] == [("ip9", "2026-10-01T14:30:00+00:00", 200)]

    async def test_the_same_hotkey_under_another_uid_is_another_miner_row(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.upsert_miners(
            [
                ("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 100, "ip1", 100),
                ("99", MINER, "ip1", "2026-10-09T14:30:00+00:00", 100, "ip1", 100),
            ]
        )

        assert await dkg_operations.get_miners_count() == 2

    async def test_the_latest_sync_gives_the_current_miners(
        self, dkg_operations: DkgDatabaseOperations
    ):
        await dkg_operations.upsert_miners(
            [
                ("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 100, "ip1", 100),
                ("2", OTHER_MINER, "ip2", "2026-10-02T14:30:00+00:00", 100, "ip2", 100),
            ]
        )
        await dkg_operations.upsert_miners(
            [("1", MINER, "ip1", "2026-10-01T14:30:00+00:00", 200, "ip1", 200)]
        )

        assert await dkg_operations.get_current_miners() == [
            DkgMinerModel(miner_uid=1, miner_hotkey=MINER)
        ]

    async def test_vacuum_database_runs_on_a_migrated_db(
        self, dkg_operations: DkgDatabaseOperations
    ):
        assert await dkg_operations.vacuum_database(500) is None
