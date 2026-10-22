from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from neurons.validator.db.client import DatabaseClient
from neurons.validator.db.dkg_operations import DkgDatabaseOperations
from neurons.validator.models.dkg import DkgScoreModel
from neurons.validator.models.dkg_client import MinerScore, ScoringRun, StoredScores
from neurons.validator.numinous_client.dkg_client import DkgClient
from neurons.validator.tasks.post_dkg_scores import PostDkgScores
from neurons.validator.utils.logger.logger import NuminousLogger

MINER = "5MinerHotkey"
OTHER_MINER = "5OtherMinerHotkey"
UID_OF = {MINER: 1, OTHER_MINER: 2}
FIRST_RUN = datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc)
SECOND_RUN = FIRST_RUN + timedelta(days=1)


def make_score(
    miner_hotkey: str, scored_at: datetime, score: float, spec_version: int
) -> DkgScoreModel:
    return DkgScoreModel(
        miner_uid=UID_OF[miner_hotkey],
        miner_hotkey=miner_hotkey,
        scored_at=scored_at,
        score=score,
        spec_version=spec_version,
    )


def rejected(status: int) -> ClientResponseError:
    request = RequestInfo(
        URL("https://stg.numinous.earth/api/v3/dkg/scores"),
        "POST",
        CIMultiDictProxy(CIMultiDict()),
    )
    return ClientResponseError(request, (), status=status)


async def unposted_runs(db_client: DatabaseClient) -> list[str]:
    rows = await db_client.many(
        "SELECT DISTINCT scored_at FROM dkg_scores WHERE posted_at IS NULL ORDER BY scored_at"
    )
    return [row[0] for row in rows]


class TestPostDkgScores:
    @pytest.fixture
    def dkg_operations(self, db_client: DatabaseClient) -> DkgDatabaseOperations:
        return DkgDatabaseOperations(db_client=db_client)

    @pytest.fixture
    def dkg_client(self) -> AsyncMock:
        dkg_client = AsyncMock(spec=DkgClient)
        dkg_client.post_scores.side_effect = lambda scoring_run: StoredScores(
            stored=len(scoring_run.scores)
        )
        return dkg_client

    @pytest.fixture
    def logger(self) -> MagicMock:
        return MagicMock(spec=NuminousLogger)

    @pytest.fixture
    def task(
        self, dkg_operations: DkgDatabaseOperations, dkg_client: AsyncMock, logger: MagicMock
    ) -> PostDkgScores:
        return PostDkgScores(
            interval_seconds=300.0,
            dkg_operations=dkg_operations,
            dkg_client=dkg_client,
            logger=logger,
        )

    @pytest.fixture
    async def two_runs(self, dkg_operations: DkgDatabaseOperations) -> None:
        await dkg_operations.save_scores(
            [
                make_score(MINER, SECOND_RUN, 2.0, 3014),
                make_score(OTHER_MINER, SECOND_RUN, 0.0, 3014),
            ]
        )
        await dkg_operations.save_scores(
            [
                make_score(OTHER_MINER, FIRST_RUN, 0.0, 3013),
                make_score(MINER, FIRST_RUN, 1.41, 3013),
            ]
        )

    @pytest.mark.parametrize(
        "changed, error",
        [
            ({"interval_seconds": 300}, ValueError),
            ({"interval_seconds": 0.0}, ValueError),
            ({"dkg_operations": "not operations"}, TypeError),
            ({"dkg_client": "not a client"}, TypeError),
            ({"logger": "not a logger"}, TypeError),
        ],
    )
    def test_init_rejects_bad_arguments(
        self,
        dkg_operations: DkgDatabaseOperations,
        dkg_client: AsyncMock,
        logger: MagicMock,
        changed: dict,
        error: type[Exception],
    ):
        arguments = {
            "interval_seconds": 300.0,
            "dkg_operations": dkg_operations,
            "dkg_client": dkg_client,
            "logger": logger,
        }

        with pytest.raises(error):
            PostDkgScores(**(arguments | changed))

    def test_name_and_interval(self, task: PostDkgScores):
        assert task.name == "post-dkg-scores"
        assert task.interval_seconds == 300.0

    async def test_nothing_scored_posts_nothing(
        self, task: PostDkgScores, dkg_client: AsyncMock, logger: MagicMock
    ):
        await task.run()

        dkg_client.post_scores.assert_not_awaited()
        logger.info.assert_not_called()

    @pytest.mark.usefixtures("two_runs")
    async def test_posts_every_run_oldest_first_with_every_miner_and_its_version(
        self, task: PostDkgScores, dkg_client: AsyncMock
    ):
        await task.run()

        assert [call.args[0] for call in dkg_client.post_scores.await_args_list] == [
            ScoringRun(
                scored_at=FIRST_RUN,
                spec_version=3013,
                scores=[
                    MinerScore(miner_uid=1, miner_hotkey=MINER, score=1.41),
                    MinerScore(miner_uid=2, miner_hotkey=OTHER_MINER, score=0.0),
                ],
            ),
            ScoringRun(
                scored_at=SECOND_RUN,
                spec_version=3014,
                scores=[
                    MinerScore(miner_uid=1, miner_hotkey=MINER, score=2.0),
                    MinerScore(miner_uid=2, miner_hotkey=OTHER_MINER, score=0.0),
                ],
            ),
        ]

    @pytest.mark.usefixtures("two_runs")
    async def test_a_posted_run_is_marked_and_logged(
        self, task: PostDkgScores, db_client: DatabaseClient, logger: MagicMock
    ):
        await task.run()

        assert await unposted_runs(db_client) == []
        assert [call.args for call in logger.info.call_args_list] == [
            ("DKG scores posted",),
            ("DKG scores posted",),
        ]
        assert logger.info.call_args_list[0].kwargs == {
            "extra": {
                "scored_at": FIRST_RUN.isoformat(),
                "spec_version": 3013,
                "miners": 2,
                "stored": 2,
            }
        }

    @pytest.mark.usefixtures("two_runs")
    async def test_a_posted_run_is_not_posted_again(
        self, task: PostDkgScores, dkg_client: AsyncMock
    ):
        await task.run()
        dkg_client.post_scores.reset_mock()

        await task.run()

        dkg_client.post_scores.assert_not_awaited()

    @pytest.mark.usefixtures("two_runs")
    async def test_only_a_run_scored_after_the_last_post_is_posted(
        self, task: PostDkgScores, dkg_client: AsyncMock, dkg_operations: DkgDatabaseOperations
    ):
        await task.run()
        dkg_client.post_scores.reset_mock()
        third_run = SECOND_RUN + timedelta(days=1)
        await dkg_operations.save_scores([make_score(MINER, third_run, 3.0, 3014)])

        await task.run()

        [call] = dkg_client.post_scores.await_args_list
        assert call.args[0].scored_at == third_run

    @pytest.mark.parametrize("status", [401, 422, 500])
    @pytest.mark.usefixtures("two_runs")
    async def test_a_failed_post_leaves_the_run_for_the_next_check(
        self,
        task: PostDkgScores,
        dkg_client: AsyncMock,
        db_client: DatabaseClient,
        logger: MagicMock,
        status: int,
    ):
        dkg_client.post_scores.side_effect = rejected(status)

        with pytest.raises(ClientResponseError):
            await task.run()

        assert await unposted_runs(db_client) == ["2026-10-19T15:00:00Z", "2026-10-20T15:00:00Z"]
        logger.info.assert_not_called()

    @pytest.mark.usefixtures("two_runs")
    async def test_a_failure_after_the_first_run_keeps_the_first_posted(
        self, task: PostDkgScores, dkg_client: AsyncMock, db_client: DatabaseClient
    ):
        dkg_client.post_scores.side_effect = [StoredScores(stored=2), rejected(500)]

        with pytest.raises(ClientResponseError):
            await task.run()

        assert await unposted_runs(db_client) == ["2026-10-20T15:00:00Z"]

        dkg_client.post_scores.side_effect = [StoredScores(stored=2)]
        await task.run()

        assert dkg_client.post_scores.await_args_list[-1].args[0].scored_at == SECOND_RUN
        assert await unposted_runs(db_client) == []

    @pytest.mark.usefixtures("two_runs")
    async def test_a_run_the_backend_already_stored_is_marked_posted(
        self,
        task: PostDkgScores,
        dkg_client: AsyncMock,
        db_client: DatabaseClient,
        logger: MagicMock,
    ):
        dkg_client.post_scores.side_effect = None
        dkg_client.post_scores.return_value = StoredScores(stored=0)

        await task.run()

        assert await unposted_runs(db_client) == []
        assert [call.kwargs["extra"]["stored"] for call in logger.info.call_args_list] == [0, 0]
