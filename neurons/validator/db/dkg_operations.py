from datetime import datetime, timedelta
from typing import Iterable

from pydantic import BaseModel

from neurons.validator.db.client import DatabaseClient
from neurons.validator.models.dkg import (
    DkgForecastHashModel,
    DkgForecastModel,
    DkgForecastRightsModel,
    DkgMinerModel,
    DkgQuestionModel,
    DkgRoundModel,
    DkgScoreModel,
    DkgStatementResultModel,
)

DKG_HISTORY = timedelta(days=90)

_EXPIRED_QUESTIONS = """
    SELECT question_id FROM dkg_questions
    JOIN dkg_rounds USING (round_id)
    WHERE julianday(dkg_rounds.deadline_at) <= julianday(:expired_until)
"""

_DELETE_EXPIRED = [
    f"""
        DELETE FROM dkg_statement_results WHERE ROWID IN (
            SELECT ROWID FROM dkg_statement_results
            WHERE question_id IN ({_EXPIRED_QUESTIONS})
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
    f"""
        DELETE FROM dkg_forecasts WHERE ROWID IN (
            SELECT forecast.ROWID FROM dkg_forecasts AS forecast
            WHERE
                forecast.question_id IN ({_EXPIRED_QUESTIONS})
                AND NOT EXISTS (
                    SELECT 1 FROM dkg_statement_results AS result
                    WHERE
                        result.miner_uid = forecast.miner_uid
                        AND result.miner_hotkey = forecast.miner_hotkey
                        AND result.question_id = forecast.question_id
                )
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
    f"""
        DELETE FROM dkg_forecast_hashes WHERE ROWID IN (
            SELECT hash.ROWID FROM dkg_forecast_hashes AS hash
            WHERE
                hash.question_id IN ({_EXPIRED_QUESTIONS})
                AND NOT EXISTS (
                    SELECT 1 FROM dkg_forecasts AS forecast
                    WHERE
                        forecast.miner_uid = hash.miner_uid
                        AND forecast.miner_hotkey = hash.miner_hotkey
                        AND forecast.question_id = hash.question_id
                )
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
    f"""
        DELETE FROM dkg_questions WHERE ROWID IN (
            SELECT question.ROWID FROM dkg_questions AS question
            WHERE
                question.question_id IN ({_EXPIRED_QUESTIONS})
                AND NOT EXISTS (
                    SELECT 1 FROM dkg_forecast_hashes AS hash
                    WHERE hash.question_id = question.question_id
                )
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
    """
        DELETE FROM dkg_rounds WHERE ROWID IN (
            SELECT closed_round.ROWID FROM dkg_rounds AS closed_round
            WHERE
                julianday(closed_round.deadline_at) <= julianday(:expired_until)
                AND NOT EXISTS (
                    SELECT 1 FROM dkg_questions AS question
                    WHERE question.round_id = closed_round.round_id
                )
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
]

_DELETE_EXPIRED_RECORDS = [
    """
        DELETE FROM dkg_scores WHERE ROWID IN (
            SELECT ROWID FROM dkg_scores
            WHERE julianday(scored_at) <= julianday(:expired_until)
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
    """
        DELETE FROM dkg_results_pulls WHERE ROWID IN (
            SELECT ROWID FROM dkg_results_pulls
            WHERE
                julianday(started_at) <= julianday(:expired_until)
                AND started_at != (SELECT MAX(started_at) FROM dkg_results_pulls)
            LIMIT :batch_size
        )
        RETURNING ROWID
    """,
]


def _columns(model: type[BaseModel]) -> list[str]:
    return list(model.model_fields)


def _rows(records: Iterable[BaseModel]) -> list[tuple]:
    return [tuple(record.model_dump(mode="json").values()) for record in records]


def _insert(table: str, model: type[BaseModel]) -> str:
    columns = _columns(model)
    placeholders = ", ".join(["?"] * len(columns))
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"


def _insert_ignoring_repeats(table: str, model: type[BaseModel]) -> str:
    return f"{_insert(table, model)} ON CONFLICT DO NOTHING"


def _insert_overwriting_repeats(table: str, model: type[BaseModel], key: list[str]) -> str:
    updates = ", ".join(
        f"{column} = excluded.{column}" for column in _columns(model) if column not in key
    )
    return (
        f"{_insert(table, model)} ON CONFLICT ({', '.join(key)}) "
        f"DO UPDATE SET {updates}, updated_at = CURRENT_TIMESTAMP"
    )


class DkgDatabaseOperations:
    __db_client: DatabaseClient

    def __init__(self, db_client: DatabaseClient):
        if not isinstance(db_client, DatabaseClient):
            raise TypeError("db_client must be an instance of DatabaseClient.")

        self.__db_client = db_client

    async def save_round(
        self, closed_round: DkgRoundModel, questions: list[DkgQuestionModel]
    ) -> None:
        await self.__db_client.insert_many(
            _insert_ignoring_repeats("dkg_rounds", DkgRoundModel), _rows([closed_round])
        )
        await self.__db_client.insert_many(
            _insert_ignoring_repeats("dkg_questions", DkgQuestionModel), _rows(questions)
        )

    async def save_forecast_hashes(self, hashes: list[DkgForecastHashModel]) -> None:
        await self.__db_client.insert_many(
            _insert_ignoring_repeats("dkg_forecast_hashes", DkgForecastHashModel), _rows(hashes)
        )

    async def mark_round_hashes_pulled(self, round_id: str) -> None:
        await self.__db_client.update(
            """
                UPDATE dkg_rounds
                SET hashes_pulled_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE round_id = ?
            """,
            [round_id],
        )

    async def get_last_complete_round_deadline(self) -> datetime | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(deadline_at) FROM dkg_rounds WHERE hashes_pulled_at IS NOT NULL
            """
        )
        return datetime.fromisoformat(row[0]) if row[0] else None

    async def get_forecast_hashes(self, question_ids: list[str]) -> list[DkgForecastHashModel]:
        if not question_ids:
            return []

        placeholders = ", ".join(["?"] * len(question_ids))
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(_columns(DkgForecastHashModel))}
                FROM dkg_forecast_hashes
                WHERE question_id IN ({placeholders})
            """,
            question_ids,
            use_row_factory=True,
        )
        return [DkgForecastHashModel(**dict(row)) for row in rows]

    async def save_forecasts(
        self, forecasts: list[DkgForecastModel], results: list[DkgStatementResultModel]
    ) -> None:
        await self.__db_client.insert_many(
            _insert_overwriting_repeats(
                "dkg_forecasts", DkgForecastModel, ["miner_uid", "miner_hotkey", "question_id"]
            ),
            _rows(forecasts),
        )
        await self.__db_client.insert_many(
            _insert_overwriting_repeats(
                "dkg_statement_results",
                DkgStatementResultModel,
                ["miner_uid", "miner_hotkey", "question_id", "statement_position"],
            ),
            _rows(results),
        )

    async def record_results_pull(self, started_at: datetime) -> None:
        await self.__db_client.insert(
            """
                INSERT INTO dkg_results_pulls (started_at) VALUES (?)
            """,
            [started_at.isoformat()],
        )

    async def get_last_results_pull(self) -> datetime | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(started_at) FROM dkg_results_pulls
            """
        )
        return datetime.fromisoformat(row[0]) if row[0] else None

    async def get_current_miners(self) -> list[DkgMinerModel]:
        rows = await self.__db_client.many(
            """
                SELECT CAST(miner_uid AS INTEGER) AS miner_uid, miner_hotkey
                FROM miners
                WHERE blocktime = (SELECT MAX(blocktime) FROM miners)
            """,
            use_row_factory=True,
        )
        return [DkgMinerModel(**dict(row)) for row in rows]

    async def get_miners_count(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT COUNT(*) FROM miners
            """
        )

        return row[0]

    async def upsert_miners(self, miners: list[list[any]]) -> None:
        return await self.__db_client.insert_many(
            """
                INSERT INTO miners
                    (
                        miner_uid,
                        miner_hotkey,
                        node_ip,
                        registered_date,
                        last_updated,
                        blocktime,
                        blocklisted
                    )
                VALUES
                    (
                        ?,
                        ?,
                        ?,
                        ?,
                        CURRENT_TIMESTAMP,
                        ?,
                        FALSE
                    )
                ON CONFLICT
                    (miner_hotkey, miner_uid)
                DO UPDATE
                    set node_ip = ?,
                    last_updated = CURRENT_TIMESTAMP,
                    blocktime = ?
            """,
            miners,
        )

    async def vacuum_database(self, pages: int):
        await self.__db_client.script(f"PRAGMA incremental_vacuum({pages})")

    async def get_forecast_rights(
        self, judged_after: datetime, judged_before: datetime
    ) -> list[DkgForecastRightsModel]:
        rows = await self.__db_client.many(
            """
                WITH rights AS (
                    SELECT
                        result.miner_uid,
                        result.miner_hotkey,
                        result.question_id,
                        result.judged_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY result.miner_uid, result.miner_hotkey, result.event_id
                            ORDER BY hash.received_at, result.question_id, result.statement_position
                        ) = 1 AS first_of_its_event
                    FROM dkg_statement_results AS result
                    JOIN dkg_forecast_hashes AS hash
                        USING (miner_uid, miner_hotkey, question_id)
                    WHERE
                        result.outcome = 1
                        AND julianday(result.judged_at) < julianday(:judged_before)
                        AND NOT EXISTS (
                            SELECT 1 FROM dkg_forecasts AS failed
                            WHERE
                                failed.miner_uid = result.miner_uid
                                AND failed.miner_hotkey = result.miner_hotkey
                                AND failed.question_id = result.question_id
                                AND failed.failed_check IS NOT NULL
                        )
                ),
                counted AS (
                    SELECT
                        miner_uid,
                        miner_hotkey,
                        question_id,
                        SUM(
                            first_of_its_event
                            AND julianday(judged_at) > julianday(:judged_after)
                        ) AS rights,
                        SUM(NOT first_of_its_event) AS repeats
                    FROM rights
                    GROUP BY miner_uid, miner_hotkey, question_id
                    HAVING rights > 0
                )
                SELECT
                    counted.miner_uid,
                    counted.miner_hotkey,
                    counted.rights,
                    json_array_length(forecast.opened, '$.statements') - counted.repeats
                        AS statements
                FROM counted
                JOIN dkg_forecasts AS forecast
                    USING (miner_uid, miner_hotkey, question_id)
            """,
            {"judged_after": judged_after.isoformat(), "judged_before": judged_before.isoformat()},
            use_row_factory=True,
        )
        return [DkgForecastRightsModel(**dict(row)) for row in rows]

    async def get_last_scored_at(self) -> datetime | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(scored_at) FROM dkg_scores
            """
        )
        return datetime.fromisoformat(row[0]) if row[0] else None

    async def save_scores(self, scores: list[DkgScoreModel]) -> None:
        await self.__db_client.insert_many(_insert("dkg_scores", DkgScoreModel), _rows(scores))

    async def get_latest_scores(self, scored_until: datetime) -> list[DkgScoreModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(_columns(DkgScoreModel))}
                FROM dkg_scores
                WHERE scored_at = (
                    SELECT MAX(scored_at) FROM dkg_scores
                    WHERE julianday(scored_at) <= julianday(?)
                )
            """,
            [scored_until.isoformat()],
            use_row_factory=True,
        )
        return [DkgScoreModel(**dict(row)) for row in rows]

    async def get_unposted_scores(self) -> list[DkgScoreModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(_columns(DkgScoreModel))}
                FROM dkg_scores
                WHERE posted_at IS NULL
                ORDER BY julianday(scored_at), miner_uid
            """,
            use_row_factory=True,
        )
        return [DkgScoreModel(**dict(row)) for row in rows]

    async def mark_scores_posted(self, scored_at: datetime) -> None:
        await self.__db_client.update(
            """
                UPDATE dkg_scores
                SET posted_at = CURRENT_TIMESTAMP
                WHERE julianday(scored_at) = julianday(?)
            """,
            [scored_at.isoformat()],
        )

    async def delete_expired(self, expired_until: datetime, batch_size: int) -> int:
        parameters = {"expired_until": expired_until.isoformat(), "batch_size": batch_size}
        deleted = 0

        for query in _DELETE_EXPIRED_RECORDS:
            deleted += len(await self.__db_client.delete(query, parameters))

        expired_round = await self.__db_client.one(
            """
                SELECT 1 FROM dkg_rounds
                WHERE julianday(deadline_at) <= julianday(:expired_until)
                LIMIT 1
            """,
            parameters,
        )
        if expired_round is None:
            return deleted

        for query in _DELETE_EXPIRED:
            deleted += len(await self.__db_client.delete(query, parameters))

        return deleted
