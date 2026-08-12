import json
from typing import Iterable, Optional, Type, TypeVar

from pydantic import BaseModel

from neurons.validator.db.client import DatabaseClient
from neurons.validator.models.agent_run_logs import (
    AGENT_RUN_LOGS_FIELDS,
    AgentRunLogExportedStatus,
    AgentRunLogsModel,
)
from neurons.validator.models.agent_runs import (
    AGENT_RUNS_FIELDS,
    AgentRunExportedStatus,
    AgentRunsModel,
    AgentRunStatus,
)
from neurons.validator.models.event import EVENTS_FIELDS, EventsModel, EventStatus
from neurons.validator.models.miner_agent import MINER_AGENTS_FIELDS, MinerAgentsModel
from neurons.validator.models.prediction import (
    PREDICTION_FIELDS,
    PredictionExportedStatus,
    PredictionsModel,
)
from neurons.validator.models.reasoning import ReasoningForExport
from neurons.validator.models.reforecast_memory import MAX_MEMORY_CHARS, ReforecastMemoryForExport
from neurons.validator.models.sources import SourceItem, SourcesForExport
from neurons.validator.utils.logger.logger import NuminousLogger

GenericModel = TypeVar("GenericModel", bound=BaseModel)


class DatabaseOperations:
    __db_client: DatabaseClient
    logger: NuminousLogger

    def __init__(self, db_client: DatabaseClient, logger: NuminousLogger):
        if not isinstance(db_client, DatabaseClient):
            raise ValueError("Invalid db_client arg")

        if not isinstance(logger, NuminousLogger):
            raise TypeError("logger must be an instance of NuminousLogger.")

        self.__db_client = db_client
        self.logger = logger

    def _parse_rows(
        self, model: Type[GenericModel], rows: Iterable[tuple], throw_on_error: bool = False
    ) -> list[GenericModel]:
        parsed_rows = []

        for row in rows:
            try:
                parsed_rows.append(model(**dict(row)))
            except Exception as e:
                if throw_on_error:
                    raise e

                self.logger.exception("Error parsing model", extra={"model": model.__name__})

        return parsed_rows

    async def delete_event(self, event_id: str, deleted_at: str) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    deleted_at = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE event_id = ?
                RETURNING event_id
            """,
            [EventStatus.DELETED, deleted_at, event_id],
        )

    async def delete_events_hard_delete(self, batch_size: int) -> Iterable[tuple[str]]:
        return await self.__db_client.delete(
            """
            WITH events_to_delete AS (
                SELECT
                    ROWID
                FROM
                    events
                WHERE
                    status = ?
                    AND datetime(deleted_at) < datetime(CURRENT_TIMESTAMP, '-14 day')
                ORDER BY
                    ROWID ASC
                LIMIT ?
            )
            DELETE FROM
                events
            WHERE
                ROWID IN (
                    SELECT
                        ROWID
                    FROM
                        events_to_delete
                )
            RETURNING
                ROWID
            """,
            [EventStatus.DELETED, batch_size],
        )

    async def delete_predictions(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH predictions_to_delete AS (
                    SELECT
                        p.ROWID
                    FROM
                        predictions p
                    WHERE
                        p.exported = ?
                        AND datetime(p.submitted) < datetime(CURRENT_TIMESTAMP, '-7 day')
                    ORDER BY
                        p.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    predictions
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            predictions_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [
                PredictionExportedStatus.EXPORTED,
                batch_size,
            ],
        )

    async def get_event(self, unique_event_id: str) -> None | EventsModel:
        result = await self.__db_client.one(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE
                    unique_event_id = ?
            """,
            parameters=[unique_event_id],
            use_row_factory=True,
        )

        if result is None:
            return None

        return EventsModel(**dict(result))

    async def get_events(self, unique_event_ids) -> list[EventsModel]:
        placeholders = ", ".join(["?"] * len(unique_event_ids))

        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(EVENTS_FIELDS)}
                FROM
                    events
                WHERE
                    unique_event_id IN ({placeholders})
            """,
            parameters=unique_event_ids,
            use_row_factory=True,
        )

        events = self._parse_rows(model=EventsModel, rows=rows)

        return events

    async def get_events_last_deleted_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(deleted_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_events_last_resolved_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(resolved_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_events_pending_first_created_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MIN(created_at) FROM events WHERE status = ?
            """,
            [EventStatus.PENDING],
        )

        if row is not None:
            return row[0]

    async def get_events_to_predict(self, interval_start_datetime: str) -> list[EventsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(EVENTS_FIELDS)}
                FROM events
                WHERE
                    status = ?
                    AND datetime(CURRENT_TIMESTAMP) < datetime(cutoff)
                    AND datetime(registered_date) < datetime(?)
                ORDER BY
                    unique_event_id ASC
            """,
            parameters=[EventStatus.PENDING, interval_start_datetime],
            use_row_factory=True,
        )
        return self._parse_rows(model=EventsModel, rows=rows)

    async def get_last_event_from(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(created_at) FROM events
            """
        )

        if row is not None:
            return row[0]

    async def get_miners_count(self) -> int:
        row = await self.__db_client.one(
            """
                SELECT COUNT(*) FROM miners
            """
        )

        return row[0]

    async def get_predictions_to_export(self, batch_size: int):
        return await self.__db_client.many(
            """
                SELECT
                    p.ROWID,
                    p.unique_event_id,
                    p.miner_uid,
                    p.miner_hotkey,
                    p.track,
                    e.event_type,
                    p.latest_prediction,
                    p.interval_start_minutes,
                    p.interval_agg_prediction,
                    p.interval_count,
                    p.submitted,
                    p.run_id,
                    p.version_id
                FROM
                    predictions p
                JOIN
                    events e ON e.unique_event_id = p.unique_event_id
                WHERE
                    p.exported = ?
                ORDER BY
                    p.ROWID ASC
                LIMIT
                    ?
            """,
            [PredictionExportedStatus.NOT_EXPORTED, batch_size],
        )

    async def mark_predictions_as_exported(self, ids: list[str]):
        placeholders = ", ".join(["?"] * len(ids))

        return await self.__db_client.update(
            f"""
                UPDATE
                    predictions
                SET
                    exported = ?
                WHERE
                    ROWID IN ({placeholders})
                RETURNING
                    ROWID
            """,
            [PredictionExportedStatus.EXPORTED] + ids,
        )

    async def resolve_event(
        self, event_id: str, outcome: str, resolved_at: str, forecasts: str
    ) -> Iterable[tuple[str]]:
        return await self.__db_client.update(
            """
                UPDATE
                    events
                SET
                    status = ?,
                    outcome = ?,
                    resolved_at = ?,
                    forecasts = ?,
                    local_updated_at = CURRENT_TIMESTAMP
                WHERE
                    event_id = ?
                    AND status = ?
                RETURNING
                    event_id
            """,
            [EventStatus.SETTLED, outcome, resolved_at, forecasts, event_id, EventStatus.PENDING],
        )

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

    async def upsert_predictions(self, predictions: list[PredictionsModel]) -> None:
        if not predictions:
            return

        fields_to_insert = [
            "unique_event_id",
            "miner_hotkey",
            "miner_uid",
            "track",
            "latest_prediction",
            "interval_start_minutes",
            "interval_agg_prediction",
            "run_id",
            "version_id",
        ]
        prediction_tuples = [
            tuple(getattr(pred, field) for field in fields_to_insert) for pred in predictions
        ]

        await self.__db_client.insert_many(
            """
                INSERT INTO predictions (
                    unique_event_id,
                    miner_hotkey,
                    miner_uid,
                    track,
                    latest_prediction,
                    interval_start_minutes,
                    interval_agg_prediction,
                    interval_count,
                    run_id,
                    version_id
                )
                VALUES (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    1,
                    ?,
                    ?
                )
                ON CONFLICT(unique_event_id, miner_uid, miner_hotkey, track, interval_start_minutes)
                DO UPDATE SET
                    latest_prediction = excluded.latest_prediction,
                    interval_agg_prediction = (interval_agg_prediction * interval_count + excluded.interval_agg_prediction) / (interval_count + 1),
                    interval_count = interval_count + 1,
                    updated_at = CURRENT_TIMESTAMP,
                    run_id = excluded.run_id,
                    version_id = excluded.version_id
            """,
            prediction_tuples,
        )

    async def upsert_events(self, events: list[EventsModel]) -> None:
        """Upsert a list of EventsModel objects into the database"""

        fields_to_insert = [
            field_name
            for field_name in EVENTS_FIELDS
            if field_name not in ("registered_date", "local_updated_at")
        ]
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        # Convert each event into a tuple of values in the same order as fields_to_insert
        event_tuples = [
            tuple(event.model_dump()[field_name] for field_name in fields_to_insert)
            for event in events
        ]

        sql = f"""
                INSERT INTO events
                    ({columns}, registered_date, local_updated_at)
                VALUES
                    ({placeholders}, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT
                    (unique_event_id)
                DO NOTHING
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=event_tuples,
        )

    async def insert_reasoning(self, run_id: str, reasoning: str) -> None:
        await self.__db_client.insert_many(
            """
                INSERT INTO reasoning (run_id, reasoning, exported)
                VALUES (?, ?, ?)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    reasoning = excluded.reasoning,
                    updated_at = CURRENT_TIMESTAMP
            """,
            [(run_id, reasoning, False)],
        )

    async def get_reasonings_for_export(self, limit: int = 200) -> list[ReasoningForExport]:
        rows = await self.__db_client.many(
            """
                SELECT
                    r.run_id,
                    r.reasoning,
                    r.created_at,
                    ar.unique_event_id AS event_id,
                    ar.miner_uid,
                    ar.miner_hotkey,
                    ar.track
                FROM reasoning r
                JOIN agent_runs ar ON r.run_id = ar.run_id
                WHERE r.exported = 0
                ORDER BY r.created_at ASC
                LIMIT ?
            """,
            parameters=[limit],
            use_row_factory=True,
        )

        return self._parse_rows(model=ReasoningForExport, rows=rows)

    async def mark_reasonings_as_exported(self, run_ids: list[str]) -> None:
        if not run_ids:
            return

        placeholders = ", ".join(["?" for _ in run_ids])

        await self.__db_client.update(
            f"""
                UPDATE reasoning
                SET
                    exported = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )

    async def delete_reasonings(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH reasonings_to_delete AS (
                    SELECT
                        r.ROWID
                    FROM
                        reasoning r
                    WHERE
                        r.exported = 1
                        AND datetime(r.created_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                    ORDER BY
                        r.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    reasoning
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            reasonings_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [batch_size],
        )

    async def insert_sources(self, run_id: str, sources: list[SourceItem]) -> None:
        sources_json = json.dumps([source.model_dump(mode="json") for source in sources])

        await self.__db_client.insert_many(
            """
                INSERT INTO sources (run_id, sources, exported)
                VALUES (?, ?, ?)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    sources = excluded.sources,
                    updated_at = CURRENT_TIMESTAMP
            """,
            [(run_id, sources_json, False)],
        )

    async def get_sources_for_export(self, limit: int = 200) -> list[SourcesForExport]:
        rows = await self.__db_client.many(
            """
                SELECT
                    s.run_id,
                    s.sources,
                    s.created_at,
                    ar.unique_event_id AS event_id,
                    ar.miner_uid,
                    ar.miner_hotkey,
                    ar.track
                FROM sources s
                JOIN agent_runs ar ON s.run_id = ar.run_id
                WHERE s.exported = 0
                ORDER BY s.created_at ASC
                LIMIT ?
            """,
            parameters=[limit],
            use_row_factory=True,
        )

        return self._parse_rows(model=SourcesForExport, rows=rows)

    async def mark_sources_as_exported(self, run_ids: list[str]) -> None:
        if not run_ids:
            return

        placeholders = ", ".join(["?" for _ in run_ids])

        await self.__db_client.update(
            f"""
                UPDATE sources
                SET
                    exported = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )

    async def delete_sources(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH sources_to_delete AS (
                    SELECT
                        s.ROWID
                    FROM
                        sources s
                    WHERE
                        s.exported = 1
                        AND datetime(s.created_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                    ORDER BY
                        s.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    sources
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            sources_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [batch_size],
        )

    async def insert_reforecast_memory(
        self, run_id: str, memory: str, interval_start_minutes: int
    ) -> None:
        await self.__db_client.insert_many(
            """
                INSERT INTO reforecast_memory
                    (run_id, memory, interval_start_minutes, exported)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    memory = excluded.memory,
                    interval_start_minutes = excluded.interval_start_minutes,
                    updated_at = CURRENT_TIMESTAMP
            """,
            [(run_id, memory[:MAX_MEMORY_CHARS], interval_start_minutes, False)],
        )

    async def get_reforecast_memories_for_export(
        self, limit: int = 200
    ) -> list[ReforecastMemoryForExport]:
        rows = await self.__db_client.many(
            """
                SELECT
                    m.run_id,
                    m.memory,
                    m.interval_start_minutes,
                    m.created_at,
                    ar.unique_event_id AS event_id,
                    ar.miner_uid,
                    ar.miner_hotkey
                FROM reforecast_memory m
                JOIN agent_runs ar ON m.run_id = ar.run_id
                WHERE m.exported = 0
                ORDER BY m.created_at ASC
                LIMIT ?
            """,
            parameters=[limit],
            use_row_factory=True,
        )

        return self._parse_rows(model=ReforecastMemoryForExport, rows=rows)

    async def mark_reforecast_memories_as_exported(self, run_ids: list[str]) -> None:
        if not run_ids:
            return

        placeholders = ", ".join(["?" for _ in run_ids])

        await self.__db_client.update(
            f"""
                UPDATE reforecast_memory
                SET
                    exported = 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id IN ({placeholders})
            """,
            run_ids,
        )

    async def delete_reforecast_memories(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH memories_to_delete AS (
                    SELECT
                        m.ROWID
                    FROM
                        reforecast_memory m
                    WHERE
                        m.exported = 1
                        AND datetime(m.created_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                    ORDER BY
                        m.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    reforecast_memory
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            memories_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [batch_size],
        )

    async def prediction_exists(
        self,
        unique_event_id: str,
        miner_uid: int,
        miner_hotkey: str,
        interval_start_minutes: int,
    ) -> bool:
        result = await self.__db_client.one(
            """
            SELECT 1 FROM predictions
            WHERE unique_event_id = ?
              AND miner_uid = ?
              AND miner_hotkey = ?
              AND interval_start_minutes = ?
            LIMIT 1
            """,
            parameters=[unique_event_id, miner_uid, miner_hotkey, interval_start_minutes],
        )
        return result is not None

    async def get_latest_prediction_for_event_and_miner(
        self,
        unique_event_id: str,
        miner_uid: int,
        miner_hotkey: str,
        track: str,
    ) -> Optional[PredictionsModel]:
        """
        Get the latest prediction for the given event, miner, and track, regardless of interval.
        Returns the most recent prediction, or None if no prediction exists.
        """
        rows = await self.__db_client.many(
            f"""
            SELECT {', '.join(PREDICTION_FIELDS)}
            FROM
                predictions
            WHERE
                unique_event_id = ?
                AND miner_uid = ?
                AND miner_hotkey = ?
                AND track = ?
            ORDER BY
                interval_start_minutes DESC,
                updated_at DESC
            LIMIT 1
            """,
            parameters=[unique_event_id, miner_uid, miner_hotkey, track],
            use_row_factory=True,
        )

        if not rows:
            return None

        predictions = self._parse_rows(model=PredictionsModel, rows=rows)
        return predictions[0] if predictions else None

    async def get_predictions_for_event(
        self, unique_event_id: str, interval_start_minutes: int
    ) -> list[PredictionsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT
                    {', '.join(PREDICTION_FIELDS)}
                FROM
                    predictions
                WHERE
                    unique_event_id = ?
                    AND interval_start_minutes = ?
                ORDER BY
                    miner_uid ASC,
                    miner_hotkey ASC
            """,
            parameters=[unique_event_id, interval_start_minutes],
            use_row_factory=True,
        )

        predictions = self._parse_rows(model=PredictionsModel, rows=rows)

        return predictions

    async def vacuum_database(self, pages: int):
        await self.__db_client.script(f"PRAGMA incremental_vacuum({pages})")

    async def get_last_agent_pulled_at(self) -> str | None:
        row = await self.__db_client.one(
            """
                SELECT MAX(pulled_at) FROM miner_agents
            """
        )

        if row is not None:
            return row[0]

    async def upsert_miner_agents(self, agents: list[MinerAgentsModel]) -> None:
        if not agents:
            return

        fields_to_insert = list(MINER_AGENTS_FIELDS)
        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        agent_tuples = [
            tuple(getattr(agent, field_name) for field_name in fields_to_insert) for agent in agents
        ]

        sql = f"""
                INSERT INTO miner_agents
                    ({columns})
                VALUES
                    ({placeholders})
                ON CONFLICT
                    (miner_uid, miner_hotkey, track, version_number)
                DO UPDATE SET
                    file_path = excluded.file_path,
                    pulled_at = CURRENT_TIMESTAMP
        """
        return await self.__db_client.insert_many(
            sql=sql,
            parameters=agent_tuples,
        )

    async def get_agent_by_version(self, version_id: str) -> None | MinerAgentsModel:
        row = await self.__db_client.one(
            f"""
                SELECT
                    {', '.join(MINER_AGENTS_FIELDS)}
                FROM
                    miner_agents
                WHERE
                    version_id = ?
            """,
            parameters=[version_id],
            use_row_factory=True,
        )

        if row is None:
            return None

        parsed = self._parse_rows(model=MinerAgentsModel, rows=[row])
        return parsed[0] if parsed else None

    async def get_active_agents(self, limit: int | None = None) -> list[MinerAgentsModel]:
        sql = f"""
                WITH ranked_agents AS (
                    SELECT
                        {', '.join(MINER_AGENTS_FIELDS)},
                        ROW_NUMBER() OVER (
                            PARTITION BY miner_uid, miner_hotkey, track
                            ORDER BY version_number DESC
                        ) as rn
                    FROM
                        miner_agents
                )
                SELECT
                    {', '.join(MINER_AGENTS_FIELDS)}
                FROM
                    ranked_agents
                WHERE
                    rn = 1
                ORDER BY
                    version_id ASC
        """

        parameters = []
        if limit:
            sql += " LIMIT ?"
            parameters.append(limit)

        rows = await self.__db_client.many(
            sql=sql,
            parameters=parameters if parameters else None,
            use_row_factory=True,
        )

        return self._parse_rows(model=MinerAgentsModel, rows=rows)

    async def upsert_agent_runs(self, runs: list[AgentRunsModel]) -> None:
        if not runs:
            return

        fields_to_insert = [
            "run_id",
            "unique_event_id",
            "agent_version_id",
            "miner_uid",
            "miner_hotkey",
            "track",
            "status",
            "interval_start_minutes",
            "exported",
            "is_final",
        ]

        run_tuples = [
            (
                run.run_id,
                run.unique_event_id,
                run.agent_version_id,
                run.miner_uid,
                run.miner_hotkey,
                run.track,
                run.status.value,
                run.interval_start_minutes,
                1 if run.exported else 0,
                1 if run.is_final else 0,
            )
            for run in runs
        ]

        placeholders = ", ".join(["?"] * len(fields_to_insert))
        columns = ", ".join(fields_to_insert)

        await self.__db_client.insert_many(
            f"""
                INSERT INTO agent_runs ({columns})
                VALUES ({placeholders})
                ON CONFLICT (run_id)
                DO UPDATE SET
                    status = excluded.status,
                    exported = excluded.exported,
                    is_final = excluded.is_final,
                    updated_at = CURRENT_TIMESTAMP
            """,
            run_tuples,
        )

    async def get_unexported_agent_runs(self, limit: int = 1000) -> list[AgentRunsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(AGENT_RUNS_FIELDS)}
                FROM agent_runs
                WHERE exported = ?
                ORDER BY created_at ASC
                LIMIT ?
            """,
            [AgentRunExportedStatus.NOT_EXPORTED, limit],
            use_row_factory=True,
        )

        return self._parse_rows(model=AgentRunsModel, rows=rows)

    async def mark_agent_runs_as_exported(self, run_ids: list[str]) -> None:
        if not run_ids:
            return

        placeholders = ", ".join(["?" for _ in run_ids])

        await self.__db_client.update(
            f"""
                UPDATE agent_runs
                SET
                    exported = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id IN ({placeholders})
            """,
            [AgentRunExportedStatus.EXPORTED] + run_ids,
        )

    async def insert_agent_run_log(self, run_id: str, log_content: str) -> None:
        truncated_log = log_content[:30000] if len(log_content) > 30000 else log_content

        await self.__db_client.insert_many(
            """
                INSERT INTO agent_run_logs (run_id, log_content, exported)
                VALUES (?, ?, ?)
                ON CONFLICT (run_id)
                DO UPDATE SET
                    log_content = excluded.log_content,
                    exported = excluded.exported,
                    updated_at = CURRENT_TIMESTAMP
            """,
            [(run_id, truncated_log, AgentRunLogExportedStatus.NOT_EXPORTED)],
        )

    async def get_unexported_agent_run_logs(self, limit: int = 100) -> list[AgentRunLogsModel]:
        rows = await self.__db_client.many(
            f"""
                SELECT {', '.join(AGENT_RUN_LOGS_FIELDS)}
                FROM agent_run_logs
                WHERE exported = ?
                ORDER BY created_at ASC
                LIMIT ?
            """,
            [AgentRunLogExportedStatus.NOT_EXPORTED, limit],
            use_row_factory=True,
        )

        return self._parse_rows(model=AgentRunLogsModel, rows=rows)

    async def mark_agent_run_logs_as_exported(self, run_ids: list[str]) -> None:
        if not run_ids:
            return

        placeholders = ", ".join(["?" for _ in run_ids])

        await self.__db_client.update(
            f"""
                UPDATE agent_run_logs
                SET
                    exported = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id IN ({placeholders})
            """,
            [AgentRunLogExportedStatus.EXPORTED] + run_ids,
        )

    async def delete_agent_run_logs(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH logs_to_delete AS (
                    SELECT
                        ROWID
                    FROM
                        agent_run_logs
                    WHERE
                        exported = ?
                        AND datetime(created_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                    ORDER BY
                        ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    agent_run_logs
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            logs_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [AgentRunLogExportedStatus.EXPORTED, batch_size],
        )

    async def delete_agent_runs(self, batch_size: int) -> Iterable[tuple[int]]:
        return await self.__db_client.delete(
            """
                WITH runs_to_delete AS (
                    SELECT
                        ar.ROWID
                    FROM
                        agent_runs ar
                    LEFT JOIN
                        agent_run_logs arl ON ar.run_id = arl.run_id
                    LEFT JOIN
                        reasoning re ON ar.run_id = re.run_id
                    LEFT JOIN
                        sources so ON ar.run_id = so.run_id
                    WHERE
                        ar.exported = ?
                        AND datetime(ar.created_at) < datetime(CURRENT_TIMESTAMP, '-7 day')
                        AND arl.run_id IS NULL
                        AND re.run_id IS NULL
                        AND so.run_id IS NULL
                    ORDER BY
                        ar.ROWID ASC
                    LIMIT ?
                )
                DELETE FROM
                    agent_runs
                WHERE
                    ROWID IN (
                        SELECT
                            ROWID
                        FROM
                            runs_to_delete
                    )
                RETURNING
                    ROWID
            """,
            [AgentRunExportedStatus.EXPORTED, batch_size],
        )

    async def count_runs_for_event_and_agent(
        self,
        unique_event_id: str,
        agent_version_id: str,
        interval_start_minutes: int,
        status: Optional[AgentRunStatus] = None,
        is_final: Optional[bool] = None,
    ) -> int:
        conditions = [
            "unique_event_id = ?",
            "agent_version_id = ?",
            "interval_start_minutes = ?",
        ]
        params: list = [unique_event_id, agent_version_id, interval_start_minutes]

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        if is_final is not None:
            conditions.append("is_final = ?")
            params.append(1 if is_final else 0)

        sql = f"""
            SELECT COUNT(*)
            FROM agent_runs
            WHERE {' AND '.join(conditions)}
        """

        result = await self.__db_client.one(sql, params)
        return result[0] if result else 0

    async def has_final_run(
        self,
        unique_event_id: str,
        agent_version_id: str,
        interval_start_minutes: int,
    ) -> bool:
        sql = """
            SELECT 1
            FROM agent_runs
            WHERE unique_event_id = ?
                AND agent_version_id = ?
                AND interval_start_minutes = ?
                AND is_final = 1
            LIMIT 1
        """

        result = await self.__db_client.one(
            sql, [unique_event_id, agent_version_id, interval_start_minutes]
        )
        return result is not None
