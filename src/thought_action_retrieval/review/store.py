"""SQLite persistence for the local pilot review application."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


QUEUE_NAMES = {
    "intent_first_pass",
    "pair_first_pass",
    "generated_first_pass",
    "blind_repeat",
    "flagged_difficult",
    "self_adjudication",
}

_BLIND_FIELDS = {
    "first_label",
    "first_reason",
    "automatic_suggestion",
    "preliminary_label",
    "preliminary_reason",
    "suggestion",
}


def effective_elapsed_seconds(*, started: float, submitted: float, idle_seconds: float = 0) -> float:
    """Return review time after idle time, clamped at zero."""
    return max(0, submitted - started - idle_seconds)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _without_blind_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_blind_fields(child)
            for key, child in value.items()
            if key not in _BLIND_FIELDS
        }
    if isinstance(value, list):
        return [_without_blind_fields(child) for child in value]
    return value


class ReviewStore:
    """Persist isolated review queues and immutable review event history."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_items (
                review_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                risk_stratum TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE (queue_name, record_id)
            );

            CREATE TABLE IF NOT EXISTS review_events (
                review_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewer_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                queue_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                risk_stratum TEXT NOT NULL,
                started_at REAL NOT NULL,
                submitted_at REAL NOT NULL,
                idle_seconds REAL NOT NULL,
                effective_elapsed_seconds REAL NOT NULL,
                decision_json TEXT NOT NULL,
                used_suggestion INTEGER NOT NULL CHECK (used_suggestion IN (0, 1)),
                revisit INTEGER NOT NULL CHECK (revisit IN (0, 1)),
                supersedes_event_id INTEGER REFERENCES review_events(review_event_id)
            );

            CREATE INDEX IF NOT EXISTS review_events_queue_record
                ON review_events (queue_name, record_id, review_event_id);
            CREATE UNIQUE INDEX IF NOT EXISTS review_events_one_superseder
                ON review_events (supersedes_event_id)
                WHERE supersedes_event_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS review_sessions (
                review_session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reviewer_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS review_plans (
                review_plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_name TEXT NOT NULL UNIQUE,
                plan_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS exports (
                output_path TEXT PRIMARY KEY,
                last_exported_event_id INTEGER NOT NULL,
                exported_at REAL NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ReviewStore:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    @staticmethod
    def _validate_queue(queue_name: str) -> None:
        if queue_name not in QUEUE_NAMES:
            raise ValueError(f"unsupported review queue: {queue_name!r}")

    def add_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Add an item once, allowing an identical call during resume."""
        required = {"queue_name", "record_id", "stage", "risk_stratum", "payload"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"review item missing fields: {sorted(missing)}")
        queue_name = item["queue_name"]
        self._validate_queue(queue_name)
        payload_json = _json_dumps(item["payload"])
        existing = self.connection.execute(
            """SELECT queue_name, record_id, stage, risk_stratum, payload_json
               FROM review_items WHERE queue_name = ? AND record_id = ?""",
            (queue_name, item["record_id"]),
        ).fetchone()
        normalized = {
            "queue_name": queue_name,
            "record_id": item["record_id"],
            "stage": item["stage"],
            "risk_stratum": item["risk_stratum"],
            "payload_json": payload_json,
        }
        if existing is not None:
            if dict(existing) != normalized:
                raise ValueError(
                    f"review item already exists with different content: {queue_name}/{item['record_id']}"
                )
            return self._item_from_row(
                self.connection.execute(
                    "SELECT * FROM review_items WHERE queue_name = ? AND record_id = ?",
                    (queue_name, item["record_id"]),
                ).fetchone()
            )
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO review_items
                   (queue_name, record_id, stage, risk_stratum, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    queue_name,
                    item["record_id"],
                    item["stage"],
                    item["risk_stratum"],
                    payload_json,
                    time.time(),
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM review_items WHERE review_item_id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._item_from_row(row)

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "review_item_id": row["review_item_id"],
            "queue_name": row["queue_name"],
            "record_id": row["record_id"],
            "stage": row["stage"],
            "risk_stratum": row["risk_stratum"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def _completed_record_ids(self, queue_name: str) -> set[str]:
        rows = self.connection.execute(
            "SELECT * FROM review_events WHERE queue_name = ? ORDER BY review_event_id",
            (queue_name,),
        ).fetchall()
        superseded = {
            row["supersedes_event_id"]
            for row in rows
            if row["supersedes_event_id"] is not None
        }
        return {
            row["record_id"]
            for row in rows
            if row["review_event_id"] not in superseded
            and json.loads(row["decision_json"]).get("action") != "undo"
        }

    def next_item(self, queue_name: str) -> dict[str, Any] | None:
        """Return the next incomplete item from exactly one queue."""
        self._validate_queue(queue_name)
        completed = self._completed_record_ids(queue_name)
        rows = self.connection.execute(
            "SELECT * FROM review_items WHERE queue_name = ? ORDER BY review_item_id",
            (queue_name,),
        ).fetchall()
        for row in rows:
            if row["record_id"] in completed:
                continue
            result = self._item_from_row(row)
            if queue_name == "blind_repeat":
                result = _without_blind_fields(result)
            return result
        return None

    def submit(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Append one complete review event and return its stored representation."""
        required = {
            "reviewer_id",
            "stage",
            "queue_name",
            "record_id",
            "risk_stratum",
            "started_at",
            "submitted_at",
            "decision",
            "used_suggestion",
            "revisit",
        }
        missing = required - set(event)
        if missing:
            raise ValueError(f"review event missing fields: {sorted(missing)}")
        queue_name = event["queue_name"]
        self._validate_queue(queue_name)
        known = self.connection.execute(
            """SELECT stage, risk_stratum FROM review_items
               WHERE queue_name = ? AND record_id = ?""",
            (queue_name, event["record_id"]),
        ).fetchone()
        if known is None:
            raise KeyError(f"unknown review item: {queue_name}/{event['record_id']}")
        if event["stage"] != known["stage"] or event["risk_stratum"] != known["risk_stratum"]:
            raise ValueError("review event stage and risk_stratum must match the review item")
        if not isinstance(event["decision"], Mapping):
            raise ValueError("decision must be a JSON object")
        idle_seconds = event.get("idle_seconds", 0)
        elapsed = effective_elapsed_seconds(
            started=event["started_at"],
            submitted=event["submitted_at"],
            idle_seconds=idle_seconds,
        )
        supersedes_event_id = event.get("supersedes_event_id")
        if supersedes_event_id is not None:
            target = self.connection.execute(
                """SELECT queue_name, record_id, stage, risk_stratum FROM review_events
                   WHERE review_event_id = ?""",
                (supersedes_event_id,),
            ).fetchone()
            if target is None:
                raise KeyError(f"unknown review event: {supersedes_event_id}")
            target_identity = tuple(
                target[field] for field in ("queue_name", "record_id", "stage", "risk_stratum")
            )
            event_identity = tuple(
                event[field] for field in ("queue_name", "record_id", "stage", "risk_stratum")
            )
            if event_identity != target_identity:
                raise ValueError("superseded event must belong to the same review item")
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO review_events
                   (reviewer_id, stage, queue_name, record_id, risk_stratum,
                    started_at, submitted_at, idle_seconds, effective_elapsed_seconds,
                    decision_json, used_suggestion, revisit, supersedes_event_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["reviewer_id"],
                    event["stage"],
                    queue_name,
                    event["record_id"],
                    event["risk_stratum"],
                    event["started_at"],
                    event["submitted_at"],
                    idle_seconds,
                    elapsed,
                    _json_dumps(event["decision"]),
                    int(bool(event["used_suggestion"])),
                    int(bool(event["revisit"])),
                    supersedes_event_id,
                ),
            )
        row = self.connection.execute(
            "SELECT * FROM review_events WHERE review_event_id = ?", (cursor.lastrowid,)
        ).fetchone()
        return self._event_from_row(row)

    def undo(
        self,
        review_event_id: int,
        *,
        reviewer_id: str,
        submitted_at: float | None = None,
    ) -> dict[str, Any]:
        """Append an undo marker that supersedes, but never mutates, an event."""
        row = self.connection.execute(
            "SELECT * FROM review_events WHERE review_event_id = ?", (review_event_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown review event: {review_event_id}")
        timestamp = time.time() if submitted_at is None else submitted_at
        return self.submit(
            {
                "reviewer_id": reviewer_id,
                "stage": row["stage"],
                "queue_name": row["queue_name"],
                "record_id": row["record_id"],
                "risk_stratum": row["risk_stratum"],
                "started_at": timestamp,
                "submitted_at": timestamp,
                "idle_seconds": 0,
                "decision": {"action": "undo"},
                "used_suggestion": False,
                "revisit": True,
                "supersedes_event_id": review_event_id,
            }
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "review_event_id": row["review_event_id"],
            "reviewer_id": row["reviewer_id"],
            "stage": row["stage"],
            "queue_name": row["queue_name"],
            "record_id": row["record_id"],
            "risk_stratum": row["risk_stratum"],
            "started_at": row["started_at"],
            "submitted_at": row["submitted_at"],
            "idle_seconds": row["idle_seconds"],
            "effective_elapsed_seconds": row["effective_elapsed_seconds"],
            "decision": json.loads(row["decision_json"]),
            "used_suggestion": bool(row["used_suggestion"]),
            "revisit": bool(row["revisit"]),
            "supersedes_event_id": row["supersedes_event_id"],
        }

    def events(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM review_events ORDER BY review_event_id"
        ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def event_count(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM review_events").fetchone()[0]

    def progress(self, queue_name: str) -> dict[str, int]:
        self._validate_queue(queue_name)
        total = self.connection.execute(
            "SELECT COUNT(*) FROM review_items WHERE queue_name = ?", (queue_name,)
        ).fetchone()[0]
        completed = len(self._completed_record_ids(queue_name))
        return {"total": total, "completed": completed, "remaining": total - completed}

    def export_jsonl(self, output_path: str | Path) -> int:
        """Atomically export all events when new rows exist; return new-row count."""
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        export_row = self.connection.execute(
            "SELECT last_exported_event_id FROM exports WHERE output_path = ?", (str(output),)
        ).fetchone()
        previous_id = export_row[0] if export_row is not None else 0
        events = self.events()
        latest_id = events[-1]["review_event_id"] if events else 0
        if export_row is not None and previous_id == latest_id and output.exists():
            return 0

        raw = b"".join((_json_dumps(event) + "\n").encode("utf-8") for event in events)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, output)
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        with self.connection:
            self.connection.execute(
                """INSERT INTO exports (output_path, last_exported_event_id, exported_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(output_path) DO UPDATE SET
                       last_exported_event_id = excluded.last_exported_event_id,
                       exported_at = excluded.exported_at""",
                (str(output), latest_id, time.time()),
            )
        return max(0, latest_id - previous_id)
