"""Local standard-library HTTP service for single-reviewer pilot work."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .store import QUEUE_NAMES, ReviewStore


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
_STATIC_PAGE = Path(__file__).with_name("static") / "index.html"


@dataclass(frozen=True)
class ServiceResponse:
    """Small response value usable both in-process and through HTTP."""

    status: int
    body: bytes
    headers: dict[str, str]

    @property
    def status_code(self) -> int:
        return self.status

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


def _json_response(status: int, payload: Mapping[str, Any]) -> ServiceResponse:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ServiceResponse(
        status=status,
        body=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )


def _error(status: int, message: str) -> ServiceResponse:
    return _json_response(status, {"status": "error", "error": message})


def _exception_message(error: BaseException) -> str:
    return str(error.args[0]) if error.args else str(error)


class ReviewService:
    """Dispatch the review API without requiring a long-lived test server."""

    def __init__(self, store: ReviewStore):
        self.store = store

    def dispatch(
        self,
        method: str,
        target: str,
        body: bytes | bytearray | str | Mapping[str, Any] | None = None,
    ) -> ServiceResponse:
        parsed = urlsplit(target)
        try:
            if method == "GET" and parsed.path == "/":
                return ServiceResponse(
                    status=200,
                    body=_STATIC_PAGE.read_bytes(),
                    headers={"Content-Type": "text/html; charset=utf-8"},
                )
            if method == "GET" and parsed.path == "/api/next":
                query = parse_qs(parsed.query)
                queue_name = query.get("queue", [None])[0]
                if not queue_name:
                    return _error(400, "missing queue query parameter")
                item = self.store.next_item(queue_name)
                return _json_response(200, {"status": "ok", "item": item})
            if method == "GET" and parsed.path == "/api/progress":
                return _json_response(200, self._progress())
            if method == "POST" and parsed.path == "/api/submit":
                payload = self._parse_json_object(body)
                self._validate_decision(payload)
                saved = self.store.submit(payload)
                return _json_response(200, {"status": "saved", "event": saved})
            if method == "POST" and parsed.path == "/api/undo":
                payload = self._parse_json_object(body)
                if "review_event_id" not in payload or "reviewer_id" not in payload:
                    raise ValueError("undo requires review_event_id and reviewer_id")
                saved = self.store.undo(
                    payload["review_event_id"],
                    reviewer_id=payload["reviewer_id"],
                    submitted_at=payload.get("submitted_at"),
                )
                return _json_response(200, {"status": "saved", "event": saved})
            if method == "POST" and parsed.path == "/api/flag":
                payload = self._parse_json_object(body)
                self._validate_decision(payload)
                return self._flag(payload)
            if parsed.path in {
                "/api/next",
                "/api/progress",
                "/api/submit",
                "/api/undo",
                "/api/flag",
            }:
                return _error(405, f"method {method} is not allowed for {parsed.path}")
            return _error(404, f"unknown endpoint: {parsed.path}")
        except json.JSONDecodeError as error:
            return _error(400, f"malformed JSON: {error.msg}")
        except ValueError as error:
            return _error(400, _exception_message(error))
        except KeyError as error:
            return _error(404, _exception_message(error))
        except sqlite3.IntegrityError as error:
            return _error(409, f"review conflict: {_exception_message(error)}")

    @staticmethod
    def _parse_json_object(
        body: bytes | bytearray | str | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(body, Mapping):
            return dict(body)
        if body is None:
            raise ValueError("request body must be a JSON object")
        if isinstance(body, (bytes, bytearray)):
            body = bytes(body).decode("utf-8")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    @staticmethod
    def _validate_decision(payload: Mapping[str, Any]) -> None:
        decision = payload.get("decision")
        if not isinstance(decision, Mapping) or not decision:
            raise ValueError("decision must be a non-empty JSON object")
        if not any(key in decision for key in ("label", "action")):
            raise ValueError("decision requires a label or action")

    def _flag(self, payload: Mapping[str, Any]) -> ServiceResponse:
        queue_name = payload.get("queue_name")
        if queue_name not in QUEUE_NAMES:
            raise ValueError(f"unsupported review queue: {queue_name!r}")
        record_id = payload.get("record_id")
        row = self.store.connection.execute(
            "SELECT * FROM review_items WHERE queue_name = ? AND record_id = ?",
            (queue_name, record_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown review item: {queue_name}/{record_id}")

        source_item = self.store._item_from_row(row)
        flagged_item = self.store.add_item(
            {
                "queue_name": "flagged_difficult",
                "record_id": source_item["record_id"],
                "stage": source_item["stage"],
                "risk_stratum": source_item["risk_stratum"],
                "payload": source_item["payload"],
            }
        )
        item_events = [
            event
            for event in self.store.events()
            if event["queue_name"] == queue_name and event["record_id"] == record_id
        ]
        superseded = {
            event["supersedes_event_id"]
            for event in item_events
            if event["supersedes_event_id"] is not None
        }
        for event in reversed(item_events):
            if (
                event["review_event_id"] not in superseded
                and event["decision"].get("action") == "flag"
            ):
                return _json_response(
                    200,
                    {"status": "saved", "event": event, "flagged_item": flagged_item},
                )
        saved = self.store.submit(payload)
        return _json_response(
            200,
            {"status": "saved", "event": saved, "flagged_item": flagged_item},
        )

    def _progress(self) -> dict[str, Any]:
        queues = {queue_name: self.store.progress(queue_name) for queue_name in sorted(QUEUE_NAMES)}
        decisions = [
            event
            for event in self.store.events()
            if event["decision"].get("action") != "undo"
        ]
        effective_seconds = float(sum(event["effective_elapsed_seconds"] for event in decisions))
        decision_count = len(decisions)
        decisions_per_hour = (
            decision_count * 3600.0 / effective_seconds if effective_seconds > 0 else 0.0
        )
        grouped: dict[str, list[float]] = defaultdict(list)
        for event in decisions:
            grouped[f"{event['stage']}/{event['risk_stratum']}"] .append(
                float(event["effective_elapsed_seconds"])
            )
        medians = {key: float(median(values)) for key, values in sorted(grouped.items())}
        remaining = sum(queue["remaining"] for queue in queues.values())
        projected = (
            effective_seconds / decision_count * 900.0 / 3600.0 if decision_count else 0.0
        )
        return {
            "status": "ok",
            "queues": queues,
            "queue_counts": queues,
            "effective_review_seconds": effective_seconds,
            "effective_review_time_seconds": effective_seconds,
            "decisions_per_hour": decisions_per_hour,
            "remaining_items": remaining,
            "median_seconds_by_stage_risk": medians,
            "projected_full_900_review_hours": projected,
            "projected_full_900_hours": projected,
        }


def make_http_server(
    store: ReviewStore,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> HTTPServer:
    """Create, but do not start, the local HTTP server."""

    service = ReviewService(store)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._dispatch("POST", self.rfile.read(length))

        def _dispatch(self, method: str, body: bytes | None = None) -> None:
            response = service.dispatch(method, self.path, body)
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)

    return HTTPServer((host, port), Handler)
