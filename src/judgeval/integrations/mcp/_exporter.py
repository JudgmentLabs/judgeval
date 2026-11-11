"""Bounded native MCP delivery, independent of OTel sampling and export."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional, Mapping, cast

from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.internal.api.models import (
    IngestMcpCallsRequestCall,
    IngestMcpCallsRequest,
)

from judgeval.logger import judgeval_logger

PayloadField = Literal["input", "output"]
RedactPayload = Callable[[Any, PayloadField], Any]
_MAX_QUEUE = 1_000
_MAX_QUEUE_BYTES = 8 * 1024 * 1024
_BATCH_SIZE = 100
_FLUSH_INTERVAL = 1.0
_REQUEST_TIMEOUT = 5.0
_MAX_ATTEMPTS = 3
_WARNING_INTERVAL = 60.0
OMITTED = object()


McpCall = IngestMcpCallsRequestCall


@dataclass
class _Batch:
    body: bytes
    attempts: int = 0
    retry_at: float = 0


def capture_payload(
    value: Any, field: PayloadField, redact: Optional[RedactPayload]
) -> Any:
    """Redact a detached JSON snapshot, omitting unsafe/oversized payloads."""
    try:
        snapshot = json.loads(json.dumps(value, allow_nan=False))
        redacted = snapshot if redact is None else redact(snapshot, field)
        if redacted is OMITTED:
            return OMITTED
        serialized = json.dumps(redacted, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode("utf-8")) > 128 * 1024:
            return OMITTED
        return json.loads(serialized)
    except Exception:
        return OMITTED


class McpCallExporter:
    def __init__(
        self, api_url: str, api_key: str, organization_id: str, project_id: str
    ) -> None:
        self._project_id = project_id
        self._client = JudgmentSyncClient(api_url, api_key, organization_id)
        self._queue: list[str] = []
        self._queued_bytes = 0
        self._lock = threading.Lock()
        self._sending = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._closed = False
        self._batch: Optional[_Batch] = None
        self._next_warning = 0.0

    @property
    def closed(self) -> bool:
        return self._closed

    def _warn(self, message: str) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_warning:
                return
            self._next_warning = now + _WARNING_INTERVAL
        judgeval_logger.warning(message)

    def enqueue(self, call: Mapping[str, Any]) -> None:
        if self._closed:
            return
        serialized = (
            json.dumps(call, ensure_ascii=False, allow_nan=False)
            .encode("utf-8", errors="replace")
            .decode("utf-8")
        )
        if not _valid_call(json.loads(serialized)):
            self._warn("Invalid MCP event; dropping call")
            return
        size = len(serialized.encode("utf-8"))
        with self._lock:
            if self._closed:
                return
            if (
                len(self._queue) < _MAX_QUEUE
                and self._queued_bytes + size <= _MAX_QUEUE_BYTES
            ):
                self._queue.append(serialized)
                self._queued_bytes += size
                self._schedule()
                return
        self._warn("MCP export queue unavailable; dropping call")

    def _schedule(self) -> None:
        if (
            self._closed
            or self._timer is not None
            or (not self._queue and self._batch is None)
        ):
            return
        delay = max(
            _FLUSH_INTERVAL, self._batch.retry_at - time.time() if self._batch else 0
        )
        self._timer = threading.Timer(
            min(delay, threading.TIMEOUT_MAX), self._flush_background
        )
        self._timer.daemon = True
        self._timer.start()

    def _flush_background(self) -> None:
        with self._lock:
            self._timer = None
        try:
            self.force_flush()
        finally:
            with self._lock:
                self._schedule()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        deadline = time.monotonic() + timeout_millis / 1000
        if not self._sending.acquire(timeout=max(0, deadline - time.monotonic())):
            return False
        try:
            with self._lock:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                remaining = len(self._queue)
            while remaining or self._batch is not None:
                if time.monotonic() >= deadline:
                    return False
                if self._batch is None:
                    with self._lock:
                        count = min(remaining, _BATCH_SIZE)
                        calls = self._queue[:count]
                        del self._queue[:count]
                        self._queued_bytes -= sum(
                            len(call.encode("utf-8")) for call in calls
                        )
                    remaining -= count
                    self._batch = _Batch(
                        ('{"calls":[' + ",".join(calls) + "]}").encode("utf-8")
                    )
                if not self._send(self._batch, deadline):
                    return False
                self._batch = None
            return True
        finally:
            with self._lock:
                self._schedule()
            self._sending.release()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self.force_flush()
        self._client.client.close()

    def _send(self, batch: _Batch, deadline: float) -> bool:
        while batch.attempts < _MAX_ATTEMPTS:
            delay = max(0, batch.retry_at - time.time())
            if time.monotonic() + delay >= deadline:
                return False
            if delay:
                time.sleep(delay)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            batch.attempts += 1
            retry_after = 0.0
            try:
                self._client.post_projects_mcp_calls(
                    self._project_id,
                    cast(IngestMcpCallsRequest, json.loads(batch.body)),
                    content=batch.body,
                    timeout=min(_REQUEST_TIMEOUT, remaining),
                )
                return True
            except JudgmentAPIError as error:
                if error.status_code not in (408, 429) and error.status_code < 500:
                    break
                retry_after = _retry_delay(
                    error.response.headers.get("Retry-After")
                    if error.response is not None
                    else None
                )
            except Exception:
                pass
            batch.retry_at = time.time() + max(
                retry_after, 0.25 * 2 ** (batch.attempts - 1)
            )
        self._warn("MCP export failed; dropping batch")
        return True


_STRING_LIMITS = {
    "call_id": (1, 512),
    "session_id": (1, 512),
    "user_id": (1, 512),
    "server_name": (1, 256),
    "tool_name": (1, 256),
    "client_name": (1, 256),
    "environment_name": (1, 256),
    "model": (1, 256),
    "error_type": (1, 256),
    "intent": (0, 2048),
    "error_message": (0, 8192),
    "trace_id": (0, 512),
    "span_id": (0, 512),
}


def _valid_call(call: Mapping[str, Any]) -> bool:
    """Reject individual malformed events before they poison a delivery batch."""
    required = {
        "call_id",
        "timestamp",
        "duration_ms",
        "outcome",
        "server_name",
        "tool_name",
    }
    other = {
        "timestamp",
        "duration_ms",
        "outcome",
        "argument_keys",
        "input",
        "output",
        "metadata",
    }
    if not required.issubset(call) or call.keys() - _STRING_LIMITS.keys() - other:
        return False
    for key, value in call.items():
        if key in _STRING_LIMITS:
            lower, upper = _STRING_LIMITS[key]
            if (
                not isinstance(value, str)
                or not lower <= len(value.encode("utf-16-le")) // 2 <= upper
            ):
                return False
    timestamp = call["timestamp"]
    if not isinstance(timestamp, str) or not re.fullmatch(
        r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d)", timestamp
    ):
        return False
    try:
        instant = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if (
            not datetime(1970, 1, 1, tzinfo=timezone.utc)
            <= instant
            < datetime(2100, 1, 1, tzinfo=timezone.utc)
        ):
            return False
    except ValueError:
        return False
    duration = call["duration_ms"]
    keys = call.get("argument_keys", [])
    metadata = call.get("metadata", {})
    return (
        type(duration) in (int, float)
        and math.isfinite(duration)
        and duration >= 0
        and call["outcome"] in ("success", "error", "cancelled")
        and isinstance(keys, list)
        and len(keys) <= 100
        and all(
            isinstance(key, str) and 0 < len(key.encode("utf-16-le")) // 2 <= 256
            for key in keys
        )
        and all(
            len(json.dumps(call[key], ensure_ascii=False).encode("utf-8")) <= 128 * 1024
            for key in ("input", "output")
            if key in call
        )
        and isinstance(metadata, dict)
        and len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) <= 16 * 1024
    )


def _retry_delay(value: Optional[str]) -> float:
    if value is None:
        return 0
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0
