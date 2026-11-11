"""Bounded native MCP delivery, independent of OTel sampling and export."""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Literal, Optional, TypedDict, Mapping
from urllib.parse import quote

import httpx

from judgeval.logger import judgeval_logger

PayloadField = Literal["input", "output"]
RedactPayload = Callable[[Any, PayloadField], Any]
_MAX_QUEUE = 1_000
_MAX_QUEUE_BYTES = 8 * 1024 * 1024
_BATCH_SIZE = 100
_FLUSH_INTERVAL = 1.0
_REQUEST_TIMEOUT = 5.0
_MAX_ATTEMPTS = 3
OMITTED = object()


class _CompletedCall(TypedDict):
    timestamp: str
    duration_ms: float
    outcome: Literal["success", "error", "cancelled"]
    server_name: str
    tool_name: str
    argument_keys: list[str]
    metadata: dict[str, Any]


class McpCall(_CompletedCall, total=False):
    call_id: str
    session_id: str
    environment_name: str
    client_name: str
    user_id: str
    intent: str
    model: str
    error_type: str
    error_message: str
    input: Any
    output: Any
    trace_id: str
    span_id: str


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
        self._url = (
            f"{api_url.rstrip('/')}/v1/projects/{quote(project_id, safe='')}/mcp/calls"
        )
        self._client = httpx.Client(
            timeout=_REQUEST_TIMEOUT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Organization-Id": organization_id,
                "Content-Type": "application/json",
            },
        )
        self._queue: list[str] = []
        self._queued_bytes = 0
        self._lock = threading.Lock()
        self._sending = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._closed = False

    def enqueue(self, call: Mapping[str, Any]) -> None:
        serialized = json.dumps(call, ensure_ascii=False, allow_nan=False)
        size = len(serialized.encode("utf-8"))
        with self._lock:
            if (
                self._closed
                or len(self._queue) >= _MAX_QUEUE
                or self._queued_bytes + size > _MAX_QUEUE_BYTES
            ):
                judgeval_logger.warning("MCP export queue unavailable; dropping call")
                return
            self._queue.append(serialized)
            self._queued_bytes += size
            if self._timer is None:
                self._timer = threading.Timer(_FLUSH_INTERVAL, self.force_flush)
                self._timer.daemon = True
                self._timer.start()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        deadline = time.monotonic() + timeout_millis / 1000
        if not self._sending.acquire(timeout=max(0, deadline - time.monotonic())):
            return False
        try:
            with self._lock:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
                calls, self._queue = self._queue, []
                self._queued_bytes = 0
            delivered = True
            for offset in range(0, len(calls), _BATCH_SIZE):
                body = (
                    '{"calls":[' + ",".join(calls[offset : offset + _BATCH_SIZE]) + "]}"
                )
                delivered = self._send(body, deadline) and delivered
            return delivered and time.monotonic() <= deadline
        finally:
            self._sending.release()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self.force_flush()
        self._client.close()

    def _send(self, body: str, deadline: float) -> bool:
        for attempt in range(_MAX_ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = self._client.post(
                    self._url,
                    content=body.encode("utf-8"),
                    timeout=min(_REQUEST_TIMEOUT, remaining),
                )
                if response.is_success:
                    return True
                if (
                    response.status_code not in (408, 429)
                    and response.status_code < 500
                ):
                    break
            except Exception:
                pass
            if attempt + 1 < _MAX_ATTEMPTS:
                time.sleep(min(0.25 * 2**attempt, max(0, deadline - time.monotonic())))
        judgeval_logger.warning("MCP export failed; dropping batch")
        return False
