"""Bounded native MCP delivery, independent of OTel sampling and export."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
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
            self._schedule()

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
            min(delay, threading.TIMEOUT_MAX), self.force_flush
        )
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
        judgeval_logger.warning("MCP export failed; dropping batch")
        return True


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
