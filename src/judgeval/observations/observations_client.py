from __future__ import annotations

import atexit
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.internal.api.models import (
    IngestObservationsRequestObservation,
    Observation,
)
from judgeval.logger import judgeval_logger
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.serialize import json_encoder

# The most observations the server accepts in one request.
MAX_BATCH_SIZE = 100
# The most observations held while waiting to be sent; more are dropped.
MAX_QUEUE_SIZE = 2048
# How long an observation waits for its batch to fill before it is sent.
FLUSH_INTERVAL_SECONDS = 1.0
# How many times a batch is sent before a temporary failure drops it.
MAX_ATTEMPTS = 3
# Statuses that mean a retry may succeed.
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


class ObservationsClient:
    """Sends observations to Judgment without traces.

    ``record`` returns at once and never raises: observations are queued and
    sent in the background, in batches. Each one is stored in its conversation
    and evaluated by the project's conversation judges, as model calls recorded
    from traces are. Call ``flush`` before a short-lived process exits; pending
    observations are also sent when the interpreter exits.

    Get it from ``Judgeval.observations``.
    """

    def __init__(self, client: JudgmentSyncClient, project_id: Optional[str]):
        self._client = client
        self._project_id = project_id
        self._queue: deque[IngestObservationsRequestObservation] = deque()
        self._condition = threading.Condition()
        self._sending = False
        self._flushing = 0
        self._shutdown = False
        self._thread: Optional[threading.Thread] = None
        self._reported: set[str] = set()
        atexit.register(self.shutdown)

    @dont_throw
    def record(
        self,
        observation: Observation,
        *,
        observed_at: Optional[datetime] = None,
        environment: Optional[str] = None,
    ) -> None:
        """Queue one model call to be sent to Judgment.

        Args:
            observation: The call, in the conversation format: its
                ``conversation_id``, ``input`` (everything the model was
                given, including history from earlier calls), and optionally
                its ``output``, ``tools``, ``model``, ``usage``, and more.
            observed_at: When the call completed; defaults to now. A naive
                datetime is read as local time. Two identical calls at
                different times are two calls.
            environment: The deployment environment the call ran in, such as
                ``"production"``; judges can be limited to environments.

        Examples:
            ```python
            client = Judgeval(project_name="support-bot")
            client.observations.record({
                "conversation_id": "ticket-4812",
                "model": "my-model-v2",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Where is my order?"}],
                }],
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "It ships today."}],
                }],
            }, environment="production")
            ```
        """
        if self._project_id is None:
            self._report_once(
                "project",
                "Observations are not sent: the project was not found",
            )
            return
        sent: IngestObservationsRequestObservation = {
            # Encoded now, as it will be sent, so a value that cannot be sent
            # fails here and later changes to the caller's dict are not sent.
            "observation": json_encoder(observation),
            "observed_at": (observed_at or datetime.now(timezone.utc))
            .astimezone(timezone.utc)
            .isoformat(),
        }
        if environment is not None:
            sent["environment"] = environment
        with self._condition:
            if self._shutdown:
                return
            if len(self._queue) >= MAX_QUEUE_SIZE:
                self._report_once(
                    "full",
                    f"Observation queue is full ({MAX_QUEUE_SIZE}); "
                    "dropping observations until it drains",
                )
                return
            self._queue.append(sent)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="judgeval-observations", daemon=True
                )
                self._thread.start()
            if len(self._queue) >= MAX_BATCH_SIZE:
                self._condition.notify_all()

    def flush(self, timeout_seconds: float = 30.0) -> bool:
        """Send every queued observation now.

        Args:
            timeout_seconds: The longest to wait.

        Returns:
            True if every queued observation was sent or dropped in time.
        """
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            self._flushing += 1
            self._condition.notify_all()
            try:
                while self._queue or self._sending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(remaining)
                return True
            finally:
                self._flushing -= 1

    def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Send queued observations, then stop sending.

        Observations recorded afterwards are ignored.

        Args:
            timeout_seconds: The longest to wait for queued observations.
        """
        self.flush(timeout_seconds)
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if (
                    not self._shutdown
                    and not self._flushing
                    and len(self._queue) < MAX_BATCH_SIZE
                ):
                    self._condition.wait(FLUSH_INTERVAL_SECONDS)
                if self._shutdown and not self._queue:
                    return
                batch = [
                    self._queue.popleft()
                    for _ in range(min(MAX_BATCH_SIZE, len(self._queue)))
                ]
                self._sending = bool(batch)
            try:
                if batch:
                    self._send(batch)
            except Exception as error:
                judgeval_logger.error(
                    f"[Observations] Failed to send observations: {error!r}"
                )
            finally:
                with self._condition:
                    self._sending = False
                    self._condition.notify_all()

    def _send(self, batch: list[IngestObservationsRequestObservation]) -> None:
        project_id = self._project_id
        assert project_id is not None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                self._client.post_projects_observations(
                    project_id, {"observations": batch}
                )
                return
            except JudgmentAPIError as error:
                if error.status_code == 400 and len(batch) > 1:
                    # The server rejects a request when any observation in it
                    # does not fit the format; find which by halving it.
                    middle = len(batch) // 2
                    self._send(batch[:middle])
                    self._send(batch[middle:])
                    return
                if error.status_code == 404:
                    self._report_once(
                        "disabled",
                        "Observations are not stored: conversations are not "
                        "enabled for this organization",
                    )
                    return
                if error.status_code == 402:
                    self._report_once(
                        "usage",
                        "Observations are not stored: the organization is "
                        "over its ingestion usage limit",
                    )
                    return
                if (
                    error.status_code not in RETRYABLE_STATUSES
                    or attempt == MAX_ATTEMPTS
                ):
                    judgeval_logger.error(
                        f"[Observations] Dropped {len(batch)} observations: "
                        f"{error.status_code} {error.detail}"
                    )
                    return
                time.sleep(error.retry_after_seconds or 2 ** (attempt - 1))
            except Exception as error:
                if attempt == MAX_ATTEMPTS:
                    judgeval_logger.error(
                        f"[Observations] Dropped {len(batch)} observations: {error!r}"
                    )
                    return
                time.sleep(2 ** (attempt - 1))

    def _report_once(self, reason: str, message: str) -> None:
        if reason in self._reported:
            return
        self._reported.add(reason)
        judgeval_logger.warning(f"[Observations] {message}")
