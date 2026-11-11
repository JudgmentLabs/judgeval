from __future__ import annotations

import atexit
import copy
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import ClassVar, Optional

from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.internal.api.models import (
    IngestObservationsRequestObservation,
    Observation,
)
from judgeval.logger import judgeval_logger
from judgeval.utils import resolve_project_id
from judgeval.utils.decorators.dont_throw import dont_throw

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


class Conversations:
    """Records model calls in the conversation format, without traces.

    Call ``Conversations.init`` once at startup, then ``Conversations.record``
    from anywhere. Each call is stored in its conversation and evaluated by
    the project's conversation judges, as model calls recorded from traces
    are. Nothing here raises into your code: ``init`` logs a warning and
    leaves recording off when a credential is missing or the project is not
    found, and ``record`` queues the call and returns at once.

    Calls are sent in the background, in batches. Call
    ``Conversations.flush`` before a short-lived process, such as a
    serverless handler, returns; queued calls are also sent at interpreter
    exit.

    Examples:
        ```python
        from judgeval import Conversations

        Conversations.init(project_name="support-bot", environment="production")

        Conversations.record({
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
        })
        ```
    """

    _active: ClassVar[Optional[Conversations]] = None
    _not_initialized_reported: ClassVar[bool] = False

    def __init__(
        self,
        client: Optional[JudgmentSyncClient],
        project_id: Optional[str],
        environment: Optional[str],
    ):
        self._client = client
        self._project_id = project_id
        self._environment = environment
        self._reported: set[str] = set()
        self._reset_sender()

    @classmethod
    @dont_throw
    def init(
        cls,
        *,
        project_name: Optional[str] = None,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        api_url: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        """Configure where recorded calls are sent.

        Credentials not given are read from ``JUDGMENT_API_KEY``,
        ``JUDGMENT_ORG_ID``, and ``JUDGMENT_API_URL``. Calling ``init`` again
        sends what the previous configuration queued, then replaces it.

        Args:
            project_name: The Judgment project to record into.
            api_key: Your Judgment API key.
            organization_id: Your organization ID.
            api_url: Override the API endpoint URL.
            environment: The deployment environment calls run in, such as
                ``"production"``; judges can be limited to environments.
        """
        api_key = api_key or JUDGMENT_API_KEY
        organization_id = organization_id or JUDGMENT_ORG_ID
        api_url = api_url or JUDGMENT_API_URL
        missing = [
            name
            for name, value in [
                ("project_name", project_name),
                ("api_key", api_key),
                ("organization_id", organization_id),
                ("api_url", api_url),
            ]
            if not value
        ]
        client: Optional[JudgmentSyncClient] = None
        project_id: Optional[str] = None
        if missing:
            judgeval_logger.warning(
                f"{', '.join(missing)} not provided. "
                "Conversations will not record observations."
            )
        else:
            assert project_name and api_key and organization_id and api_url
            client = JudgmentSyncClient(api_url, api_key, organization_id)
            project_id = resolve_project_id(client, project_name)
            if not project_id:
                judgeval_logger.warning(
                    f"Project '{project_name}' not found. "
                    "Conversations will not record observations."
                )
        previous = cls._active
        cls._active = cls(client, project_id, environment)
        if previous is not None:
            previous._stop()

    @staticmethod
    @dont_throw
    def record(
        observation: Observation, *, observed_at: Optional[datetime] = None
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
        """
        active = Conversations._active
        if active is None:
            if not Conversations._not_initialized_reported:
                Conversations._not_initialized_reported = True
                judgeval_logger.warning(
                    "[Conversations] Conversations.init() was not called; "
                    "observations are not recorded"
                )
            return
        active._enqueue(observation, observed_at)

    @staticmethod
    def flush(timeout_seconds: float = 30.0) -> bool:
        """Send every queued call now.

        Args:
            timeout_seconds: The longest to wait.

        Returns:
            True if every queued call was sent or dropped in time.
        """
        active = Conversations._active
        return True if active is None else active._flush(timeout_seconds)

    @staticmethod
    def shutdown(timeout_seconds: float = 30.0) -> None:
        """Send queued calls, then stop recording.

        Calls recorded afterwards are ignored until ``init`` is called again.

        Args:
            timeout_seconds: The longest to wait for queued calls.
        """
        active = Conversations._active
        if active is not None:
            active._stop(timeout_seconds)

    def _reset_sender(self) -> None:
        self._queue: deque[IngestObservationsRequestObservation] = deque()
        self._condition = threading.Condition()
        self._sending = False
        self._flushing = 0
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def _enqueue(
        self, observation: Observation, observed_at: Optional[datetime]
    ) -> None:
        if self._client is None or self._project_id is None:
            return
        sent: IngestObservationsRequestObservation = {
            # Sent later, so copy it: changes the caller makes afterwards, such
            # as appending to a shared message history, are not sent.
            "observation": copy.deepcopy(observation),
            "observed_at": (observed_at or datetime.now(timezone.utc))
            .astimezone(timezone.utc)
            .isoformat(),
        }
        if self._environment is not None:
            sent["environment"] = self._environment
        with self._condition:
            if self._stopped:
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
                    target=self._run, name="judgeval-conversations", daemon=True
                )
                self._thread.start()
            if len(self._queue) >= MAX_BATCH_SIZE:
                self._condition.notify_all()

    def _flush(self, timeout_seconds: float) -> bool:
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

    def _stop(self, timeout_seconds: float = 30.0) -> None:
        self._flush(timeout_seconds)
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                if (
                    not self._stopped
                    and not self._flushing
                    and len(self._queue) < MAX_BATCH_SIZE
                ):
                    self._condition.wait(FLUSH_INTERVAL_SECONDS)
                if self._stopped and not self._queue:
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
                    f"[Conversations] Dropped {len(batch)} observations: {error!r}"
                )
            finally:
                with self._condition:
                    self._sending = False
                    self._condition.notify_all()

    def _send(self, batch: list[IngestObservationsRequestObservation]) -> None:
        client, project_id = self._client, self._project_id
        assert client is not None and project_id is not None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                client.post_projects_observations(project_id, {"observations": batch})
                return
            except JudgmentAPIError as error:
                if error.status_code == 400 and len(batch) > 1:
                    # The server rejects a request when any observation in it
                    # does not fit the format; find which by halving it.
                    middle = len(batch) // 2
                    self._send(batch[:middle])
                    self._send(batch[middle:])
                    return
                if error.status_code == 402:
                    self._report_once(
                        "usage",
                        "Observations are not stored: the organization is "
                        "over its ingestion usage limit",
                    )
                    return
                if error.status_code == 404:
                    self._report_once(
                        "disabled",
                        "Observations are not stored: conversations are not "
                        "enabled for this organization",
                    )
                    return
                if (
                    error.status_code not in RETRYABLE_STATUSES
                    or attempt == MAX_ATTEMPTS
                ):
                    judgeval_logger.error(
                        f"[Conversations] Dropped {len(batch)} observations: "
                        f"{error.status_code} {error.detail}"
                    )
                    return
                time.sleep(error.retry_after_seconds or 2 ** (attempt - 1))
            except Exception as error:
                if attempt == MAX_ATTEMPTS:
                    judgeval_logger.error(
                        f"[Conversations] Dropped {len(batch)} observations: {error!r}"
                    )
                    return
                time.sleep(2 ** (attempt - 1))

    def _report_once(self, reason: str, message: str) -> None:
        if reason in self._reported:
            return
        self._reported.add(reason)
        judgeval_logger.warning(f"[Conversations] {message}")


def _after_fork_in_child() -> None:
    # The parent's sender thread does not exist in a forked child, and its
    # queued calls are the parent's to send; start the child with an empty
    # sender.
    active = Conversations._active
    if active is not None:
        active._reset_sender()


atexit.register(Conversations.shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
