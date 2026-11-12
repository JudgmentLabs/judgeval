from __future__ import annotations

import atexit
import copy
from datetime import datetime, timezone
from typing import ClassVar, Optional

from judgeval.trajectory.batch_observation_processor import (
    BatchObservationProcessor,
)
from judgeval.trajectory.observation_exporter import ObservationExporter
from judgeval.env import JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.internal.api.models import (
    IngestObservationsRequestObservation,
    Observation,
)
from judgeval.logger import judgeval_logger
from judgeval.utils import resolve_project_id
from judgeval.utils.decorators.dont_throw import dont_throw


class Trajectory:
    """Records model calls in the conversation format, without traces.

    Call ``Trajectory.init`` once at startup, then ``Trajectory.record``
    from anywhere. Each call is stored in its conversation and evaluated by
    the project's conversation judges, as model calls recorded from traces
    are. Nothing here raises into your code: ``init`` logs a warning and
    leaves recording off when a credential is missing or the project is not
    found, and ``record`` queues the call and returns at once.

    Calls are sent in the background, in batches. Call
    ``Trajectory.flush`` before a short-lived process, such as a
    serverless handler, returns; queued calls are also sent at interpreter
    exit.

    Examples:
        ```python
        from judgeval import Trajectory

        Trajectory.init(project_name="support-bot", environment="production")

        Trajectory.record({
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

    _initialized: ClassVar[bool] = False
    _processor: ClassVar[Optional[BatchObservationProcessor]] = None
    _environment: ClassVar[Optional[str]] = None
    _not_initialized_reported: ClassVar[bool] = False

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
        processor: Optional[BatchObservationProcessor] = None
        if project_name and api_key and organization_id and api_url:
            client = JudgmentSyncClient(api_url, api_key, organization_id)
            project_id = resolve_project_id(client, project_name)
            if project_id:
                processor = BatchObservationProcessor(
                    ObservationExporter(client, project_id)
                )
            else:
                judgeval_logger.warning(
                    f"Project '{project_name}' not found. "
                    "Trajectory will not record observations."
                )
        else:
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
            judgeval_logger.warning(
                f"{', '.join(missing)} not provided. "
                "Trajectory will not record observations."
            )
        previous = cls._processor
        cls._initialized = True
        cls._processor = processor
        cls._environment = environment
        if previous is not None:
            previous.shutdown()

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
        if not Trajectory._initialized:
            if not Trajectory._not_initialized_reported:
                Trajectory._not_initialized_reported = True
                judgeval_logger.warning(
                    "[Trajectory] Trajectory.init() was not called; "
                    "observations are not recorded"
                )
            return
        processor = Trajectory._processor
        if processor is None:
            return
        sent: IngestObservationsRequestObservation = {
            # Sent later, so copy it: changes the caller makes afterwards, such
            # as appending to a shared message history, are not sent.
            "observation": copy.deepcopy(observation),
            "observed_at": (observed_at or datetime.now(timezone.utc))
            .astimezone(timezone.utc)
            .isoformat(),
        }
        if Trajectory._environment is not None:
            sent["environment"] = Trajectory._environment
        processor.emit(sent)

    @staticmethod
    def flush() -> bool:
        """Send every queued call now, on the calling thread.

        Returns:
            False if recording is shut down.
        """
        processor = Trajectory._processor
        return True if processor is None else processor.force_flush()

    @staticmethod
    def shutdown(timeout_seconds: float = 30.0) -> None:
        """Send queued calls, then stop recording. Calls recorded afterwards
        are ignored until ``init`` is called again.

        Args:
            timeout_seconds: The longest to wait for queued calls.
        """
        processor = Trajectory._processor
        if processor is not None:
            processor.shutdown(timeout_seconds)


atexit.register(Trajectory.shutdown)
