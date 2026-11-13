from __future__ import annotations

import time
from typing import Optional

import httpx

from judgeval.exceptions import JudgmentAPIError
from judgeval.internal.api.api_client import JudgmentSyncClient
from judgeval.internal.api.models import IngestObservationsRequestObservation
from judgeval.logger import judgeval_logger

# How many times a batch is sent before a temporary failure drops it.
MAX_ATTEMPTS = 3
# The longest wait between attempts, whatever the server's Retry-After says.
MAX_BACKOFF_SECONDS = 30


def _is_retryable(status: int) -> bool:
    """Server errors, timeouts, and rate limits may pass on retry; other
    client errors will not."""
    return not (400 <= status < 500 and status not in (408, 429))


def _backoff(attempt: int, retry_after_seconds: Optional[int]) -> None:
    """Waits exponentially longer after each attempt, and at least as long as
    the server's Retry-After; both are capped."""
    retry_after = min(retry_after_seconds or 0, MAX_BACKOFF_SECONDS)
    time.sleep(max(min(2**attempt, MAX_BACKOFF_SECONDS), retry_after))


class ObservationExporter:
    """Sends one batch of observations to Judgment. Never raises: a batch
    that cannot be stored is logged and dropped."""

    def __init__(self, client: JudgmentSyncClient, project_id: str):
        self._client = client
        self._project_id = project_id
        self._reported: set[int] = set()

    def export(self, batch: list[IngestObservationsRequestObservation]) -> None:
        rejection = self._send(batch)
        if rejection is None:
            return
        if len(batch) > 1:
            self._drop_invalid(batch)
        else:
            self._log_dropped(1, rejection)

    def _drop_invalid(self, batch: list[IngestObservationsRequestObservation]) -> None:
        """The server rejects a request when any observation in it does not
        fit the format; this finds which by halving the batch. When both
        halves are rejected, the fault is most likely in the request as a
        whole, such as a clock more than five minutes fast, so the batch is
        dropped rather than split further."""
        middle = len(batch) // 2
        halves = (batch[:middle], batch[middle:])
        rejections = [self._send(half) for half in halves]
        if rejections[0] is not None and rejections[1] is not None:
            self._log_dropped(len(batch), rejections[0])
            return
        for half, rejection in zip(halves, rejections):
            if rejection is None:
                continue
            if len(half) > 1:
                self._drop_invalid(half)
            else:
                self._log_dropped(1, rejection)

    def _send(self, batch: list[IngestObservationsRequestObservation]) -> Optional[str]:
        """Sends a batch, retrying temporary failures. Returns the reason when
        the batch is invalid: the server rejects it (400), or it holds a value
        JSON cannot represent, such as NaN. Handles every other outcome."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                self._client.post_projects_observations(
                    self._project_id, {"observations": batch}
                )
                return None
            except JudgmentAPIError as error:
                status = error.status_code
                if status == 400:
                    return f"{status} {error.detail}"
                if status in (402, 404):
                    self._report_once(status)
                    return None
                if not _is_retryable(status) or attempt == MAX_ATTEMPTS - 1:
                    self._log_dropped(len(batch), f"{status} {error.detail}")
                    return None
                _backoff(attempt, error.retry_after_seconds)
            except httpx.TransportError as error:
                if attempt == MAX_ATTEMPTS - 1:
                    self._log_dropped(len(batch), repr(error))
                    return None
                _backoff(attempt, None)
            except Exception as error:
                # Raised before anything is sent.
                return repr(error)
        return None

    def _log_dropped(self, count: int, reason: str) -> None:
        judgeval_logger.error(f"[Trajectory] Dropped {count} observations: {reason}")

    def _report_once(self, status: int) -> None:
        if status in self._reported:
            return
        self._reported.add(status)
        reason = (
            "the organization is over its ingestion usage limit"
            if status == 402
            else "conversations are not enabled for this organization"
        )
        judgeval_logger.warning(f"[Trajectory] Observations are not stored: {reason}")
