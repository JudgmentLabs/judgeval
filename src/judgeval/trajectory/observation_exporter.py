from __future__ import annotations

import time
from typing import Optional

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
        for attempt in range(MAX_ATTEMPTS):
            try:
                self._client.post_projects_observations(
                    self._project_id, {"observations": batch}
                )
                return
            except JudgmentAPIError as error:
                status = error.status_code
                if status == 400 and len(batch) > 1:
                    # The server rejects a request when any observation in it
                    # does not fit the format; find which by halving it.
                    middle = len(batch) // 2
                    self.export(batch[:middle])
                    self.export(batch[middle:])
                    return
                if status in (402, 404):
                    self._report_once(status)
                    return
                if not _is_retryable(status) or attempt == MAX_ATTEMPTS - 1:
                    judgeval_logger.error(
                        f"[Trajectory] Dropped {len(batch)} observations: "
                        f"{status} {error.detail}"
                    )
                    return
                _backoff(attempt, error.retry_after_seconds)
            except Exception as error:
                if attempt == MAX_ATTEMPTS - 1:
                    judgeval_logger.error(
                        f"[Trajectory] Dropped {len(batch)} observations: {error!r}"
                    )
                    return
                _backoff(attempt, None)

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
