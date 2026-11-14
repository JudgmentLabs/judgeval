from __future__ import annotations

import json
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
# The longest rejection reason logged.
MAX_REASON_LENGTH = 2000


def _is_retryable(status: int) -> bool:
    """Server errors, timeouts, and rate limits may pass on retry; other
    client errors will not."""
    return not (400 <= status < 500 and status not in (408, 429))


def _backoff(attempt: int, retry_after_seconds: Optional[int]) -> None:
    """Waits exponentially longer after each attempt, and at least as long as
    the server's Retry-After; both are capped."""
    retry_after = min(retry_after_seconds or 0, MAX_BACKOFF_SECONDS)
    time.sleep(max(min(2**attempt, MAX_BACKOFF_SECONDS), retry_after))


def _read_rejection(detail: str, count: int) -> tuple[set[int], str]:
    """Reads a 400 from the observations route: the positions of the
    observations it names, and a reason to log. Each validation issue has a
    path such as ``["observations", 3, "input", 0, "role"]``. The positions
    are empty when no issue names an observation, as when the request as a
    whole is malformed or the detail is not in this format. The reason lists
    each issue by path. It never includes the detail itself, which can echo
    the request and so the caller's conversation content."""
    try:
        issues = json.loads(detail)["errors"]
        paths = [issue["path"] for issue in issues]
        reason = "; ".join(
            f"{'.'.join(map(str, issue['path']))}: {issue['message']}"
            for issue in issues
        )
    except Exception:
        return set(), "the request was rejected as invalid"
    invalid = {
        path[1]
        for path in paths
        if len(path) > 1
        and path[0] == "observations"
        and type(path[1]) is int
        and 0 <= path[1] < count
    }
    return invalid, reason[:MAX_REASON_LENGTH]


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
        invalid, reason = _read_rejection(rejection, len(batch))
        if not invalid:
            self._log_dropped(len(batch), reason)
            return
        # The server stores nothing from a rejected request, so the
        # observations it did not name are sent again, once.
        self._log_dropped(len(invalid), reason)
        rest = [item for index, item in enumerate(batch) if index not in invalid]
        if not rest:
            return
        rejection = self._send(rest)
        if rejection is not None:
            self._log_dropped(len(rest), _read_rejection(rejection, len(rest))[1])

    def _send(self, batch: list[IngestObservationsRequestObservation]) -> Optional[str]:
        """Sends a batch, retrying temporary failures. Returns the server's
        detail when it rejects the batch as invalid (400), or the error when
        the batch cannot be encoded. Handles every other outcome."""
        for attempt in range(MAX_ATTEMPTS):
            try:
                self._client.post_projects_observations(
                    self._project_id, {"observations": batch}
                )
                return None
            except JudgmentAPIError as error:
                status = error.status_code
                if status == 400:
                    return error.detail
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
