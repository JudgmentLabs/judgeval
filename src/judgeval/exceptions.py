from __future__ import annotations

from httpx import HTTPError, Response
from typing import Optional


class JudgmentAPIError(HTTPError):
    status_code: int
    detail: str
    response: Optional[Response]

    def __init__(self, status_code: int, detail: str, response: Optional[Response]):
        self.status_code = status_code
        self.detail = detail
        self.response = response
        super().__init__(f"{status_code}: {detail}")


class JudgmentConflictError(JudgmentAPIError):
    """Raised when the server reports a conflict (HTTP 409).

    For example, creating a dataset whose name already exists in the
    project, or an illegal test-run status transition.
    """

    ...


class JudgmentValidationError(JudgmentAPIError):
    """Raised when the server rejects a request as invalid (HTTP 422).

    For example, dataset examples that fail JSON Schema validation, an
    incompatible judge/dataset pairing, or an unknown judge version.
    """

    ...


def map_judgment_api_error(
    error: JudgmentAPIError, message: Optional[str] = None
) -> JudgmentAPIError:
    """Map a raw `JudgmentAPIError` to a more specific SDK exception.

    409 responses become `JudgmentConflictError` and 422 responses become
    `JudgmentValidationError`; other statuses are returned unchanged.

    Args:
        error: The original API error.
        message: Optional message overriding the server-provided detail.
    """
    detail = message or error.detail
    if error.status_code == 409:
        return JudgmentConflictError(error.status_code, detail, error.response)
    if error.status_code == 422:
        return JudgmentValidationError(error.status_code, detail, error.response)
    if message:
        return JudgmentAPIError(error.status_code, detail, error.response)
    return error


class JudgmentTestError(Exception): ...


class JudgmentRuntimeError(RuntimeError):
    """Raised when judgeval encounters an unrecoverable runtime error."""

    ...


class InvalidJudgeModelError(Exception):
    """Raised when a judge is configured with an unsupported model."""

    ...


__all__ = (
    "JudgmentAPIError",
    "JudgmentConflictError",
    "JudgmentValidationError",
    "JudgmentRuntimeError",
    "JudgmentTestError",
    "InvalidJudgeModelError",
    "map_judgment_api_error",
)
