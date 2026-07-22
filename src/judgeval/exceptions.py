from __future__ import annotations

from httpx import HTTPError, Response
from typing import Optional


class JudgmentAPIError(HTTPError):
    status_code: int
    detail: str
    response: Optional[Response]

    code: Optional[str]
    hint: Optional[str]
    retry_after_seconds: Optional[int]

    def __init__(
        self,
        status_code: int,
        detail: str,
        response: Optional[Response],
        *,
        code: Optional[str] = None,
        hint: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
    ):
        self.status_code = status_code
        self.detail = detail
        self.response = response
        self.code = code
        self.hint = hint
        self.retry_after_seconds = retry_after_seconds
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
    error_type: type[JudgmentAPIError] | None = {
        409: JudgmentConflictError,
        422: JudgmentValidationError,
    }.get(error.status_code)
    if error_type is None:
        if not message:
            return error
        error_type = JudgmentAPIError
    return error_type(
        error.status_code,
        detail,
        error.response,
        code=error.code,
        hint=error.hint,
        retry_after_seconds=error.retry_after_seconds,
    )


class JudgmentTestError(Exception): ...


class JudgmentRuntimeError(RuntimeError):
    """Raised when judgeval encounters an unrecoverable runtime error."""

    ...


class JudgmentProjectNotFoundError(ValueError):
    """Raised when a project is not visible to the configured organization."""

    ...


class InvalidJudgeModelError(Exception):
    """Raised when a judge is configured with an unsupported model."""

    ...


__all__ = (
    "JudgmentAPIError",
    "JudgmentConflictError",
    "JudgmentProjectNotFoundError",
    "JudgmentValidationError",
    "JudgmentRuntimeError",
    "JudgmentTestError",
    "InvalidJudgeModelError",
    "map_judgment_api_error",
)
