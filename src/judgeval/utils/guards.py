from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from judgeval.logger import judgeval_logger

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T")


def expect_exists(value: T | None, message: str, default: T) -> T:
    if not value:
        judgeval_logger.error(message)
        return default

    return value


def expect_api_key(api_key: str | None) -> str | None:
    return expect_exists(
        api_key,
        "API Key is not set, please set JUDGMENT_API_KEY in the environment variables or pass it as `api_key`",
        default=None,
    )


def expect_organization_id(organization_id: str | None) -> str | None:
    return expect_exists(
        organization_id,
        "Organization ID is not set, please set JUDGMENT_ORG_ID in the environment variables or pass it as `organization_id`",
        default=None,
    )


def expect_project_id(project_id: str | None) -> str | None:
    """
    Validates that a project_id exists. Returns None and logs if missing.

    Args:
        project_id: The project_id to validate.

    Returns:
        The project_id if it exists, None otherwise.
    """
    if project_id:
        return project_id

    caller = "unknown"
    frame = inspect.currentframe()
    try:
        if frame and frame.f_back:
            caller = frame.f_back.f_code.co_name
    finally:
        del frame
    judgeval_logger.error(f"project_id is not set. {caller}() will be skipped.")
    return None


__all__ = (
    "expect_exists",
    "expect_api_key",
    "expect_organization_id",
    "expect_project_id",
)
