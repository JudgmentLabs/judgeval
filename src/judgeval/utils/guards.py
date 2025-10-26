from __future__ import annotations

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


__all__ = ("expect_exists", "expect_api_key", "expect_organization_id")
