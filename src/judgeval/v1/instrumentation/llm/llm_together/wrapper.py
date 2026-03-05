from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.v1.instrumentation.llm.llm_together.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)


if TYPE_CHECKING:
    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    TClient = Union[Together, AsyncTogether]


def wrap_together_client_sync(client: Together) -> Together:
    wrap_chat_completions_create_sync(client)
    return client


def wrap_together_client_async(client: AsyncTogether) -> AsyncTogether:
    wrap_chat_completions_create_async(client)
    return client


@typing.overload
def wrap_together_client(client: Together) -> Together: ...
@typing.overload
def wrap_together_client(  # type: ignore[overload-cannot-match]
    client: AsyncTogether,
) -> AsyncTogether: ...


def wrap_together_client(client: TClient) -> TClient:
    from judgeval.v1.instrumentation.llm.llm_together.config import HAS_TOGETHER
    from judgeval.logger import judgeval_logger

    if not HAS_TOGETHER:
        judgeval_logger.error(
            "Cannot wrap Together client: 'together' library not installed. "
            "Install it with: pip install together"
        )
        return client

    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    if isinstance(client, AsyncTogether):
        return wrap_together_client_async(client)
    elif isinstance(client, Together):
        return wrap_together_client_sync(client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
