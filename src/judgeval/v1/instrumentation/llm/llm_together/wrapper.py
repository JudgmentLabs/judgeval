from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.tracer.llm.llm_together.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)


if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    TClient = Union[Together, AsyncTogether]


def wrap_together_client_sync(tracer: Tracer, client: Together) -> Together:
    wrap_chat_completions_create_sync(tracer, client)
    return client


def wrap_together_client_async(tracer: Tracer, client: AsyncTogether) -> AsyncTogether:
    wrap_chat_completions_create_async(tracer, client)
    return client


@typing.overload
def wrap_together_client(tracer: Tracer, client: Together) -> Together: ...
@typing.overload
def wrap_together_client(tracer: Tracer, client: AsyncTogether) -> AsyncTogether: ...  # type: ignore[overload-cannot-match]


def wrap_together_client(tracer: Tracer, client: TClient) -> TClient:
    from judgeval.tracer.llm.llm_together.config import HAS_TOGETHER
    from judgeval.logger import judgeval_logger

    if not HAS_TOGETHER:
        judgeval_logger.error(
            "Cannot wrap Together client: 'together' library not installed. "
            "Install it with: pip install together"
        )
        return client

    from together import Together, AsyncTogether  # type: ignore[import-untyped]

    if isinstance(client, AsyncTogether):
        return wrap_together_client_async(tracer, client)
    elif isinstance(client, Together):
        return wrap_together_client_sync(tracer, client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
