from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.instrumentation.llm.llm_minimax.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI

    TClient = Union[OpenAI, AsyncOpenAI]


def wrap_minimax_client_sync(client: OpenAI) -> OpenAI:
    wrap_chat_completions_create_sync(client)
    return client


def wrap_minimax_client_async(client: AsyncOpenAI) -> AsyncOpenAI:
    wrap_chat_completions_create_async(client)
    return client


@typing.overload
def wrap_minimax_client(client: OpenAI) -> OpenAI: ...
@typing.overload
def wrap_minimax_client(  # type: ignore[overload-cannot-match]
    client: AsyncOpenAI,
) -> AsyncOpenAI: ...


def wrap_minimax_client(client: TClient) -> TClient:
    from judgeval.instrumentation.llm.llm_minimax.config import HAS_MINIMAX
    from judgeval.logger import judgeval_logger

    if not HAS_MINIMAX:
        judgeval_logger.error(
            "Cannot wrap MiniMax client: 'openai' library not installed. "
            "Install it with: pip install openai"
        )
        return client

    from openai import OpenAI, AsyncOpenAI

    if isinstance(client, AsyncOpenAI):
        return wrap_minimax_client_async(client)
    elif isinstance(client, OpenAI):
        return wrap_minimax_client_sync(client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
