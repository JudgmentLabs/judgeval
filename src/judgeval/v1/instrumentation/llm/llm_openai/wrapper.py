from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.v1.instrumentation.llm.llm_openai.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from judgeval.v1.instrumentation.llm.llm_openai.responses import (
    wrap_responses_create_sync,
    wrap_responses_create_async,
)
from judgeval.v1.instrumentation.llm.llm_openai.beta_chat_completions import (
    wrap_beta_chat_completions_parse_sync,
    wrap_beta_chat_completions_parse_async,
)
from judgeval.v1.instrumentation.llm.llm_openai.with_streaming_response import (
    wrap_chat_with_streaming_response_sync,
    wrap_chat_with_streaming_response_async,
    wrap_responses_with_streaming_response_sync,
    wrap_responses_with_streaming_response_async,
)

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI

    TClient = Union[OpenAI, AsyncOpenAI]


def wrap_openai_client_sync(client: OpenAI) -> OpenAI:
    wrap_chat_completions_create_sync(client)
    wrap_responses_create_sync(client)
    wrap_beta_chat_completions_parse_sync(client)
    wrap_chat_with_streaming_response_sync(client)
    wrap_responses_with_streaming_response_sync(client)
    return client


def wrap_openai_client_async(client: AsyncOpenAI) -> AsyncOpenAI:
    wrap_chat_completions_create_async(client)
    wrap_responses_create_async(client)
    wrap_beta_chat_completions_parse_async(client)
    wrap_chat_with_streaming_response_async(client)
    wrap_responses_with_streaming_response_async(client)
    return client


@typing.overload
def wrap_openai_client(client: OpenAI) -> OpenAI: ...
@typing.overload
def wrap_openai_client(client: AsyncOpenAI) -> AsyncOpenAI: ...


def wrap_openai_client(client: TClient) -> TClient:
    from judgeval.v1.instrumentation.llm.llm_openai.config import HAS_OPENAI
    from judgeval.logger import judgeval_logger

    if not HAS_OPENAI:
        judgeval_logger.error(
            "Cannot wrap OpenAI client: 'openai' library not installed. "
            "Install it with: pip install openai"
        )
        return client

    from openai import OpenAI, AsyncOpenAI

    if isinstance(client, AsyncOpenAI):
        return wrap_openai_client_async(client)
    elif isinstance(client, OpenAI):
        return wrap_openai_client_sync(client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
