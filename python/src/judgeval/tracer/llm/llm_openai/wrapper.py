from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.tracer.llm.llm_openai.chat_completions import (
    wrap_chat_completions_create_sync,
    wrap_chat_completions_create_async,
)
from judgeval.tracer.llm.llm_openai.responses import (
    wrap_responses_create_sync,
    wrap_responses_create_async,
)
from judgeval.tracer.llm.llm_openai.beta_chat_completions import (
    wrap_beta_chat_completions_parse_sync,
    wrap_beta_chat_completions_parse_async,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from openai import OpenAI, AsyncOpenAI

    TClient = Union[OpenAI, AsyncOpenAI]


def wrap_openai_client_sync(tracer: Tracer, client: OpenAI) -> OpenAI:
    wrap_chat_completions_create_sync(tracer, client)
    wrap_responses_create_sync(tracer, client)
    wrap_beta_chat_completions_parse_sync(tracer, client)
    return client


def wrap_openai_client_async(tracer: Tracer, client: AsyncOpenAI) -> AsyncOpenAI:
    wrap_chat_completions_create_async(tracer, client)
    wrap_responses_create_async(tracer, client)
    wrap_beta_chat_completions_parse_async(tracer, client)
    return client


@typing.overload
def wrap_openai_client(tracer: Tracer, client: OpenAI) -> OpenAI: ...
@typing.overload
def wrap_openai_client(tracer: Tracer, client: AsyncOpenAI) -> AsyncOpenAI: ...


def wrap_openai_client(tracer: Tracer, client: TClient) -> TClient:
    from judgeval.tracer.llm.llm_openai.config import HAS_OPENAI
    from judgeval.logger import judgeval_logger

    if not HAS_OPENAI:
        judgeval_logger.error(
            "Cannot wrap OpenAI client: 'openai' library not installed. "
            "Install it with: pip install openai"
        )
        return client

    from openai import OpenAI, AsyncOpenAI

    if isinstance(client, AsyncOpenAI):
        return wrap_openai_client_async(tracer, client)
    elif isinstance(client, OpenAI):
        return wrap_openai_client_sync(tracer, client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
