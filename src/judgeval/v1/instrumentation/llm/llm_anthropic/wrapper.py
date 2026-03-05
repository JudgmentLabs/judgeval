from __future__ import annotations
from typing import TYPE_CHECKING, Union
import typing

from judgeval.v1.instrumentation.llm.llm_anthropic.messages import (
    wrap_messages_create_sync,
    wrap_messages_create_async,
)
from judgeval.v1.instrumentation.llm.llm_anthropic.messages_stream import (
    wrap_messages_stream_sync,
    wrap_messages_stream_async,
)

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic

    TClient = Union[Anthropic, AsyncAnthropic]


def wrap_anthropic_client_sync(client: Anthropic) -> Anthropic:
    wrap_messages_create_sync(client)
    wrap_messages_stream_sync(client)
    return client


def wrap_anthropic_client_async(client: AsyncAnthropic) -> AsyncAnthropic:
    wrap_messages_create_async(client)
    wrap_messages_stream_async(client)
    return client


@typing.overload
def wrap_anthropic_client(client: Anthropic) -> Anthropic: ...
@typing.overload
def wrap_anthropic_client(client: AsyncAnthropic) -> AsyncAnthropic: ...


def wrap_anthropic_client(client: TClient) -> TClient:
    from judgeval.v1.instrumentation.llm.llm_anthropic.config import HAS_ANTHROPIC
    from judgeval.logger import judgeval_logger

    if not HAS_ANTHROPIC:
        judgeval_logger.error(
            "Cannot wrap Anthropic client: 'anthropic' library not installed. "
            "Install it with: pip install anthropic"
        )
        return client

    from anthropic import Anthropic, AsyncAnthropic

    if isinstance(client, AsyncAnthropic):
        return wrap_anthropic_client_async(client)
    elif isinstance(client, Anthropic):
        return wrap_anthropic_client_sync(client)
    else:
        raise TypeError(f"Invalid client type: {type(client)}")
