from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Tuple,
    Protocol,
    TypeVar,
    Union,
    Sequence,
    Callable,
    Iterator,
    AsyncIterator,
    runtime_checkable,
)

from judgeval.tracer.llm.llm_openai.config import (
    HAS_OPENAI,
    openai_OpenAI,
    openai_AsyncOpenAI,
)
from judgeval.tracer.managers import sync_span_context, async_span_context
from judgeval.logger import judgeval_logger
from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from opentelemetry.trace import Span


@runtime_checkable
class OpenAIPromptTokensDetails(Protocol):
    cached_tokens: Optional[int]


@runtime_checkable
class OpenAIUsage(Protocol):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    prompt_tokens_details: Optional[OpenAIPromptTokensDetails]


@runtime_checkable
class OpenAIResponseUsage(Protocol):
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]


@runtime_checkable
class OpenAIUnifiedUsage(Protocol):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]

    input_tokens: Optional[int]
    output_tokens: Optional[int]

    total_tokens: Optional[int]
    prompt_tokens_details: Optional[OpenAIPromptTokensDetails]


@runtime_checkable
class OpenAIMessage(Protocol):
    content: Optional[str]
    role: str


@runtime_checkable
class OpenAIParsedMessage(Protocol):
    parsed: Optional[str]
    content: Optional[str]
    role: str


@runtime_checkable
class OpenAIChoice(Protocol):
    index: int
    message: OpenAIMessage
    finish_reason: Optional[str]


@runtime_checkable
class OpenAIParsedChoice(Protocol):
    index: int
    message: OpenAIParsedMessage
    finish_reason: Optional[str]


@runtime_checkable
class OpenAIResponseContent(Protocol):
    text: str


@runtime_checkable
class OpenAIResponseOutput(Protocol):
    content: Sequence[OpenAIResponseContent]


@runtime_checkable
class OpenAIChatCompletionBase(Protocol):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[Union[OpenAIChoice, OpenAIParsedChoice]]
    usage: Optional[OpenAIUnifiedUsage]


OpenAIChatCompletion = OpenAIChatCompletionBase
OpenAIParsedChatCompletion = OpenAIChatCompletionBase


@runtime_checkable
class OpenAIResponse(Protocol):
    id: str
    object: str
    created: int
    model: str
    output: Sequence[OpenAIResponseOutput]
    usage: Optional[OpenAIUnifiedUsage]


@runtime_checkable
class OpenAIStreamDelta(Protocol):
    content: Optional[str]


@runtime_checkable
class OpenAIStreamChoice(Protocol):
    index: int
    delta: OpenAIStreamDelta


@runtime_checkable
class OpenAIStreamChunk(Protocol):
    choices: Sequence[OpenAIStreamChoice]
    usage: Optional[OpenAIUnifiedUsage]


@runtime_checkable
class OpenAIClient(Protocol):
    pass


@runtime_checkable
class OpenAIAsyncClient(Protocol):
    pass


OpenAIResponseType = Union[OpenAIChatCompletionBase, OpenAIResponse]
OpenAIStreamType = Union[Iterator[OpenAIStreamChunk], AsyncIterator[OpenAIStreamChunk]]


def _extract_openai_content(chunk: OpenAIStreamChunk) -> str:
    if chunk.choices and len(chunk.choices) > 0:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            return delta_content
    return ""


def _extract_openai_tokens(usage_data: OpenAIUnifiedUsage) -> Tuple[int, int, int, int]:
    if hasattr(usage_data, "prompt_tokens") and usage_data.prompt_tokens is not None:
        prompt_tokens = usage_data.prompt_tokens
        completion_tokens = usage_data.completion_tokens or 0

    elif hasattr(usage_data, "input_tokens") and usage_data.input_tokens is not None:
        prompt_tokens = usage_data.input_tokens
        completion_tokens = usage_data.output_tokens or 0
    else:
        prompt_tokens = 0
        completion_tokens = 0

    # Extract cached tokens
    cache_read_input_tokens = 0
    if (
        usage_data.prompt_tokens_details
        and usage_data.prompt_tokens_details.cached_tokens
    ):
        cache_read_input_tokens = usage_data.prompt_tokens_details.cached_tokens

    cache_creation_input_tokens = 0  # OpenAI doesn't have cache creation tokens

    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _format_openai_output(
    response: OpenAIResponseType,
) -> Tuple[Optional[Union[str, list[dict[str, Any]]]], Optional[OpenAIUnifiedUsage]]:
    message_content: Optional[Union[str, list[dict[str, Any]]]] = None
    usage_data: Optional[OpenAIUnifiedUsage] = None

    try:
        if isinstance(response, OpenAIResponse):
            usage_data = response.usage
            if response.output and len(response.output) > 0:
                output0 = response.output[0]
                if output0.content and len(output0.content) > 0:
                    try:
                        content_blocks = []
                        for seg in output0.content:
                            if hasattr(seg, "type"):
                                seg_type = getattr(seg, "type", None)
                                if seg_type == "text" and hasattr(seg, "text"):
                                    block_data = {
                                        "type": "text",
                                        "text": getattr(seg, "text", ""),
                                    }
                                elif seg_type == "function_call":
                                    block_data = {
                                        "type": "function_call",
                                        "name": getattr(seg, "name", None),
                                        "call_id": getattr(seg, "call_id", None),
                                        "arguments": getattr(seg, "arguments", None),
                                    }
                                else:
                                    # Handle unknown types
                                    block_data = {"type": seg_type}
                                    for attr in [
                                        "text",
                                        "name",
                                        "call_id",
                                        "arguments",
                                        "content",
                                    ]:
                                        if hasattr(seg, attr):
                                            block_data[attr] = getattr(seg, attr)
                                content_blocks.append(block_data)
                            elif hasattr(seg, "text") and seg.text:
                                # Fallback for segments without type
                                content_blocks.append(
                                    {"type": "text", "text": seg.text}
                                )

                        message_content = (
                            content_blocks if content_blocks else str(output0.content)
                        )
                    except (TypeError, AttributeError):
                        message_content = str(output0.content)
        elif isinstance(response, OpenAIChatCompletionBase):
            usage_data = response.usage
            if response.choices and len(response.choices) > 0:
                message = response.choices[0].message

                if (
                    hasattr(message, "parsed")
                    and getattr(message, "parsed", None) is not None
                ):
                    # For parsed responses, return as structured data
                    parsed_data = getattr(message, "parsed")
                    message_content = [{"type": "parsed", "content": parsed_data}]
                else:
                    content_blocks = []

                    # Handle regular content
                    if hasattr(message, "content") and message.content:
                        content_blocks.append(
                            {"type": "text", "text": str(message.content)}
                        )

                    # Handle tool calls (standard Chat Completions API)
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_call_data = {
                                "type": "tool_call",
                                "id": getattr(tool_call, "id", None),
                                "function": {
                                    "name": getattr(tool_call.function, "name", None)
                                    if hasattr(tool_call, "function")
                                    else None,
                                    "arguments": getattr(
                                        tool_call.function, "arguments", None
                                    )
                                    if hasattr(tool_call, "function")
                                    else None,
                                },
                            }
                            content_blocks.append(tool_call_data)

                    message_content = content_blocks if content_blocks else None
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


class TracedOpenAIGenerator:
    def __init__(
        self,
        tracer: Tracer,
        generator: Iterator[OpenAIStreamChunk],
        client: OpenAIClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.generator = generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __iter__(self) -> Iterator[OpenAIStreamChunk]:
        return self

    def __next__(self) -> OpenAIStreamChunk:
        try:
            chunk = next(self.generator)
            content = _extract_openai_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_openai_tokens(chunk.usage)
                )
                set_span_attribute(
                    self.span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                    completion_tokens,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    cache_read,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    cache_creation,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )
            return chunk
        except StopIteration:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


class TracedOpenAIAsyncGenerator:
    def __init__(
        self,
        tracer: Tracer,
        async_generator: AsyncIterator[OpenAIStreamChunk],
        client: OpenAIAsyncClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.async_generator = async_generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __aiter__(self) -> AsyncIterator[OpenAIStreamChunk]:
        return self

    async def __anext__(self) -> OpenAIStreamChunk:
        try:
            chunk = await self.async_generator.__anext__()
            content = _extract_openai_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_openai_tokens(chunk.usage)
                )
                set_span_attribute(
                    self.span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                    completion_tokens,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                    cache_read,
                )
                set_span_attribute(
                    self.span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    cache_creation,
                )

                set_span_attribute(
                    self.span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )
            return chunk
        except StopAsyncIteration:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


TClient = TypeVar("TClient", bound=Union[OpenAIClient, OpenAIAsyncClient])


def wrap_openai_client(tracer: Tracer, client: TClient) -> TClient:
    if not HAS_OPENAI:
        return client

    assert openai_OpenAI is not None
    assert openai_AsyncOpenAI is not None

    def wrapped(function: Callable, span_name: str):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if kwargs.get("stream", False):
                span = tracer.get_tracer().start_span(
                    span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                )
                tracer.add_agent_attributes_to_span(span)
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                )
                model_name = kwargs.get("model", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                stream_response = function(*args, **kwargs)
                return TracedOpenAIGenerator(
                    tracer, stream_response, client, span, model_name
                )
            else:
                with sync_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    try:
                        tracer.add_agent_attributes_to_span(span)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                        )
                        model_name = kwargs.get("model", "")
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[openai wrapped] Error adding span metadata: {e}"
                        )

                    response = function(*args, **kwargs)

                    try:
                        if isinstance(
                            response, (OpenAIChatCompletionBase, OpenAIResponse)
                        ):
                            output, usage_data = _format_openai_output(response)
                            # Serialize structured data to JSON for span attribute
                            if isinstance(output, list):
                                output_str = safe_serialize(output)
                            else:
                                output_str = str(output) if output is not None else None
                            set_span_attribute(
                                span, AttributeKeys.GEN_AI_COMPLETION, output_str
                            )
                            if usage_data:
                                (
                                    prompt_tokens,
                                    completion_tokens,
                                    cache_read,
                                    cache_creation,
                                ) = _extract_openai_tokens(usage_data)
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                    prompt_tokens,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                    completion_tokens,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                    cache_read,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                                    cache_creation,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                                    safe_serialize(usage_data),
                                )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                getattr(response, "model", model_name),
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[openai wrapped] Error adding span metadata: {e}"
                        )
                    finally:
                        return response

        return wrapper

    def wrapped_async(function: Callable, span_name: str):
        @functools.wraps(function)
        async def wrapper(*args, **kwargs):
            if kwargs.get("stream", False):
                span = tracer.get_tracer().start_span(
                    span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                )
                tracer.add_agent_attributes_to_span(span)
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                )
                model_name = kwargs.get("model", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)
                stream_response = await function(*args, **kwargs)
                return TracedOpenAIAsyncGenerator(
                    tracer, stream_response, client, span, model_name
                )
            else:
                async with async_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    try:
                        tracer.add_agent_attributes_to_span(span)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                        )
                        model_name = kwargs.get("model", "")
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[openai wrapped_async] Error adding span metadata: {e}"
                        )

                    response = await function(*args, **kwargs)

                    try:
                        if isinstance(
                            response, (OpenAIChatCompletionBase, OpenAIResponse)
                        ):
                            output, usage_data = _format_openai_output(response)
                            # Serialize structured data to JSON for span attribute
                            if isinstance(output, list):
                                output_str = safe_serialize(output)
                            else:
                                output_str = str(output) if output is not None else None
                            set_span_attribute(
                                span, AttributeKeys.GEN_AI_COMPLETION, output_str
                            )
                            if usage_data:
                                (
                                    prompt_tokens,
                                    completion_tokens,
                                    cache_read,
                                    cache_creation,
                                ) = _extract_openai_tokens(usage_data)
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                    prompt_tokens,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                    completion_tokens,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                    cache_read,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                                    cache_creation,
                                )
                                set_span_attribute(
                                    span,
                                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                                    safe_serialize(usage_data),
                                )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                getattr(response, "model", model_name),
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[openai wrapped_async] Error adding span metadata: {e}"
                        )
                    finally:
                        return response

        return wrapper

    span_name = "OPENAI_API_CALL"
    if isinstance(client, openai_OpenAI):
        setattr(
            client.chat.completions,
            "create",
            wrapped(client.chat.completions.create, span_name),
        )
        setattr(client.responses, "create", wrapped(client.responses.create, span_name))
        setattr(
            client.beta.chat.completions,
            "parse",
            wrapped(client.beta.chat.completions.parse, span_name),
        )
    elif isinstance(client, openai_AsyncOpenAI):
        setattr(
            client.chat.completions,
            "create",
            wrapped_async(client.chat.completions.create, span_name),
        )
        setattr(
            client.responses,
            "create",
            wrapped_async(client.responses.create, span_name),
        )
        setattr(
            client.beta.chat.completions,
            "parse",
            wrapped_async(client.beta.chat.completions.parse, span_name),
        )

    return client
