from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    TypeVar,
    Tuple,
    Union,
    Iterator,
    AsyncIterator,
    Sequence,
    runtime_checkable,
)

from judgeval.tracer.llm.llm_anthropic.config import (
    anthropic_Anthropic,
    anthropic_AsyncAnthropic,
)
from judgeval.tracer.managers import sync_span_context, async_span_context
from judgeval.logger import judgeval_logger
from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from opentelemetry.trace import Span


# Content block protocols
@runtime_checkable
class AnthropicContentBlock(Protocol):
    text: str
    type: str


# Usage protocols
@runtime_checkable
class AnthropicUsage(Protocol):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: Optional[int]
    cache_creation_input_tokens: Optional[int]


# Message protocols
@runtime_checkable
class AnthropicMessage(Protocol):
    content: Sequence[AnthropicContentBlock]
    usage: AnthropicUsage
    model: Optional[str]


# Stream event protocols
@runtime_checkable
class AnthropicStreamDelta(Protocol):
    text: Optional[str]


@runtime_checkable
class AnthropicStreamEvent(Protocol):
    type: str
    delta: Optional[AnthropicStreamDelta]
    message: Optional[AnthropicMessage]
    usage: Optional[AnthropicUsage]


# Client protocols
@runtime_checkable
class AnthropicClient(Protocol):
    pass


@runtime_checkable
class AnthropicAsyncClient(Protocol):
    pass


# Generic client type bound to both sync and async client protocols
TClient = TypeVar("TClient", bound=Union[AnthropicClient, AnthropicAsyncClient])


# Union types
AnthropicResponseType = AnthropicMessage
AnthropicStreamType = Union[
    Iterator[AnthropicStreamEvent], AsyncIterator[AnthropicStreamEvent]
]


def _extract_anthropic_content(chunk: AnthropicStreamEvent) -> str:
    if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "text"):
        return chunk.delta.text or ""

    if isinstance(chunk, AnthropicStreamEvent) and chunk.type == "content_block_delta":
        if chunk.delta and chunk.delta.text:
            return chunk.delta.text
    return ""


def _extract_anthropic_tokens(usage_data: AnthropicUsage) -> Tuple[int, int, int, int]:
    prompt_tokens = usage_data.input_tokens or 0
    completion_tokens = usage_data.output_tokens or 0
    cache_read_input_tokens = usage_data.cache_read_input_tokens or 0
    cache_creation_input_tokens = usage_data.cache_creation_input_tokens or 0

    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _extract_anthropic_chunk_usage(
    chunk: AnthropicStreamEvent,
) -> Optional[AnthropicUsage]:
    if hasattr(chunk, "usage") and chunk.usage:
        return chunk.usage

    if isinstance(chunk, AnthropicStreamEvent):
        if chunk.type == "message_start" and chunk.message:
            return chunk.message.usage
        elif chunk.type in ("message_delta", "message_stop"):
            return chunk.usage
    return None


def _format_anthropic_output(
    response: AnthropicMessage,
) -> Tuple[Optional[Union[str, list]], Optional[AnthropicUsage]]:
    message_content: Optional[Union[str, list]] = None
    usage_data: Optional[AnthropicUsage] = None

    try:
        if isinstance(response, AnthropicMessage):
            usage_data = response.usage
            if response.content:
                content_blocks = []
                for block in response.content:
                    if isinstance(block, AnthropicContentBlock):
                        block_type = getattr(block, "type", None)
                        if block_type == "text":
                            block_data = {
                                "type": "text",
                                "text": getattr(block, "text", ""),
                            }
                            # Add citations if present
                            if hasattr(block, "citations"):
                                block_data["citations"] = getattr(
                                    block, "citations", None
                                )
                        elif block_type == "tool_use":
                            block_data = {
                                "type": "tool_use",
                                "id": getattr(block, "id", None),
                                "name": getattr(block, "name", None),
                                "input": getattr(block, "input", None),
                            }
                        elif block_type == "tool_result":
                            block_data = {
                                "type": "tool_result",
                                "tool_use_id": getattr(block, "tool_use_id", None),
                                "content": getattr(block, "content", None),
                            }
                        else:
                            # Handle unknown block types
                            block_data = {"type": block_type}
                            for attr in [
                                "id",
                                "text",
                                "name",
                                "input",
                                "content",
                                "tool_use_id",
                                "citations",
                            ]:
                                if hasattr(block, attr):
                                    block_data[attr] = getattr(block, attr)

                        content_blocks.append(block_data)

                # Return structured data instead of string
                message_content = content_blocks if content_blocks else None
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


class TracedAnthropicGenerator:
    def __init__(
        self,
        tracer: Tracer,
        generator: Iterator[AnthropicStreamEvent],
        client: AnthropicClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.generator = generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __iter__(self) -> Iterator[AnthropicStreamEvent]:
        return self

    def __next__(self) -> AnthropicStreamEvent:
        try:
            chunk = next(self.generator)
            content = _extract_anthropic_content(chunk)
            if content:
                self.accumulated_content += content

            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
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
                    safe_serialize(usage_data),
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


class TracedAnthropicAsyncGenerator:
    def __init__(
        self,
        tracer: Tracer,
        async_generator: AsyncIterator[AnthropicStreamEvent],
        client: AnthropicAsyncClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.async_generator = async_generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __aiter__(self) -> AsyncIterator[AnthropicStreamEvent]:
        return self

    async def __anext__(self) -> AnthropicStreamEvent:
        try:
            chunk = await self.async_generator.__anext__()
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

        try:
            content = _extract_anthropic_content(chunk)
            if content:
                self.accumulated_content += content

            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
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
                    safe_serialize(usage_data),
                )
        except Exception as e:
            if self.span:
                self.span.end()
            judgeval_logger.error(
                f"[anthropic wrapped_async] Error adding span metadata: {e}"
            )
        finally:
            return chunk


class TracedAnthropicSyncContextManager:
    def __init__(
        self,
        tracer: Tracer,
        context_manager,
        client: AnthropicClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.model_name = model_name

    def __enter__(self):
        self.stream = self.context_manager.__enter__()
        return TracedAnthropicGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.context_manager.__exit__(exc_type, exc_val, exc_tb)


class TracedAnthropicAsyncContextManager:
    def __init__(
        self,
        tracer: Tracer,
        context_manager,
        client: AnthropicAsyncClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.model_name = model_name

    async def __aenter__(self):
        self.stream = await self.context_manager.__aenter__()
        return TracedAnthropicAsyncGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.context_manager.__aexit__(exc_type, exc_val, exc_tb)


def wrap_anthropic_client(tracer: Tracer, client: TClient) -> TClient:
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
                return TracedAnthropicGenerator(
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
                            f"[anthropic wrapped] Error adding span metadata: {e}"
                        )

                    response = function(*args, **kwargs)

                    try:
                        if isinstance(response, AnthropicMessage):
                            output, usage_data = _format_anthropic_output(response)
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
                                ) = _extract_anthropic_tokens(usage_data)
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
                            f"[anthropic wrapped] Error adding span metadata: {e}"
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
                return TracedAnthropicAsyncGenerator(
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
                            f"[anthropic wrapped_async] Error adding span metadata: {e}"
                        )

                    response = await function(*args, **kwargs)

                    try:
                        if isinstance(response, AnthropicMessage):
                            output, usage_data = _format_anthropic_output(response)
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
                                ) = _extract_anthropic_tokens(usage_data)
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
                            f"[anthropic wrapped_async] Error adding span metadata: {e}"
                        )
                    finally:
                        return response

        return wrapper

    def wrapped_sync_context_manager(function, span_name: str):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            span = tracer.get_tracer().start_span(
                span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
            )
            tracer.add_agent_attributes_to_span(span)
            set_span_attribute(
                span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
            )
            model_name = kwargs.get("model", "")
            set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)

            original_context_manager = function(*args, **kwargs)
            return TracedAnthropicSyncContextManager(
                tracer, original_context_manager, client, span, model_name
            )

        return wrapper

    def wrapped_async_context_manager(function, span_name: str):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            span = tracer.get_tracer().start_span(
                span_name, attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
            )
            tracer.add_agent_attributes_to_span(span)
            set_span_attribute(
                span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
            )
            model_name = kwargs.get("model", "")
            set_span_attribute(span, AttributeKeys.GEN_AI_REQUEST_MODEL, model_name)

            original_context_manager = function(*args, **kwargs)
            return TracedAnthropicAsyncContextManager(
                tracer, original_context_manager, client, span, model_name
            )

        return wrapper

    span_name = "ANTHROPIC_API_CALL"
    if anthropic_Anthropic is not None and isinstance(client, anthropic_Anthropic):
        setattr(client.messages, "create", wrapped(client.messages.create, span_name))
        setattr(
            client.messages,
            "stream",
            wrapped_sync_context_manager(client.messages.stream, span_name),
        )
    elif anthropic_AsyncAnthropic is not None and isinstance(
        client, anthropic_AsyncAnthropic
    ):
        setattr(
            client.messages,
            "create",
            wrapped_async(client.messages.create, span_name),
        )
        setattr(
            client.messages,
            "stream",
            wrapped_async_context_manager(client.messages.stream, span_name),
        )

    return client
