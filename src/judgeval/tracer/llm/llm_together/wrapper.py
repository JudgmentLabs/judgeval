from __future__ import annotations
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Tuple,
    Union,
    Iterator,
    AsyncIterator,
    Sequence,
    runtime_checkable,
)

from judgeval.tracer.llm.llm_together.config import (
    together_Together,
    together_AsyncTogether,
)
from judgeval.tracer.managers import sync_span_context, async_span_context
from judgeval.logger import judgeval_logger
from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from opentelemetry.trace import Span

# Keep the original client type for runtime compatibility
TogetherClientType = Union[together_Together, together_AsyncTogether]


# Usage protocols
@runtime_checkable
class TogetherUsage(Protocol):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


# Message protocols
@runtime_checkable
class TogetherMessage(Protocol):
    content: Optional[str]
    role: str


@runtime_checkable
class TogetherChoice(Protocol):
    index: int
    message: TogetherMessage
    finish_reason: Optional[str]


@runtime_checkable
class TogetherChatCompletion(Protocol):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[TogetherChoice]
    usage: Optional[TogetherUsage]


# Stream protocols
@runtime_checkable
class TogetherStreamDelta(Protocol):
    content: Optional[str]


@runtime_checkable
class TogetherStreamChoice(Protocol):
    index: int
    delta: TogetherStreamDelta


@runtime_checkable
class TogetherStreamChunk(Protocol):
    choices: Sequence[TogetherStreamChoice]
    usage: Optional[TogetherUsage]


# Client protocols
@runtime_checkable
class TogetherClient(Protocol):
    pass


@runtime_checkable
class TogetherAsyncClient(Protocol):
    pass


# Union types
TogetherResponseType = TogetherChatCompletion
TogetherStreamType = Union[
    Iterator[TogetherStreamChunk], AsyncIterator[TogetherStreamChunk]
]


def _extract_together_content(chunk: TogetherStreamChunk) -> str:
    if chunk.choices and len(chunk.choices) > 0:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            return delta_content
    return ""


def _extract_together_tokens(usage_data: TogetherUsage) -> Tuple[int, int, int, int]:
    prompt_tokens = usage_data.prompt_tokens or 0
    completion_tokens = usage_data.completion_tokens or 0
    cache_read_input_tokens = 0  # Together doesn't support cache tokens
    cache_creation_input_tokens = 0  # Together doesn't support cache tokens
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _format_together_output(
    response: TogetherChatCompletion,
) -> Tuple[Optional[Union[str, list[dict[str, Any]]]], Optional[TogetherUsage]]:
    message_content: Optional[Union[str, list[dict[str, Any]]]] = None
    usage_data: Optional[TogetherUsage] = None

    try:
        if isinstance(response, TogetherChatCompletion):
            usage_data = response.usage
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    # Return structured data for consistency with other providers
                    message_content = [{"type": "text", "text": str(content)}]
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


class TracedTogetherGenerator:
    def __init__(
        self,
        tracer: Tracer,
        generator: Iterator[TogetherStreamChunk],
        client: TogetherClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.generator = generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __iter__(self) -> Iterator[TogetherStreamChunk]:
        return self

    def __next__(self) -> TogetherStreamChunk:
        try:
            chunk = next(self.generator)
            content = _extract_together_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                (
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    cache_creation,
                ) = _extract_together_tokens(chunk.usage)
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


class TracedTogetherAsyncGenerator:
    def __init__(
        self,
        tracer: Tracer,
        async_generator: AsyncIterator[TogetherStreamChunk],
        client: TogetherClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.async_generator = async_generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __aiter__(self) -> AsyncIterator[TogetherStreamChunk]:
        return self

    async def __anext__(self) -> TogetherStreamChunk:
        try:
            chunk = await self.async_generator.__anext__()
            content = _extract_together_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                (
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    cache_creation,
                ) = _extract_together_tokens(chunk.usage)
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


def wrap_together_client(
    tracer: Tracer, client: TogetherClientType
) -> TogetherClientType:
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
                # Add together_ai/ prefix for server-side cost calculation
                prefixed_model_name = f"together_ai/{model_name}" if model_name else ""
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_REQUEST_MODEL, prefixed_model_name
                )
                stream_response = function(*args, **kwargs)
                return TracedTogetherGenerator(
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
                        # Add together_ai/ prefix for server-side cost calculation
                        prefixed_model_name = (
                            f"together_ai/{model_name}" if model_name else ""
                        )
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_REQUEST_MODEL,
                            prefixed_model_name,
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[together wrapped] Error adding span metadata: {e}"
                        )

                    response = function(*args, **kwargs)

                    try:
                        if isinstance(response, TogetherChatCompletion):
                            output, usage_data = _format_together_output(response)
                            # Serialize structured data to JSON for span attribute
                            if output:
                                if isinstance(output, list):
                                    output_str = safe_serialize(output)
                                else:
                                    output_str = str(output)
                                set_span_attribute(
                                    span, AttributeKeys.GEN_AI_COMPLETION, output_str
                                )
                            if usage_data:
                                (
                                    prompt_tokens,
                                    completion_tokens,
                                    cache_read,
                                    cache_creation,
                                ) = _extract_together_tokens(usage_data)
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
                                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                                    safe_serialize(usage_data),
                                )
                            # Add together_ai/ prefix to response model for server-side cost calculation
                            response_model = getattr(response, "model", model_name)
                            prefixed_response_model = (
                                f"together_ai/{response_model}"
                                if response_model
                                else ""
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                prefixed_response_model,
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[together wrapped] Error adding span metadata: {e}"
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
                # Add together_ai/ prefix for server-side cost calculation
                prefixed_model_name = f"together_ai/{model_name}" if model_name else ""
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_REQUEST_MODEL, prefixed_model_name
                )
                stream_response = await function(*args, **kwargs)
                return TracedTogetherAsyncGenerator(
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
                        # Add together_ai/ prefix for server-side cost calculation
                        prefixed_model_name = (
                            f"together_ai/{model_name}" if model_name else ""
                        )
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_REQUEST_MODEL,
                            prefixed_model_name,
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[together wrapped_async] Error adding span metadata: {e}"
                        )

                    response = await function(*args, **kwargs)

                    try:
                        if isinstance(response, TogetherChatCompletion):
                            output, usage_data = _format_together_output(response)
                            # Serialize structured data to JSON for span attribute
                            if output:
                                if isinstance(output, list):
                                    output_str = safe_serialize(output)
                                else:
                                    output_str = str(output)
                                set_span_attribute(
                                    span, AttributeKeys.GEN_AI_COMPLETION, output_str
                                )
                            if usage_data:
                                (
                                    prompt_tokens,
                                    completion_tokens,
                                    cache_read,
                                    cache_creation,
                                ) = _extract_together_tokens(usage_data)
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
                                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                                    safe_serialize(usage_data),
                                )
                            # Add together_ai/ prefix to response model for server-side cost calculation
                            response_model = getattr(response, "model", model_name)
                            prefixed_response_model = (
                                f"together_ai/{response_model}"
                                if response_model
                                else ""
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                prefixed_response_model,
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[together wrapped_async] Error adding span metadata: {e}"
                        )
                    finally:
                        return response

        return wrapper

    span_name = "TOGETHER_API_CALL"
    if together_Together and isinstance(client, together_Together):
        setattr(
            client.chat.completions,
            "create",
            wrapped(client.chat.completions.create, span_name),
        )
    elif together_AsyncTogether and isinstance(client, together_AsyncTogether):
        setattr(
            client.chat.completions,
            "create",
            wrapped_async(client.chat.completions.create, span_name),
        )

    return client
