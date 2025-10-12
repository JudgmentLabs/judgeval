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

from judgeval.tracer.llm.llm_groq.config import (
    groq_Groq,
    groq_AsyncGroq,
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
GroqClientType = Union[groq_Groq, groq_AsyncGroq]


# Usage protocols
@runtime_checkable
class GroqPromptTokensDetails(Protocol):
    cached_tokens: Optional[int]


@runtime_checkable
class GroqUsage(Protocol):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    prompt_tokens_details: Optional[GroqPromptTokensDetails]


# Message protocols
@runtime_checkable
class GroqMessage(Protocol):
    content: Optional[str]
    role: str


@runtime_checkable
class GroqChoice(Protocol):
    index: int
    message: GroqMessage
    finish_reason: Optional[str]


@runtime_checkable
class GroqChatCompletion(Protocol):
    id: str
    object: str
    created: int
    model: str
    choices: Sequence[GroqChoice]
    usage: Optional[GroqUsage]


# Stream protocols
@runtime_checkable
class GroqStreamDelta(Protocol):
    content: Optional[str]


@runtime_checkable
class GroqStreamChoice(Protocol):
    index: int
    delta: GroqStreamDelta


@runtime_checkable
class GroqStreamChunk(Protocol):
    choices: Sequence[GroqStreamChoice]
    usage: Optional[GroqUsage]


# Client protocols
@runtime_checkable
class GroqClient(Protocol):
    pass


@runtime_checkable
class GroqAsyncClient(Protocol):
    pass


# Union types
GroqResponseType = GroqChatCompletion
GroqStreamType = Union[Iterator[GroqStreamChunk], AsyncIterator[GroqStreamChunk]]


def _extract_groq_content(chunk: GroqStreamChunk) -> str:
    if chunk.choices and len(chunk.choices) > 0:
        delta_content = chunk.choices[0].delta.content
        if delta_content:
            return delta_content
    return ""


def _extract_groq_tokens(usage_data: GroqUsage) -> Tuple[int, int, int, int]:
    prompt_tokens = usage_data.prompt_tokens or 0
    completion_tokens = usage_data.completion_tokens or 0
    cache_read_input_tokens = 0
    if (
        hasattr(usage_data, "prompt_tokens_details")
        and usage_data.prompt_tokens_details
        and hasattr(usage_data.prompt_tokens_details, "cached_tokens")
        and usage_data.prompt_tokens_details.cached_tokens is not None
    ):
        cache_read_input_tokens = usage_data.prompt_tokens_details.cached_tokens
    cache_creation_input_tokens = 0  # Groq doesn't have cache creation tokens
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _format_groq_output(
    response: GroqChatCompletion,
) -> Tuple[Optional[Union[str, list[dict[str, Any]]]], Optional[GroqUsage]]:
    message_content: Optional[Union[str, list[dict[str, Any]]]] = None
    usage_data: Optional[GroqUsage] = None

    try:
        if isinstance(response, GroqChatCompletion):
            usage_data = response.usage
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    # Return structured data for consistency with other providers
                    message_content = [{"type": "text", "text": str(content)}]
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


class TracedGroqGenerator:
    def __init__(
        self,
        tracer: Tracer,
        generator: Iterator[GroqStreamChunk],
        client: GroqClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.generator = generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __iter__(self) -> Iterator[GroqStreamChunk]:
        return self

    def __next__(self) -> GroqStreamChunk:
        try:
            chunk = next(self.generator)
            content = _extract_groq_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_groq_tokens(chunk.usage)
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


class TracedGroqAsyncGenerator:
    def __init__(
        self,
        tracer: Tracer,
        async_generator: AsyncIterator[GroqStreamChunk],
        client: GroqClientType,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.async_generator = async_generator
        self.client = client
        self.span = span
        self.model_name = model_name
        self.accumulated_content = ""

    def __aiter__(self) -> AsyncIterator[GroqStreamChunk]:
        return self

    async def __anext__(self) -> GroqStreamChunk:
        try:
            chunk = await self.async_generator.__anext__()
            content = _extract_groq_content(chunk)
            if content:
                self.accumulated_content += content
            if chunk.usage:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_groq_tokens(chunk.usage)
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


def wrap_groq_client(tracer: Tracer, client: GroqClientType) -> GroqClientType:
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
                return TracedGroqGenerator(
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
                        # Add groq/ prefix for server-side cost calculation
                        prefixed_model_name = f"groq/{model_name}" if model_name else ""
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_REQUEST_MODEL,
                            prefixed_model_name,
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[groq wrapped] Error adding span metadata: {e}"
                        )

                    response = function(*args, **kwargs)

                    try:
                        if isinstance(response, GroqChatCompletion):
                            output, usage_data = _format_groq_output(response)
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
                                ) = _extract_groq_tokens(usage_data)
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
                            # Add groq/ prefix to response model for server-side cost calculation
                            response_model = getattr(response, "model", model_name)
                            prefixed_response_model = (
                                f"groq/{response_model}" if response_model else ""
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                prefixed_response_model,
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[groq wrapped] Error adding span metadata: {e}"
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
                # Add groq/ prefix for server-side cost calculation
                prefixed_model_name = f"groq/{model_name}" if model_name else ""
                set_span_attribute(
                    span, AttributeKeys.GEN_AI_REQUEST_MODEL, prefixed_model_name
                )
                stream_response = await function(*args, **kwargs)
                return TracedGroqAsyncGenerator(
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
                        # Add groq/ prefix for server-side cost calculation
                        prefixed_model_name = f"groq/{model_name}" if model_name else ""
                        set_span_attribute(
                            span,
                            AttributeKeys.GEN_AI_REQUEST_MODEL,
                            prefixed_model_name,
                        )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[groq wrapped_async] Error adding span metadata: {e}"
                        )

                    response = await function(*args, **kwargs)

                    try:
                        if isinstance(response, GroqChatCompletion):
                            output, usage_data = _format_groq_output(response)
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
                                ) = _extract_groq_tokens(usage_data)
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
                            # Add groq/ prefix to response model for server-side cost calculation
                            response_model = getattr(response, "model", model_name)
                            prefixed_response_model = (
                                f"groq/{response_model}" if response_model else ""
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                prefixed_response_model,
                            )
                    except Exception as e:
                        judgeval_logger.error(
                            f"[groq wrapped_async] Error adding span metadata: {e}"
                        )
                    finally:
                        return response

        return wrapper

    span_name = "GROQ_API_CALL"
    if groq_Groq is not None and isinstance(client, groq_Groq):
        # Type narrowing for mypy
        groq_client = client  # type: ignore[assignment]
        setattr(
            groq_client.chat.completions,
            "create",
            wrapped(groq_client.chat.completions.create, span_name),
        )
    elif groq_AsyncGroq is not None and isinstance(client, groq_AsyncGroq):
        # Type narrowing for mypy
        async_groq_client = client  # type: ignore[assignment]
        setattr(
            async_groq_client.chat.completions,
            "create",
            wrapped_async(async_groq_client.chat.completions.create, span_name),
        )

    return client
