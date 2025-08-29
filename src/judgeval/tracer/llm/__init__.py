from __future__ import annotations
import functools
from typing import (
    Tuple,
    Optional,
    Any,
    TYPE_CHECKING,
    Union,
    AsyncGenerator,
    Generator,
    Iterator,
    AsyncIterator,
)
from functools import wraps
from enum import Enum
from judgeval.data.trace import TraceUsage
from judgeval.logger import judgeval_logger
from litellm.cost_calculator import cost_per_token as _original_cost_per_token
from opentelemetry.trace import Span

from judgeval.tracer.llm.providers import (
    HAS_OPENAI,
    HAS_TOGETHER,
    HAS_ANTHROPIC,
    HAS_GOOGLE_GENAI,
    HAS_GROQ,
    ApiClient,
)
from judgeval.tracer.managers import sync_span_context, async_span_context
from judgeval.tracer.keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.tracer.utils import set_span_attribute

if TYPE_CHECKING:
    from judgeval.tracer import Tracer


class ProviderType(Enum):
    """Enum for different LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    GOOGLE = "google"
    GROQ = "groq"
    DEFAULT = "default"


@wraps(_original_cost_per_token)
def cost_per_token(
    *args: Any, **kwargs: Any
) -> Tuple[Optional[float], Optional[float]]:
    try:
        prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar = (
            _original_cost_per_token(*args, **kwargs)
        )
        if (
            prompt_tokens_cost_usd_dollar == 0
            and completion_tokens_cost_usd_dollar == 0
        ):
            judgeval_logger.warning("LiteLLM returned a total of 0 for cost per token")
        return prompt_tokens_cost_usd_dollar, completion_tokens_cost_usd_dollar
    except Exception as e:
        judgeval_logger.warning(f"Error calculating cost per token: {e}")
        return None, None


def _detect_provider(client: ApiClient) -> ProviderType:
    """Detect the provider type of the client once to avoid repeated isinstance checks."""
    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        if isinstance(client, (openai_OpenAI, openai_AsyncOpenAI)):
            return ProviderType.OPENAI

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        if isinstance(client, (anthropic_Anthropic, anthropic_AsyncAnthropic)):
            return ProviderType.ANTHROPIC

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        if isinstance(client, (together_Together, together_AsyncTogether)):
            return ProviderType.TOGETHER

    if HAS_GOOGLE_GENAI:
        from judgeval.tracer.llm.providers import (
            google_genai_Client,
            google_genai_AsyncClient,
        )

        assert google_genai_Client is not None, "Google GenAI client not found"
        assert google_genai_AsyncClient is not None, (
            "Google GenAI async client not found"
        )
        if isinstance(client, (google_genai_Client, google_genai_AsyncClient)):
            return ProviderType.GOOGLE

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        if isinstance(client, (groq_Groq, groq_AsyncGroq)):
            return ProviderType.GROQ

    return ProviderType.DEFAULT


# Provider-specific content extraction handlers
def _extract_openai_content(chunk) -> str:
    """Extract content from OpenAI streaming chunk."""
    if (
        hasattr(chunk, "choices")
        and chunk.choices
        and hasattr(chunk.choices[0], "delta")
    ):
        delta_content = getattr(chunk.choices[0].delta, "content", None)
        if delta_content:
            return delta_content
    return ""


def _extract_anthropic_content(chunk) -> str:
    """Extract content from Anthropic streaming chunk."""
    if hasattr(chunk, "type") and chunk.type == "content_block_delta":
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            return chunk.delta.text or ""
    elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
        return chunk.delta.text or ""
    elif hasattr(chunk, "text"):
        return chunk.text or ""
    return ""


def _extract_together_content(chunk) -> str:
    """Extract content from Together streaming chunk."""
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            return choice.delta.content or ""
    return ""


def _extract_groq_content(chunk) -> str:
    """Extract content from Groq streaming chunk."""
    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            return choice.delta.content or ""
    return ""


# Provider-specific chunk usage extraction handlers
def _extract_openai_chunk_usage(chunk) -> Any:
    """Extract usage data from OpenAI streaming chunk."""
    if hasattr(chunk, "usage") and chunk.usage:
        return chunk.usage
    return None


def _extract_anthropic_chunk_usage(chunk) -> Any:
    """Extract usage data from Anthropic streaming chunk."""
    if hasattr(chunk, "type"):
        if chunk.type == "message_start":
            if hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                return chunk.message.usage
        elif chunk.type == "message_delta":
            if hasattr(chunk, "usage"):
                return chunk.usage
        elif chunk.type == "message_stop":
            if hasattr(chunk, "usage"):
                return chunk.usage
    return None


def _extract_together_chunk_usage(chunk) -> Any:
    """Extract usage data from Together streaming chunk."""
    if hasattr(chunk, "usage") and chunk.usage:
        return chunk.usage
    return None


def _extract_groq_chunk_usage(chunk) -> Any:
    """Extract usage data from Groq streaming chunk."""
    # Groq provides usage data in the last chunk when stream_options={"include_usage": True} is used
    if hasattr(chunk, "usage") and chunk.usage:
        return chunk.usage
    return None


# Provider-specific token extraction handlers
def _extract_openai_tokens(usage_data) -> tuple[int, int, int, int]:
    """Extract token counts from OpenAI usage data."""
    prompt_tokens = (
        usage_data.prompt_tokens
        if hasattr(usage_data, "prompt_tokens") and usage_data.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        usage_data.completion_tokens
        if hasattr(usage_data, "completion_tokens")
        and usage_data.completion_tokens is not None
        else 0
    )
    return prompt_tokens, completion_tokens, 0, 0


def _extract_anthropic_tokens(usage_data) -> tuple[int, int, int, int]:
    """Extract token counts from Anthropic usage data."""
    prompt_tokens = (
        usage_data.input_tokens
        if hasattr(usage_data, "input_tokens") and usage_data.input_tokens is not None
        else 0
    )
    completion_tokens = (
        usage_data.output_tokens
        if hasattr(usage_data, "output_tokens") and usage_data.output_tokens is not None
        else 0
    )
    cache_read_input_tokens = (
        usage_data.cache_read_input_tokens
        if hasattr(usage_data, "cache_read_input_tokens")
        and usage_data.cache_read_input_tokens is not None
        else 0
    )
    cache_creation_input_tokens = (
        usage_data.cache_creation_input_tokens
        if hasattr(usage_data, "cache_creation_input_tokens")
        and usage_data.cache_creation_input_tokens is not None
        else 0
    )
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _extract_together_tokens(usage_data) -> tuple[int, int, int, int]:
    """Extract token counts from Together usage data."""
    prompt_tokens = (
        usage_data.prompt_tokens
        if hasattr(usage_data, "prompt_tokens") and usage_data.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        usage_data.completion_tokens
        if hasattr(usage_data, "completion_tokens")
        and usage_data.completion_tokens is not None
        else 0
    )
    return prompt_tokens, completion_tokens, 0, 0


def _extract_groq_tokens(usage_data) -> tuple[int, int, int, int]:
    """Extract token counts from Groq usage data."""
    prompt_tokens = (
        usage_data.prompt_tokens
        if hasattr(usage_data, "prompt_tokens") and usage_data.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        usage_data.completion_tokens
        if hasattr(usage_data, "completion_tokens")
        and usage_data.completion_tokens is not None
        else 0
    )
    # Extract cached tokens from prompt_tokens_details.cached_tokens
    cache_read_input_tokens = 0
    if (
        hasattr(usage_data, "prompt_tokens_details")
        and usage_data.prompt_tokens_details
    ):
        if (
            hasattr(usage_data.prompt_tokens_details, "cached_tokens")
            and usage_data.prompt_tokens_details.cached_tokens is not None
        ):
            cache_read_input_tokens = usage_data.prompt_tokens_details.cached_tokens

    return prompt_tokens, completion_tokens, cache_read_input_tokens, 0


# Provider-specific output formatting handlers
def _format_openai_output(response: Any) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from OpenAI response."""
    from judgeval.tracer.llm.providers import (
        openai_ChatCompletion,
        openai_Response,
        openai_ParsedChatCompletion,
    )

    model_name = None
    message_content = None
    prompt_tokens = 0
    completion_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0

    if openai_ChatCompletion and isinstance(response, openai_ChatCompletion):
        model_name = response.model or ""
        prompt_tokens = (
            response.usage.prompt_tokens
            if response.usage and response.usage.prompt_tokens is not None
            else 0
        )
        completion_tokens = (
            response.usage.completion_tokens
            if response.usage and response.usage.completion_tokens is not None
            else 0
        )
        cache_read_input_tokens = (
            response.usage.prompt_tokens_details.cached_tokens
            if response.usage
            and response.usage.prompt_tokens_details
            and response.usage.prompt_tokens_details.cached_tokens is not None
            else 0
        )

        if openai_ParsedChatCompletion and isinstance(
            response, openai_ParsedChatCompletion
        ):
            message_content = response.choices[0].message.parsed
        else:
            message_content = response.choices[0].message.content
    elif openai_Response and isinstance(response, openai_Response):
        model_name = response.model or ""
        prompt_tokens = (
            response.usage.input_tokens
            if response.usage and response.usage.input_tokens is not None
            else 0
        )
        completion_tokens = (
            response.usage.output_tokens
            if response.usage and response.usage.output_tokens is not None
            else 0
        )
        cache_read_input_tokens = (
            response.usage.input_tokens_details.cached_tokens
            if response.usage
            and response.usage.input_tokens_details
            and response.usage.input_tokens_details.cached_tokens is not None
            else 0
        )
        output0 = response.output[0]
        if (
            hasattr(output0, "content")
            and output0.content
            and hasattr(output0.content, "__iter__")
        ):
            message_content = "".join(
                seg.text for seg in output0.content if hasattr(seg, "text") and seg.text
            )

    if model_name:
        return message_content, _create_usage(
            model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        )

    return None, None


def _format_anthropic_output(
    response: Any,
) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from Anthropic response."""
    model_name = getattr(response, "model", "") or ""
    usage = getattr(response, "usage", None)
    prompt_tokens = (
        usage.input_tokens
        if usage and hasattr(usage, "input_tokens") and usage.input_tokens is not None
        else 0
    )
    completion_tokens = (
        usage.output_tokens
        if usage and hasattr(usage, "output_tokens") and usage.output_tokens is not None
        else 0
    )
    cache_read_input_tokens = (
        usage.cache_read_input_tokens
        if usage
        and hasattr(usage, "cache_read_input_tokens")
        and usage.cache_read_input_tokens is not None
        else 0
    )
    cache_creation_input_tokens = (
        usage.cache_creation_input_tokens
        if usage
        and hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens is not None
        else 0
    )
    message_content = response.content[0].text if hasattr(response, "content") else None

    if model_name:
        return message_content, _create_usage(
            model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        )

    return None, None


def _format_together_output(
    response: Any,
) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from Together response."""
    model_name = (response.model or "") if hasattr(response, "model") else ""
    prompt_tokens = (
        response.usage.prompt_tokens
        if hasattr(response.usage, "prompt_tokens")
        and response.usage.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        response.usage.completion_tokens
        if hasattr(response.usage, "completion_tokens")
        and response.usage.completion_tokens is not None
        else 0
    )
    message_content = (
        response.choices[0].message.content if hasattr(response, "choices") else None
    )

    if model_name:
        model_name = "together_ai/" + model_name
        return message_content, _create_usage(
            model_name,
            prompt_tokens,
            completion_tokens,
            0,
            0,
        )

    return None, None


def _format_google_output(response: Any) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from Google GenAI response."""
    model_name = getattr(response, "model_version", "") or ""
    usage_metadata = getattr(response, "usage_metadata", None)
    prompt_tokens = (
        usage_metadata.prompt_token_count
        if usage_metadata
        and hasattr(usage_metadata, "prompt_token_count")
        and usage_metadata.prompt_token_count is not None
        else 0
    )
    completion_tokens = (
        usage_metadata.candidates_token_count
        if usage_metadata
        and hasattr(usage_metadata, "candidates_token_count")
        and usage_metadata.candidates_token_count is not None
        else 0
    )
    message_content = (
        response.candidates[0].content.parts[0].text
        if hasattr(response, "candidates")
        else None
    )

    cache_read_input_tokens = 0
    if usage_metadata and hasattr(usage_metadata, "cached_content_token_count"):
        cache_read_input_tokens = usage_metadata.cached_content_token_count or 0

    if model_name:
        return message_content, _create_usage(
            model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            0,
        )

    return None, None


def _format_groq_output(response: Any) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from Groq response."""
    model_name = (response.model or "") if hasattr(response, "model") else ""
    prompt_tokens = (
        response.usage.prompt_tokens
        if hasattr(response.usage, "prompt_tokens")
        and response.usage.prompt_tokens is not None
        else 0
    )
    completion_tokens = (
        response.usage.completion_tokens
        if hasattr(response.usage, "completion_tokens")
        and response.usage.completion_tokens is not None
        else 0
    )
    # Extract cached tokens from prompt_tokens_details.cached_tokens
    cache_read_input_tokens = 0
    if (
        hasattr(response, "usage")
        and response.usage
        and hasattr(response.usage, "prompt_tokens_details")
        and response.usage.prompt_tokens_details
    ):
        if (
            hasattr(response.usage.prompt_tokens_details, "cached_tokens")
            and response.usage.prompt_tokens_details.cached_tokens is not None
        ):
            cache_read_input_tokens = response.usage.prompt_tokens_details.cached_tokens

    message_content = (
        response.choices[0].message.content if hasattr(response, "choices") else None
    )

    if model_name:
        model_name = "groq/" + model_name
        return message_content, _create_usage(
            model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            0,
        )

    return None, None


class _TracedGeneratorBase:
    """Base class with common logic for parsing stream chunks."""

    __slots__ = (
        "tracer",
        "client",
        "span",
        "accumulated_content",
        "model_name",
        "provider_type",
    )

    tracer: Tracer
    client: ApiClient
    span: Span
    accumulated_content: str
    model_name: str
    provider_type: ProviderType

    def __init__(self, tracer: Tracer, client: ApiClient, span: Span, model_name: str):
        """Initialize the base traced generator.

        Args:
            tracer: The tracer instance
            client: The API client
            span: The OpenTelemetry span
            model_name: The model name (empty string default allows fallback to usage_data.model)
        """
        self.tracer = tracer
        self.client = client
        self.span = span
        self.accumulated_content = ""
        self.model_name = model_name
        self.provider_type = _detect_provider(client)

    def _extract_content(self, chunk) -> str:
        """Extract content from streaming chunk based on provider."""
        if self.provider_type == ProviderType.OPENAI:
            return _extract_openai_content(chunk)
        elif self.provider_type == ProviderType.ANTHROPIC:
            return _extract_anthropic_content(chunk)
        elif self.provider_type == ProviderType.TOGETHER:
            return _extract_together_content(chunk)
        elif self.provider_type == ProviderType.GROQ:
            return _extract_groq_content(chunk)
        else:
            # Default case - assume OpenAI-compatible for unknown providers
            return _extract_openai_content(chunk)

    def _process_chunk_usage(self, chunk):
        """Process usage data from streaming chunks based on provider."""
        usage_data = _extract_chunk_usage(self.client, chunk)
        if usage_data:
            _process_usage_data(
                self.span, usage_data, self.tracer, self.client, self.model_name
            )

    def __del__(self):
        """
        Fallback cleanup for unclosed spans. This is a safety mechanism only - spans
        should normally be finalized in StopIteration/StopAsyncIteration handlers.

        Note: __del__ is not guaranteed to be called in all situations (e.g., reference
        cycles, program exit), so this should not be relied upon as the primary cleanup
        mechanism. The primary finalization happens in the iterator protocol methods.
        """
        if self.span:
            try:
                self._finalize_span()
            except Exception as e:
                judgeval_logger.warning(
                    f"Error during span finalization in __del__: {e}"
                )

    def _finalize_span(self):
        """Finalize the span by setting completion content and ending it."""
        if self.span:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            self.span = None


class TracedGenerator(_TracedGeneratorBase):
    """Generator wrapper that adds OpenTelemetry tracing without consuming the stream."""

    __slots__ = ("generator",)

    generator: Union[Generator[Any, None, None], Iterator[Any]]

    def __init__(
        self,
        tracer: Tracer,
        generator: Union[Generator[Any, None, None], Iterator[Any]],
        client: ApiClient,
        span: Span,
        model_name: str,
    ):
        super().__init__(tracer, client, span, model_name)
        self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.generator)

            content = self._extract_content(chunk)
            if content:
                self.accumulated_content += content
            self._process_chunk_usage(chunk)

            return chunk

        except StopIteration:
            self._finalize_span()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


class TracedAsyncGenerator(_TracedGeneratorBase):
    """Async generator wrapper that adds OpenTelemetry tracing without consuming the stream."""

    __slots__ = ("async_generator",)

    async_generator: Union[AsyncGenerator[Any, None], AsyncIterator[Any]]

    def __init__(
        self,
        tracer: Tracer,
        async_generator: Union[AsyncGenerator[Any, None], AsyncIterator[Any]],
        client: ApiClient,
        span: Span,
        model_name: str,
    ):
        super().__init__(tracer, client, span, model_name)
        self.async_generator = async_generator

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_generator.__anext__()

            content = self._extract_content(chunk)
            if content:
                self.accumulated_content += content

            self._process_chunk_usage(chunk)

            return chunk

        except StopAsyncIteration:
            self._finalize_span()
            raise
        except Exception as e:
            if self.span:
                self.span.record_exception(e)
                self.span.end()
            raise


class TracedSyncContextManager:
    """Sync context manager wrapper for streaming methods."""

    def __init__(
        self,
        tracer: Tracer,
        context_manager: Any,
        client: ApiClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.stream: Optional[Any] = None
        self.model_name = model_name

    def __enter__(self):
        self.stream = self.context_manager.__enter__()
        return TracedGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.context_manager.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        """Cleanup span if not properly closed."""
        if self.span:
            try:
                self.span.end()
            except Exception:
                pass


class TracedAsyncContextManager:
    """Async context manager wrapper for streaming methods."""

    def __init__(
        self,
        tracer: Tracer,
        context_manager: Any,
        client: ApiClient,
        span: Span,
        model_name: str,
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.stream: Optional[Any] = None
        self.model_name = model_name

    async def __aenter__(self):
        self.stream = await self.context_manager.__aenter__()
        return TracedAsyncGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.context_manager.__aexit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        """Cleanup span if not properly closed."""
        if self.span:
            try:
                self.span.end()
            except Exception:
                pass


def _extract_chunk_usage(client: ApiClient, chunk) -> Any:
    """Extract usage data from streaming chunks based on provider."""
    provider_type = _detect_provider(client)

    if provider_type == ProviderType.OPENAI:
        return _extract_openai_chunk_usage(chunk)
    elif provider_type == ProviderType.ANTHROPIC:
        return _extract_anthropic_chunk_usage(chunk)
    elif provider_type == ProviderType.TOGETHER:
        return _extract_together_chunk_usage(chunk)
    elif provider_type == ProviderType.GROQ:
        return _extract_groq_chunk_usage(chunk)
    else:
        # Default case - assume OpenAI-compatible for unknown providers
        return _extract_openai_chunk_usage(chunk)


def _extract_usage_tokens(client: ApiClient, usage_data) -> tuple[int, int, int, int]:
    """Extract token counts from usage data based on provider."""
    provider_type = _detect_provider(client)

    if provider_type == ProviderType.OPENAI:
        return _extract_openai_tokens(usage_data)
    elif provider_type == ProviderType.ANTHROPIC:
        return _extract_anthropic_tokens(usage_data)
    elif provider_type == ProviderType.TOGETHER:
        return _extract_together_tokens(usage_data)
    elif provider_type == ProviderType.GROQ:
        return _extract_groq_tokens(usage_data)
    else:
        # Default case - assume OpenAI-compatible for unknown providers
        return _extract_openai_tokens(usage_data)


def _process_usage_data(
    span, usage_data, tracer: Tracer, client: ApiClient, model_name: str
):
    """Process usage data and set span attributes."""
    (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    ) = _extract_usage_tokens(client, usage_data)

    if prompt_tokens or completion_tokens:
        final_model_name = getattr(usage_data, "model", None) or model_name

        # Add provider prefixes for cost calculation
        provider_type = _detect_provider(client)
        if (
            provider_type == ProviderType.TOGETHER
            and final_model_name
            and not final_model_name.startswith("together_ai/")
        ):
            final_model_name = "together_ai/" + final_model_name
        elif (
            provider_type == ProviderType.GROQ
            and final_model_name
            and not final_model_name.startswith("groq/")
        ):
            final_model_name = "groq/" + final_model_name

        usage = _create_usage(
            final_model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        )
        _set_usage_attributes(span, usage, tracer)


def _set_usage_attributes(span, usage: TraceUsage, tracer: Tracer):
    """Set usage attributes on the span for non-streaming responses."""

    set_span_attribute(span, AttributeKeys.GEN_AI_RESPONSE_MODEL, usage.model_name)
    set_span_attribute(
        span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, usage.prompt_tokens
    )
    set_span_attribute(
        span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, usage.completion_tokens
    )
    set_span_attribute(
        span, AttributeKeys.GEN_AI_USAGE_COMPLETION_TOKENS, usage.completion_tokens
    )
    set_span_attribute(
        span, AttributeKeys.GEN_AI_USAGE_TOTAL_COST, usage.total_cost_usd
    )
    tracer.add_cost_to_current_context(usage.total_cost_usd)


def wrap_provider(tracer: Tracer, client: ApiClient) -> ApiClient:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, Google GenAI, and Groq clients.
    """

    def wrapped(function, span_name):
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

                # Add provider prefix for Groq clients
                if HAS_GROQ:
                    from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

                    if (
                        isinstance(client, (groq_Groq, groq_AsyncGroq))
                        and model_name
                        and not model_name.startswith("groq/")
                    ):
                        model_name = "groq/" + model_name

                response = function(*args, **kwargs)
                return TracedGenerator(tracer, response, client, span, model_name)
            else:
                with sync_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    tracer.add_agent_attributes_to_span(span)
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                    )
                    try:
                        response = function(*args, **kwargs)
                        output, usage = _format_output_data(client, response)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_COMPLETION, output
                        )
                        if usage:
                            _set_usage_attributes(span, usage, tracer)
                        return response
                    except Exception as e:
                        span.record_exception(e)
                        raise

        return wrapper

    def wrapped_async(function, span_name):
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

                # Add provider prefix for Groq clients
                if HAS_GROQ:
                    from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

                    if (
                        isinstance(client, (groq_Groq, groq_AsyncGroq))
                        and model_name
                        and not model_name.startswith("groq/")
                    ):
                        model_name = "groq/" + model_name

                response = await function(*args, **kwargs)
                return TracedAsyncGenerator(tracer, response, client, span, model_name)
            else:
                async with async_span_context(
                    tracer, span_name, {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
                ) as span:
                    tracer.add_agent_attributes_to_span(span)
                    set_span_attribute(
                        span, AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
                    )
                    try:
                        response = await function(*args, **kwargs)
                        output, usage = _format_output_data(client, response)
                        set_span_attribute(
                            span, AttributeKeys.GEN_AI_COMPLETION, output
                        )
                        if usage:
                            _set_usage_attributes(span, usage, tracer)
                        return response
                    except Exception as e:
                        span.record_exception(e)
                        raise

        return wrapper

    def wrapped_sync_context_manager(function, span_name):
        """Special wrapper for sync context manager methods."""

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

            # Add provider prefix for Groq clients
            if HAS_GROQ:
                from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

                if (
                    isinstance(client, (groq_Groq, groq_AsyncGroq))
                    and model_name
                    and not model_name.startswith("groq/")
                ):
                    model_name = "groq/" + model_name

            original_context_manager = function(*args, **kwargs)
            return TracedSyncContextManager(
                tracer, original_context_manager, client, span, model_name
            )

        return wrapper

    def wrapped_async_context_manager(function, span_name):
        """Special wrapper for async context manager methods."""

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

            # Add provider prefix for Groq clients
            if HAS_GROQ:
                from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

                if (
                    isinstance(client, (groq_Groq, groq_AsyncGroq))
                    and model_name
                    and not model_name.startswith("groq/")
                ):
                    model_name = "groq/" + model_name

            original_context_manager = function(*args, **kwargs)
            return TracedAsyncContextManager(
                tracer, original_context_manager, client, span, model_name
            )

        return wrapper

    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        span_name = "OPENAI_API_CALL"
        if isinstance(client, openai_OpenAI):
            setattr(
                client.chat.completions,
                "create",
                wrapped(client.chat.completions.create, span_name),
            )
            setattr(
                client.responses, "create", wrapped(client.responses.create, span_name)
            )
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

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        span_name = "TOGETHER_API_CALL"
        if isinstance(client, together_Together):
            setattr(
                client.chat.completions,
                "create",
                wrapped(client.chat.completions.create, span_name),
            )
        elif isinstance(client, together_AsyncTogether):
            setattr(
                client.chat.completions,
                "create",
                wrapped_async(client.chat.completions.create, span_name),
            )

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        span_name = "ANTHROPIC_API_CALL"
        if isinstance(client, anthropic_Anthropic):
            setattr(
                client.messages, "create", wrapped(client.messages.create, span_name)
            )
            setattr(
                client.messages,
                "stream",
                wrapped_sync_context_manager(client.messages.stream, span_name),
            )
        elif isinstance(client, anthropic_AsyncAnthropic):
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

    if HAS_GOOGLE_GENAI:
        from judgeval.tracer.llm.providers import (
            google_genai_Client,
            google_genai_AsyncClient,
        )

        assert google_genai_Client is not None, "Google GenAI client not found"
        assert google_genai_AsyncClient is not None, (
            "Google GenAI async client not found"
        )
        span_name = "GOOGLE_API_CALL"
        if isinstance(client, google_genai_Client):
            setattr(
                client.models,
                "generate_content",
                wrapped(client.models.generate_content, span_name),
            )
        elif isinstance(client, google_genai_AsyncClient):
            setattr(
                client.models,
                "generate_content",
                wrapped_async(client.models.generate_content, span_name),
            )

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        span_name = "GROQ_API_CALL"
        if isinstance(client, groq_Groq):
            setattr(
                client.chat.completions,
                "create",
                wrapped(client.chat.completions.create, span_name),
            )
        elif isinstance(client, groq_AsyncGroq):
            setattr(
                client.chat.completions,
                "create",
                wrapped_async(client.chat.completions.create, span_name),
            )

    return client


def _format_output_data(
    client: ApiClient, response: Any
) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format output data from LLM response based on provider."""
    provider_type = _detect_provider(client)

    if provider_type == ProviderType.OPENAI:
        return _format_openai_output(response)
    elif provider_type == ProviderType.ANTHROPIC:
        return _format_anthropic_output(response)
    elif provider_type == ProviderType.TOGETHER:
        return _format_together_output(response)
    elif provider_type == ProviderType.GOOGLE:
        return _format_google_output(response)
    elif provider_type == ProviderType.GROQ:
        return _format_groq_output(response)
    else:
        # Default case - assume OpenAI-compatible for unknown providers
        judgeval_logger.info(
            f"Unknown client type {type(client)}, assuming OpenAI-compatible"
        )
        return _format_openai_output(response)


def _create_usage(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> TraceUsage:
    prompt_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    total_cost_usd = (
        (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    )
    return TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name,
    )
