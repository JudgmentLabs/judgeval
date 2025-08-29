from __future__ import annotations
import functools
from typing import Tuple, Optional, Any, TYPE_CHECKING
from functools import wraps
from judgeval.data.trace import TraceUsage
from judgeval.logger import judgeval_logger
from litellm.cost_calculator import cost_per_token as _original_cost_per_token

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


class _TracedGeneratorBase:
    """Base class with common logic for parsing stream chunks."""

    def __init__(self, tracer: "Tracer", client: ApiClient, span, model_name: str = ""):
        self.tracer = tracer
        self.client = client
        self.span = span
        self.accumulated_content = ""
        self.model_name = model_name

    def _extract_content(self, chunk) -> str:
        """Extract content from streaming chunk based on provider."""
        if HAS_OPENAI:
            from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

            if isinstance(self.client, (openai_OpenAI, openai_AsyncOpenAI)):
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and hasattr(chunk.choices[0], "delta")
                ):
                    delta_content = getattr(chunk.choices[0].delta, "content", None)
                    if delta_content:
                        return delta_content

        if HAS_ANTHROPIC:
            from judgeval.tracer.llm.providers import (
                anthropic_Anthropic,
                anthropic_AsyncAnthropic,
            )

            if isinstance(self.client, (anthropic_Anthropic, anthropic_AsyncAnthropic)):
                if hasattr(chunk, "type") and chunk.type == "content_block_delta":
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        return chunk.delta.text or ""
                elif hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    return chunk.delta.text or ""
                elif hasattr(chunk, "text"):
                    return chunk.text or ""

        if HAS_TOGETHER:
            from judgeval.tracer.llm.providers import (
                together_Together,
                together_AsyncTogether,
            )

            if isinstance(self.client, (together_Together, together_AsyncTogether)):
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        return choice.delta.content or ""

        if HAS_GROQ:
            from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

            if isinstance(self.client, (groq_Groq, groq_AsyncGroq)):
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        return choice.delta.content or ""

        return ""

    def _process_chunk_usage(self, chunk):
        """Process usage data from streaming chunks based on provider."""
        usage_data = None

        if HAS_ANTHROPIC:
            from judgeval.tracer.llm.providers import (
                anthropic_Anthropic,
                anthropic_AsyncAnthropic,
            )

            if isinstance(self.client, (anthropic_Anthropic, anthropic_AsyncAnthropic)):
                if hasattr(chunk, "type"):
                    if chunk.type == "message_start":
                        if hasattr(chunk, "message") and hasattr(
                            chunk.message, "usage"
                        ):
                            usage_data = chunk.message.usage
                    elif chunk.type == "message_delta":
                        if hasattr(chunk, "usage"):
                            usage_data = chunk.usage
                    elif chunk.type == "message_stop":
                        if hasattr(chunk, "usage"):
                            usage_data = chunk.usage

        if not usage_data:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
            elif hasattr(chunk, "message") and hasattr(chunk.message, "usage"):
                usage_data = chunk.message.usage

        if usage_data:
            _process_usage_data(self.span, usage_data, self.tracer, self.model_name)

    def __del__(self):
        """
        Fallback cleanup for unclosed spans. Note: __del__ is not guaranteed to be called
        in all situations (e.g., reference cycles, program exit), so this should not be
        relied upon as the primary cleanup mechanism.
        """
        if hasattr(self, "span") and self.span:
            try:
                self._finalize_span()
            except Exception as e:
                judgeval_logger.warning(
                    f"Error during span finalization in __del__: {e}"
                )

    def _finalize_span(self):
        """Finalize the span by setting completion content and ending it."""
        if hasattr(self, "span") and self.span:
            set_span_attribute(
                self.span, AttributeKeys.GEN_AI_COMPLETION, self.accumulated_content
            )
            self.span.end()
            self.span = None


class TracedGenerator(_TracedGeneratorBase):
    """Generator wrapper that adds OpenTelemetry tracing without consuming the stream."""

    def __init__(
        self, tracer: "Tracer", generator, client: ApiClient, span, model_name: str = ""
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
            self.span.record_exception(e)
            self.span.end()
            raise


class TracedAsyncGenerator(_TracedGeneratorBase):
    """Async generator wrapper that adds OpenTelemetry tracing without consuming the stream."""

    def __init__(
        self,
        tracer: "Tracer",
        async_generator,
        client: ApiClient,
        span,
        model_name: str = "",
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
            self.span.record_exception(e)
            self.span.end()
            raise


class TracedSyncContextManager:
    """Sync context manager wrapper for streaming methods."""

    def __init__(
        self,
        tracer: "Tracer",
        context_manager,
        client: ApiClient,
        span,
        model_name: str = "",
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.stream = None
        self.model_name = model_name

    def __enter__(self):
        self.stream = self.context_manager.__enter__()
        return TracedGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.context_manager.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        if hasattr(self, "span") and self.span:
            try:
                self.span.end()
            except Exception:
                pass


class TracedAsyncContextManager:
    """Async context manager wrapper for streaming methods."""

    def __init__(
        self,
        tracer: "Tracer",
        context_manager,
        client: ApiClient,
        span,
        model_name: str = "",
    ):
        self.tracer = tracer
        self.context_manager = context_manager
        self.client = client
        self.span = span
        self.stream = None
        self.model_name = model_name

    async def __aenter__(self):
        self.stream = await self.context_manager.__aenter__()
        return TracedAsyncGenerator(
            self.tracer, self.stream, self.client, self.span, self.model_name
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.context_manager.__aexit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        if hasattr(self, "span") and self.span:
            try:
                self.span.end()
            except Exception:
                pass


def _process_usage_data(span, usage_data, tracer: "Tracer", model_name: str = ""):
    """Process usage data and set span attributes."""
    prompt_tokens = 0
    completion_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0

    if hasattr(usage_data, "input_tokens"):
        prompt_tokens = getattr(usage_data, "input_tokens", 0) or 0
    if hasattr(usage_data, "output_tokens"):
        completion_tokens = getattr(usage_data, "output_tokens", 0) or 0

    if not prompt_tokens and hasattr(usage_data, "prompt_tokens"):
        prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0
    if not completion_tokens and hasattr(usage_data, "completion_tokens"):
        completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0

    if hasattr(usage_data, "cache_read_input_tokens"):
        cache_read_input_tokens = getattr(usage_data, "cache_read_input_tokens", 0) or 0
    if hasattr(usage_data, "cache_creation_input_tokens"):
        cache_creation_input_tokens = (
            getattr(usage_data, "cache_creation_input_tokens", 0) or 0
        )

    if prompt_tokens or completion_tokens:
        final_model_name = getattr(usage_data, "model", None) or model_name

        usage = _create_usage(
            final_model_name,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        )
        _set_usage_attributes(span, usage, tracer)


def _set_usage_attributes(span, usage: TraceUsage, tracer: "Tracer"):
    """Set usage attributes on the span for non-streaming responses."""

    set_span_attribute(span, "gen_ai.response.model", usage.model_name)
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
    prompt_tokens = 0
    completion_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    model_name = None
    message_content = None

    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import (
            openai_OpenAI,
            openai_AsyncOpenAI,
            openai_ChatCompletion,
            openai_Response,
            openai_ParsedChatCompletion,
        )

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        assert openai_ChatCompletion is not None, "OpenAI chat completion not found"
        assert openai_Response is not None, "OpenAI response not found"
        assert openai_ParsedChatCompletion is not None, (
            "OpenAI parsed chat completion not found"
        )

        if isinstance(client, openai_OpenAI) or isinstance(client, openai_AsyncOpenAI):
            if isinstance(response, openai_ChatCompletion):
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

                if isinstance(response, openai_ParsedChatCompletion):
                    message_content = response.choices[0].message.parsed
                else:
                    message_content = response.choices[0].message.content
            elif isinstance(response, openai_Response):
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
                ):  # type: ignore[attr-defined]
                    message_content = "".join(
                        seg.text  # type: ignore[attr-defined]
                        for seg in output0.content  # type: ignore[attr-defined]
                        if hasattr(seg, "text") and seg.text
                    )

            if model_name:
                return message_content, _create_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                )

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        if isinstance(client, together_Together) or isinstance(
            client, together_AsyncTogether
        ):
            model_name = (response.model or "") if hasattr(response, "model") else ""
            prompt_tokens = (
                response.usage.prompt_tokens
                if hasattr(response.usage, "prompt_tokens")
                and response.usage.prompt_tokens is not None
                else 0
            )  # type: ignore[attr-defined]
            completion_tokens = (
                response.usage.completion_tokens
                if hasattr(response.usage, "completion_tokens")
                and response.usage.completion_tokens is not None
                else 0
            )  # type: ignore[attr-defined]
            message_content = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else None
            )  # type: ignore[attr-defined]

            if model_name:
                model_name = "together_ai/" + model_name
                return message_content, _create_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
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
        if isinstance(client, google_genai_Client) or isinstance(
            client, google_genai_AsyncClient
        ):
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
            )  # type: ignore[attr-defined]

            if usage_metadata and hasattr(usage_metadata, "cached_content_token_count"):
                cache_read_input_tokens = usage_metadata.cached_content_token_count or 0

            if model_name:
                return message_content, _create_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                )

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        if isinstance(client, anthropic_Anthropic) or isinstance(
            client, anthropic_AsyncAnthropic
        ):
            model_name = getattr(response, "model", "") or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = (
                usage.input_tokens
                if usage
                and hasattr(usage, "input_tokens")
                and usage.input_tokens is not None
                else 0
            )
            completion_tokens = (
                usage.output_tokens
                if usage
                and hasattr(usage, "output_tokens")
                and usage.output_tokens is not None
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
            message_content = (
                response.content[0].text if hasattr(response, "content") else None
            )  # type: ignore[attr-defined]

            if model_name:
                return message_content, _create_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                )

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        if isinstance(client, groq_Groq) or isinstance(client, groq_AsyncGroq):
            model_name = (response.model or "") if hasattr(response, "model") else ""
            prompt_tokens = (
                response.usage.prompt_tokens
                if hasattr(response.usage, "prompt_tokens")
                and response.usage.prompt_tokens is not None
                else 0
            )  # type: ignore[attr-defined]
            completion_tokens = (
                response.usage.completion_tokens
                if hasattr(response.usage, "completion_tokens")
                and response.usage.completion_tokens is not None
                else 0
            )  # type: ignore[attr-defined]
            message_content = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else None
            )  # type: ignore[attr-defined]

            if model_name:
                model_name = "groq/" + model_name
                return message_content, _create_usage(
                    model_name,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                )

    judgeval_logger.warning(f"Unsupported client type: {type(client)}")
    return None, None


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
