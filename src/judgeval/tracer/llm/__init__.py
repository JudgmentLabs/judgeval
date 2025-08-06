from __future__ import annotations
import sys
from typing import Callable, Tuple, Optional, Any, TYPE_CHECKING
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
from judgeval.tracer.keys import EventKeys
from judgeval.utils.serialize import safe_serialize

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


def wrap_provider(tracer: Tracer, client: ApiClient) -> ApiClient:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, Google GenAI, and Groq clients.
    """
    span_name, original_create = _get_client_config(client)

    def wrapped(function):
        def wrapper(*args, **kwargs):
            with sync_span_context(tracer, span_name, {"span.type": "llm"}) as span:
                span.add_event(
                    EventKeys.JUDGMENT_INPUT, {"value": safe_serialize(kwargs)}
                )
                try:
                    response = function(*args, **kwargs)
                    output, usage = _format_output_data(client, response)
                    if output:
                        span.add_event(EventKeys.JUDGMENT_OUTPUT, {"value": output})
                    if usage:
                        span.add_event(
                            EventKeys.LLM_USAGE, {"value": safe_serialize(usage)}
                        )
                    return response
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    def wrapped_async(function):
        async def wrapper(*args, **kwargs):
            async with async_span_context(
                tracer, span_name, {"span.type": "llm"}
            ) as span:
                span.add_event(
                    EventKeys.JUDGMENT_INPUT, {"value": safe_serialize(kwargs)}
                )
                try:
                    response = await function(*args, **kwargs)
                    output, usage = _format_output_data(client, response)
                    if output:
                        span.add_event(EventKeys.JUDGMENT_OUTPUT, {"value": output})
                    if usage:
                        span.add_event(
                            EventKeys.LLM_USAGE, {"value": safe_serialize(usage)}
                        )
                    return response
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        if isinstance(client, openai_OpenAI):
            print("wrapping openai")
            setattr(client.chat.completions, "create", wrapped(original_create))
        elif isinstance(client, openai_AsyncOpenAI):
            setattr(client.chat.completions, "create", wrapped_async(original_create))

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        if isinstance(client, together_Together):
            setattr(client.chat.completions, "create", wrapped(original_create))
        elif isinstance(client, together_AsyncTogether):
            setattr(client.chat.completions, "create", wrapped_async(original_create))

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        if isinstance(client, anthropic_Anthropic):
            setattr(client.messages, "create", wrapped(original_create))
        elif isinstance(client, anthropic_AsyncAnthropic):
            setattr(client.messages, "create", wrapped_async(original_create))

    if HAS_GOOGLE_GENAI:
        from judgeval.tracer.llm.providers import (
            google_genai_Client,
            google_genai_AsyncClient,
        )

        assert google_genai_Client is not None, "Google GenAI client not found"
        assert (
            google_genai_AsyncClient is not None
        ), "Google GenAI async client not found"
        if isinstance(client, google_genai_Client):
            setattr(client.models, "generate_content", wrapped(original_create))
        elif isinstance(client, google_genai_AsyncClient):
            setattr(client.models, "generate_content", wrapped_async(original_create))

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        if isinstance(client, groq_Groq):
            setattr(client.chat.completions, "create", wrapped(original_create))
        elif isinstance(client, groq_AsyncGroq):
            setattr(client.chat.completions, "create", wrapped_async(original_create))

    return client


def _get_client_config(client: ApiClient) -> tuple[str, Callable]:
    if HAS_OPENAI:
        from judgeval.tracer.llm.providers import openai_OpenAI, openai_AsyncOpenAI

        assert openai_OpenAI is not None, "OpenAI client not found"
        assert openai_AsyncOpenAI is not None, "OpenAI async client not found"
        if isinstance(client, openai_OpenAI):
            return "OPENAI_API_CALL", client.chat.completions.create
        elif isinstance(client, openai_AsyncOpenAI):
            return "OPENAI_API_CALL", client.chat.completions.create

    if HAS_TOGETHER:
        from judgeval.tracer.llm.providers import (
            together_Together,
            together_AsyncTogether,
        )

        assert together_Together is not None, "Together client not found"
        assert together_AsyncTogether is not None, "Together async client not found"
        if isinstance(client, together_Together):
            return "TOGETHER_API_CALL", client.chat.completions.create
        elif isinstance(client, together_AsyncTogether):
            return "TOGETHER_API_CALL", client.chat.completions.create

    if HAS_ANTHROPIC:
        from judgeval.tracer.llm.providers import (
            anthropic_Anthropic,
            anthropic_AsyncAnthropic,
        )

        assert anthropic_Anthropic is not None, "Anthropic client not found"
        assert anthropic_AsyncAnthropic is not None, "Anthropic async client not found"
        if isinstance(client, anthropic_Anthropic):
            return "ANTHROPIC_API_CALL", client.messages.create
        elif isinstance(client, anthropic_AsyncAnthropic):
            return "ANTHROPIC_API_CALL", client.messages.create

    if HAS_GOOGLE_GENAI:
        from judgeval.tracer.llm.providers import (
            google_genai_Client,
            google_genai_AsyncClient,
        )

        assert google_genai_Client is not None, "Google GenAI client not found"
        assert (
            google_genai_AsyncClient is not None
        ), "Google GenAI async client not found"
        if isinstance(client, google_genai_Client):
            return "GOOGLE_API_CALL", client.models.generate_content
        elif isinstance(client, google_genai_AsyncClient):
            return "GOOGLE_API_CALL", client.models.generate_content

    if HAS_GROQ:
        from judgeval.tracer.llm.providers import groq_Groq, groq_AsyncGroq

        assert groq_Groq is not None, "Groq client not found"
        assert groq_AsyncGroq is not None, "Groq async client not found"
        if isinstance(client, groq_Groq):
            return "GROQ_API_CALL", client.chat.completions.create
        elif isinstance(client, groq_AsyncGroq):
            return "GROQ_API_CALL", client.chat.completions.create

    raise ValueError(f"Unsupported client type: {type(client)}")


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
        assert (
            openai_ParsedChatCompletion is not None
        ), "OpenAI parsed chat completion not found"

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
                if hasattr(output0, "content") and output0.content and hasattr(output0.content, "__iter__"):  # type: ignore[attr-defined]
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
            prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") and response.usage.prompt_tokens is not None else 0  # type: ignore[attr-defined]
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") and response.usage.completion_tokens is not None else 0  # type: ignore[attr-defined]
            message_content = response.choices[0].message.content if hasattr(response, "choices") else None  # type: ignore[attr-defined]

            if model_name:
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
        assert (
            google_genai_AsyncClient is not None
        ), "Google GenAI async client not found"
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
            prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") and response.usage.prompt_tokens is not None else 0  # type: ignore[attr-defined]
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") and response.usage.completion_tokens is not None else 0  # type: ignore[attr-defined]
            message_content = response.choices[0].message.content if hasattr(response, "choices") else None  # type: ignore[attr-defined]

            if model_name:
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
