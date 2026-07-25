from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
)

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import immutable_wrap_sync, immutable_wrap_async_iterator
from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from google.genai import Client
    from google.genai.types import (
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
    )


def _extract_google_tokens(
    usage: GenerateContentResponseUsageMetadata,
) -> Tuple[int, int, int, int]:
    prompt_tokens = (
        usage.prompt_token_count if usage.prompt_token_count is not None else 0
    )
    completion_tokens = (
        usage.candidates_token_count if usage.candidates_token_count is not None else 0
    )
    cache_read_input_tokens = (
        usage.cached_content_token_count
        if usage.cached_content_token_count is not None
        else 0
    )
    cache_creation_input_tokens = 0
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _format_google_output(
    response: GenerateContentResponse,
) -> Tuple[Optional[str], Optional[GenerateContentResponseUsageMetadata]]:
    return response.text, response.usage_metadata


def _process_google_stream_chunk(
    context: Dict[str, Any],
    chunk: GenerateContentResponse,
) -> None:
    if chunk.text:
        context["accumulated_text"] = context.get("accumulated_text", "") + chunk.text
    if chunk.usage_metadata is not None:
        context["usage_data"] = chunk.usage_metadata
    if hasattr(chunk, "model_version") and chunk.model_version:
        context["model_version"] = chunk.model_version


def _flush_stream_span(ctx: Dict[str, Any]) -> None:
    span = ctx.get("span")
    if not span:
        return

    accumulated = ctx.get("accumulated_text", "")
    if accumulated:
        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

    model_version = ctx.get("model_version") or ctx.get("model_name", "")
    if model_version:
        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model_version)

    usage_data = ctx.get("usage_data")
    if usage_data:
        prompt_tokens, completion_tokens, cache_read, cache_creation = (
            _extract_google_tokens(usage_data)
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
            prompt_tokens,
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
            cache_creation,
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA,
            safe_serialize(usage_data),
        )

    span.end()


def wrap_generate_content_sync(client: Client) -> None:
    original_func = client.models.generate_content

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "GOOGLE_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: GenerateContentResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        output, usage_data = _format_google_output(result)
        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, output)

        if usage_data:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_google_tokens(usage_data)
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                prompt_tokens,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                cache_creation,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        span.set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
            result.model_version if result.model_version else ctx["model_name"],
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    wrapped = immutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )

    setattr(client.models, "generate_content", wrapped)


def wrap_generate_content_stream_async(client: Client) -> None:
    original_func = client.aio.models.generate_content_stream

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "GOOGLE_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def yield_hook(ctx: Dict[str, Any], chunk: GenerateContentResponse) -> None:
        _process_google_stream_chunk(ctx, chunk)

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))

    def finally_hook(ctx: Dict[str, Any]) -> None:
        _flush_stream_span(ctx)

    wrapped = immutable_wrap_async_iterator(
        original_func,
        pre_hook=pre_hook,
        yield_hook=yield_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )

    setattr(client.aio.models, "generate_content_stream", wrapped)
