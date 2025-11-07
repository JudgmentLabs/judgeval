from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Tuple,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import immutable_wrap_sync

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
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


def wrap_generate_content_sync(tracer: Tracer, client: Client) -> None:
    original_func = client.models.generate_content

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "GOOGLE_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer._inject_judgment_context(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: GenerateContentResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        output, usage_data = _format_google_output(result)
        set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, output)

        if usage_data:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_google_tokens(usage_data)
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                prompt_tokens,
            )
            set_span_attribute(
                span, AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            set_span_attribute(
                span, AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                cache_creation,
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        set_span_attribute(
            span,
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
            result.model_version if result.model_version else ctx["model_name"],
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

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
