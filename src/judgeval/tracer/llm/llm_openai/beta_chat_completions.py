from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    ParamSpec,
    TypeVar,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

P = ParamSpec("P")
T = TypeVar("T")


def wrap_beta_chat_completions_parse_sync(tracer: Tracer, client: OpenAI) -> None:
    original_func = client.beta.chat.completions.parse
    wrapped = _wrap_beta_non_streaming_sync(tracer, original_func)
    setattr(client.beta.chat.completions, "parse", wrapped)


def _wrap_beta_non_streaming_sync(
    tracer: Tracer, original_func: Callable[P, ParsedChatCompletion[T]]
) -> Callable[P, ParsedChatCompletion[T]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer._inject_judgment_context(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ParsedChatCompletion[T]) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        usage_data = result.usage
        if usage_data:
            prompt_tokens = usage_data.prompt_tokens or 0
            completion_tokens = usage_data.completion_tokens or 0
            cache_read = 0
            prompt_tokens_details = usage_data.prompt_tokens_details
            if prompt_tokens_details:
                cache_read = prompt_tokens_details.cached_tokens or 0

            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            result.model or ctx["model_name"],
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return immutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def wrap_beta_chat_completions_parse_async(tracer: Tracer, client: AsyncOpenAI) -> None:
    original_func = client.beta.chat.completions.parse
    wrapped = _wrap_beta_non_streaming_async(tracer, original_func)
    setattr(client.beta.chat.completions, "parse", wrapped)


def _wrap_beta_non_streaming_async(
    tracer: Tracer, original_func: Callable[P, Awaitable[ParsedChatCompletion[T]]]
) -> Callable[P, Awaitable[ParsedChatCompletion[T]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer._inject_judgment_context(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ParsedChatCompletion[T]) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        usage_data = result.usage
        if usage_data:
            prompt_tokens = usage_data.prompt_tokens or 0
            completion_tokens = usage_data.completion_tokens or 0
            cache_read = 0
            prompt_tokens_details = usage_data.prompt_tokens_details
            if prompt_tokens_details:
                cache_read = prompt_tokens_details.cached_tokens or 0

            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read
            )
            set_span_attribute(
                span, AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            result.model or ctx["model_name"],
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return immutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )
