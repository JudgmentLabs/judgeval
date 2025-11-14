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

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
)
from judgeval.v1.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

P = ParamSpec("P")
T = TypeVar("T")


def wrap_beta_chat_completions_parse_sync(tracer: BaseTracer, client: OpenAI) -> None:
    original_func = client.beta.chat.completions.parse
    wrapped = _wrap_beta_non_streaming_sync(tracer, original_func)
    setattr(client.beta.chat.completions, "parse", wrapped)


def _wrap_beta_non_streaming_sync(
    tracer: BaseTracer, original_func: Callable[P, ParsedChatCompletion[T]]
) -> Callable[P, ParsedChatCompletion[T]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ParsedChatCompletion[T]) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        usage_data = result.usage
        if usage_data:
            prompt_tokens = usage_data.prompt_tokens or 0
            completion_tokens = usage_data.completion_tokens or 0
            cache_read = 0
            prompt_tokens_details = usage_data.prompt_tokens_details
            if prompt_tokens_details:
                cache_read = prompt_tokens_details.cached_tokens or 0

            set_cost_attribute(span, usage_data)

            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                openai_tokens_converter(
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    0,
                    usage_data.total_tokens,
                )
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
                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        span.set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
            result.model or ctx["model_name"],
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

    return immutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def wrap_beta_chat_completions_parse_async(
    tracer: BaseTracer, client: AsyncOpenAI
) -> None:
    original_func = client.beta.chat.completions.parse
    wrapped = _wrap_beta_non_streaming_async(tracer, original_func)
    setattr(client.beta.chat.completions, "parse", wrapped)


def _wrap_beta_non_streaming_async(
    tracer: BaseTracer, original_func: Callable[P, Awaitable[ParsedChatCompletion[T]]]
) -> Callable[P, Awaitable[ParsedChatCompletion[T]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ParsedChatCompletion[T]) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        usage_data = result.usage
        if usage_data:
            prompt_tokens = usage_data.prompt_tokens or 0
            completion_tokens = usage_data.completion_tokens or 0
            cache_read = 0
            prompt_tokens_details = usage_data.prompt_tokens_details
            if prompt_tokens_details:
                cache_read = prompt_tokens_details.cached_tokens or 0

            set_cost_attribute(span, usage_data)

            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                openai_tokens_converter(
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    0,
                    usage_data.total_tokens,
                )
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
                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        span.set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
            result.model or ctx["model_name"],
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

    return immutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )
