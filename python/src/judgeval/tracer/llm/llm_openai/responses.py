from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    AsyncIterator,
    Generator,
    AsyncGenerator,
    ParamSpec,
    TypeVar,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.tracer.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses import Response

P = ParamSpec("P")
T = TypeVar("T")


def wrap_responses_create_sync(tracer: Tracer, client: OpenAI) -> None:
    original_func = client.responses.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return _wrap_responses_streaming_sync(tracer, original_func)(
                *args, **kwargs
            )
        return _wrap_responses_non_streaming_sync(tracer, original_func)(
            *args, **kwargs
        )

    setattr(client.responses, "create", dispatcher)


def _wrap_responses_non_streaming_sync(
    tracer: Tracer, original_func: Callable[..., Response]
) -> Callable[..., Response]:
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
            ctx["span"], AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Response) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        usage_data = result.usage if hasattr(result, "usage") else None
        if usage_data:
            prompt_tokens = usage_data.input_tokens or 0
            completion_tokens = usage_data.output_tokens or 0
            cache_read = usage_data.input_tokens_details.cached_tokens or 0

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
                span, AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        if hasattr(result, "model"):
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
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


def _wrap_responses_streaming_sync(
    tracer: Tracer, original_func: Callable[..., Iterator[Any]]
) -> Callable[..., Iterator[Any]]:
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
            ctx["span"], AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(ctx: Dict[str, Any], result: Iterator[Any]) -> Iterator[Any]:
        def traced_generator() -> Generator[Any, None, None]:
            for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
            span = ctx.get("span")
            if not span:
                return

            if hasattr(chunk, "type") and chunk.type == "response.output_text.delta":
                delta = getattr(chunk, "delta", None)
                if delta:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta
                    )

            if hasattr(chunk, "type") and chunk.type == "response.completed":
                if (
                    hasattr(chunk, "response")
                    and chunk.response
                    and hasattr(chunk.response, "usage")
                    and chunk.response.usage
                ):
                    prompt_tokens = chunk.response.usage.input_tokens or 0
                    completion_tokens = chunk.response.usage.output_tokens or 0
                    total_tokens = chunk.response.usage.total_tokens or 0
                    # Safely access nested cached_tokens
                    input_tokens_details = getattr(
                        chunk.response.usage, "input_tokens_details", None
                    )
                    cache_read = (
                        getattr(input_tokens_details, "cached_tokens", 0)
                        if input_tokens_details
                        else 0
                    )

                    set_cost_attribute(span, chunk.response.usage)
                    prompt_tokens, completion_tokens, cache_read, cache_creation = (
                        openai_tokens_converter(
                            prompt_tokens,
                            completion_tokens,
                            cache_read,
                            0,
                            total_tokens,
                        )
                    )

                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                        prompt_tokens,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                        completion_tokens,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
                        cache_read,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                        0,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_METADATA,
                        safe_serialize(chunk.response.usage),
                    )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_sync_iterator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return wrapped_generator()

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    return mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def wrap_responses_create_async(tracer: Tracer, client: AsyncOpenAI) -> None:
    original_func = client.responses.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return await _wrap_responses_streaming_async(tracer, original_func)(
                *args, **kwargs
            )
        return await _wrap_responses_non_streaming_async(tracer, original_func)(
            *args, **kwargs
        )

    setattr(client.responses, "create", dispatcher)


def _wrap_responses_non_streaming_async(
    tracer: Tracer, original_func: Callable[..., Awaitable[Response]]
) -> Callable[..., Awaitable[Response]]:
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
            ctx["span"], AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Response) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        usage_data = result.usage if hasattr(result, "usage") else None
        if usage_data:
            prompt_tokens = usage_data.input_tokens or 0
            completion_tokens = usage_data.output_tokens or 0
            cache_read = usage_data.input_tokens_details.cached_tokens or 0

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
                span, AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(usage_data),
            )

        if hasattr(result, "model"):
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_LLM_MODEL_NAME,
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


def _wrap_responses_streaming_async(
    tracer: Tracer, original_func: Callable[..., Awaitable[AsyncIterator[Any]]]
) -> Callable[..., Awaitable[AsyncIterator[Any]]]:
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
            ctx["span"], AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(
        ctx: Dict[str, Any], result: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        async def traced_generator() -> AsyncGenerator[Any, None]:
            async for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
            span = ctx.get("span")
            if not span:
                return

            if hasattr(chunk, "type") and chunk.type == "response.output_text.delta":
                delta = getattr(chunk, "delta", None)
                if delta:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta
                    )

            if hasattr(chunk, "type") and chunk.type == "response.completed":
                if (
                    hasattr(chunk, "response")
                    and chunk.response
                    and hasattr(chunk.response, "usage")
                    and chunk.response.usage
                ):
                    prompt_tokens = chunk.response.usage.input_tokens or 0
                    completion_tokens = chunk.response.usage.output_tokens or 0
                    total_tokens = chunk.response.usage.total_tokens or 0
                    # Safely access nested cached_tokens
                    input_tokens_details = getattr(
                        chunk.response.usage, "input_tokens_details", None
                    )
                    cache_read = (
                        getattr(input_tokens_details, "cached_tokens", 0)
                        if input_tokens_details
                        else 0
                    )

                    set_cost_attribute(span, chunk.response.usage)
                    prompt_tokens, completion_tokens, cache_read, cache_creation = (
                        openai_tokens_converter(
                            prompt_tokens,
                            completion_tokens,
                            cache_read,
                            0,
                            total_tokens,
                        )
                    )

                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                        prompt_tokens,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                        completion_tokens,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
                        cache_read,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                        0,
                    )
                    set_span_attribute(
                        span,
                        AttributeKeys.JUDGMENT_USAGE_METADATA,
                        safe_serialize(chunk.response.usage),
                    )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)

        def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.end()

        wrapped_generator = immutable_wrap_async_iterator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return wrapped_generator()

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    return mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )
