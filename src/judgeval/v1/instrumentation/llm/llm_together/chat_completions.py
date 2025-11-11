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
)

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    immutable_wrap_async,
    immutable_wrap_sync,
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from together import Together, AsyncTogether  # type: ignore[import-untyped]
    from together.types import ChatCompletionResponse, ChatCompletionChunk  # type: ignore[import-untyped]
    from together.types.common import UsageData  # type: ignore[import-untyped]


def _extract_together_tokens(usage: UsageData) -> tuple[int, int, int, int]:
    prompt_tokens = usage.prompt_tokens if usage.prompt_tokens is not None else 0
    completion_tokens = (
        usage.completion_tokens if usage.completion_tokens is not None else 0
    )
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def wrap_chat_completions_create_sync(tracer: BaseTracer, client: Together) -> None:
    original_func = client.chat.completions.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return _wrap_streaming_sync(tracer, original_func)(*args, **kwargs)  # type: ignore[arg-type]
        return _wrap_non_streaming_sync(tracer, original_func)(*args, **kwargs)  # type: ignore[arg-type]

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_sync(
    tracer: BaseTracer, original_func: Callable[..., ChatCompletionResponse]
) -> Callable[..., ChatCompletionResponse]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "TOGETHER_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        prefixed_model_name = (
            f"together_ai/{ctx['model_name']}" if ctx["model_name"] else ""
        )
        ctx["model_name"] = prefixed_model_name
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed_model_name
        )

    def post_hook(ctx: Dict[str, Any], result: ChatCompletionResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            prompt_tokens, completion_tokens, _, _ = _extract_together_tokens(
                result.usage
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                prompt_tokens,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(result.usage),
            )

        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"])

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


def _wrap_streaming_sync(
    tracer: BaseTracer, original_func: Callable[..., Iterator[ChatCompletionChunk]]
) -> Callable[..., Iterator[ChatCompletionChunk]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "TOGETHER_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        prefixed_model_name = (
            f"together_ai/{ctx['model_name']}" if ctx["model_name"] else ""
        )
        ctx["model_name"] = prefixed_model_name
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed_model_name
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(
        ctx: Dict[str, Any], result: Iterator[ChatCompletionChunk]
    ) -> Iterator[ChatCompletionChunk]:
        def traced_generator() -> Generator[ChatCompletionChunk, None, None]:
            for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: ChatCompletionChunk) -> None:
            span = ctx.get("span")
            if not span:
                return

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if chunk.usage:
                prompt_tokens, completion_tokens, _, _ = _extract_together_tokens(
                    chunk.usage
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                    prompt_tokens,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

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


def wrap_chat_completions_create_async(
    tracer: BaseTracer, client: AsyncTogether
) -> None:
    original_func = client.chat.completions.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return await _wrap_streaming_async(tracer, original_func)(*args, **kwargs)  # type: ignore[arg-type]
        return await _wrap_non_streaming_async(tracer, original_func)(*args, **kwargs)  # type: ignore[arg-type]

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_async(
    tracer: BaseTracer, original_func: Callable[..., Awaitable[ChatCompletionResponse]]
) -> Callable[..., Awaitable[ChatCompletionResponse]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "TOGETHER_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        prefixed_model_name = (
            f"together_ai/{ctx['model_name']}" if ctx["model_name"] else ""
        )
        ctx["model_name"] = prefixed_model_name
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed_model_name
        )

    def post_hook(ctx: Dict[str, Any], result: ChatCompletionResponse) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            prompt_tokens, completion_tokens, _, _ = _extract_together_tokens(
                result.usage
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                prompt_tokens,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(result.usage),
            )

        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"])

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


def _wrap_streaming_async(
    tracer: BaseTracer,
    original_func: Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]],
) -> Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "TOGETHER_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        prefixed_model_name = (
            f"together_ai/{ctx['model_name']}" if ctx["model_name"] else ""
        )
        ctx["model_name"] = prefixed_model_name
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed_model_name
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(
        ctx: Dict[str, Any], result: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        async def traced_generator() -> AsyncGenerator[ChatCompletionChunk, None]:
            async for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: ChatCompletionChunk) -> None:
            span = ctx.get("span")
            if not span:
                return

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and hasattr(delta, "content") and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if chunk.usage:
                prompt_tokens, completion_tokens, _, _ = _extract_together_tokens(
                    chunk.usage
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                    prompt_tokens,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

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
