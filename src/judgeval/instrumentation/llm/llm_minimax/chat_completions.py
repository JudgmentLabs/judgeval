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

from opentelemetry.trace import Status, StatusCode
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
from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


def wrap_chat_completions_create_sync(client: OpenAI) -> None:
    original_func = client.chat.completions.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        extra_headers = kwargs.get("extra_headers") or {}
        if (
            isinstance(extra_headers, dict)
            and extra_headers.get("X-Stainless-Raw-Response") == "stream"
        ):
            return original_func(*args, **kwargs)

        if kwargs.get("stream", False):
            return _wrap_streaming_sync(original_func)(*args, **kwargs)
        return _wrap_non_streaming_sync(original_func)(*args, **kwargs)

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_sync(
    original_func: Callable[..., ChatCompletion],
) -> Callable[..., ChatCompletion]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "MINIMAX_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        model_name = kwargs.get("model", "")
        prefixed = f"minimax/{model_name}" if model_name else ""
        ctx["model_name"] = prefixed
        ctx["span"].set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed)

    def post_hook(ctx: Dict[str, Any], result: ChatCompletion) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                result.usage.prompt_tokens or 0,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                result.usage.completion_tokens or 0,
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


def _wrap_streaming_sync(
    original_func: Callable[..., Iterator[ChatCompletionChunk]],
) -> Callable[..., Iterator[ChatCompletionChunk]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "MINIMAX_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        model_name = kwargs.get("model", "")
        prefixed = f"minimax/{model_name}" if model_name else ""
        ctx["model_name"] = prefixed
        ctx["span"].set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed)
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
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                    chunk.usage.prompt_tokens or 0,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                    chunk.usage.completion_tokens or 0,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.set_attribute(
                    AttributeKeys.GEN_AI_COMPLETION,
                    ctx.get("accumulated_content", ""),
                )

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

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
            span.set_status(Status(StatusCode.ERROR))

    return mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def wrap_chat_completions_create_async(client: AsyncOpenAI) -> None:
    original_func = client.chat.completions.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        extra_headers = kwargs.get("extra_headers") or {}
        if (
            isinstance(extra_headers, dict)
            and extra_headers.get("X-Stainless-Raw-Response") == "stream"
        ):
            return await original_func(*args, **kwargs)

        if kwargs.get("stream", False):
            return await _wrap_streaming_async(original_func)(*args, **kwargs)
        return await _wrap_non_streaming_async(original_func)(*args, **kwargs)

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_async(
    original_func: Callable[..., Awaitable[ChatCompletion]],
) -> Callable[..., Awaitable[ChatCompletion]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "MINIMAX_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        model_name = kwargs.get("model", "")
        prefixed = f"minimax/{model_name}" if model_name else ""
        ctx["model_name"] = prefixed
        ctx["span"].set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed)

    def post_hook(ctx: Dict[str, Any], result: ChatCompletion) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                result.usage.prompt_tokens or 0,
            )
            span.set_attribute(
                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                result.usage.completion_tokens or 0,
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


def _wrap_streaming_async(
    original_func: Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]],
) -> Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "MINIMAX_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        model_name = kwargs.get("model", "")
        prefixed = f"minimax/{model_name}" if model_name else ""
        ctx["model_name"] = prefixed
        ctx["span"].set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, prefixed)
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
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                    chunk.usage.prompt_tokens or 0,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                    chunk.usage.completion_tokens or 0,
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(chunk.usage),
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span:
                span.set_attribute(
                    AttributeKeys.GEN_AI_COMPLETION,
                    ctx.get("accumulated_content", ""),
                )

        def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
            span = ctx.get("span")
            if span:
                span.record_exception(error)
                span.set_status(Status(StatusCode.ERROR))

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
            span.set_status(Status(StatusCode.ERROR))

    return mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )
