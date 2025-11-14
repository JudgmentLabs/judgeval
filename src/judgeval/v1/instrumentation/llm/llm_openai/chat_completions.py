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
from packaging import version

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
from judgeval.v1.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

P = ParamSpec("P")
T = TypeVar("T")


def _supports_stream_options() -> bool:
    try:
        import openai

        return version.parse(openai.__version__) >= version.parse("1.26.0")
    except Exception:
        return False


def wrap_chat_completions_create_sync(tracer: BaseTracer, client: OpenAI) -> None:
    original_func = client.chat.completions.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return _wrap_streaming_sync(tracer, original_func)(*args, **kwargs)
        return _wrap_non_streaming_sync(tracer, original_func)(*args, **kwargs)

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_sync(
    tracer: BaseTracer, original_func: Callable[..., ChatCompletion]
) -> Callable[..., ChatCompletion]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ChatCompletion) -> None:
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


def _wrap_streaming_sync(
    tracer: BaseTracer, original_func: Callable[..., Iterator[ChatCompletionChunk]]
) -> Callable[..., Iterator[ChatCompletionChunk]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_kwargs_hook(ctx: Dict[str, Any], kwargs: Any) -> Any:
        if "stream_options" not in kwargs and _supports_stream_options():
            modified_kwargs = dict(kwargs)
            modified_kwargs["stream_options"] = {"include_usage": True}
            return modified_kwargs
        return kwargs

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
                if delta and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                cache_read = 0
                if chunk.usage.prompt_tokens_details:
                    cache_read = chunk.usage.prompt_tokens_details.cached_tokens or 0

                set_cost_attribute(span, chunk.usage)

                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    openai_tokens_converter(
                        prompt_tokens,
                        completion_tokens,
                        cache_read,
                        0,
                        chunk.usage.total_tokens,
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
        mutate_kwargs_hook=mutate_kwargs_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def wrap_chat_completions_create_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    original_func = client.chat.completions.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return await _wrap_streaming_async(tracer, original_func)(*args, **kwargs)
        return await _wrap_non_streaming_async(tracer, original_func)(*args, **kwargs)

    setattr(client.chat.completions, "create", dispatcher)


def _wrap_non_streaming_async(
    tracer: BaseTracer, original_func: Callable[..., Awaitable[ChatCompletion]]
) -> Callable[..., Awaitable[ChatCompletion]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: ChatCompletion) -> None:
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


def _wrap_streaming_async(
    tracer: BaseTracer,
    original_func: Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]],
) -> Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "OPENAI_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_kwargs_hook(ctx: Dict[str, Any], kwargs: Any) -> Any:
        if "stream_options" not in kwargs and _supports_stream_options():
            modified_kwargs = dict(kwargs)
            modified_kwargs["stream_options"] = {"include_usage": True}
            return modified_kwargs
        return kwargs

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
                if delta and delta.content:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + delta.content
                    )

            if hasattr(chunk, "usage") and chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                cache_read = 0
                if chunk.usage.prompt_tokens_details:
                    cache_read = chunk.usage.prompt_tokens_details.cached_tokens or 0

                set_cost_attribute(span, chunk.usage)

                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    openai_tokens_converter(
                        prompt_tokens,
                        completion_tokens,
                        cache_read,
                        0,
                        chunk.usage.total_tokens,
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
        mutate_kwargs_hook=mutate_kwargs_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )
