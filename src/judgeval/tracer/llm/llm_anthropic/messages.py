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
    Tuple,
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

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import (
        Message,
        Usage,
        MessageDeltaUsage,
        RawMessageStreamEvent,
    )


def _extract_anthropic_content(chunk: RawMessageStreamEvent) -> str:
    if chunk.type == "content_block_delta":
        delta = chunk.delta
        if delta.type == "text_delta" and delta.text:
            return delta.text
    return ""


def _extract_anthropic_tokens(
    usage: Usage | MessageDeltaUsage,
) -> Tuple[int, int, int, int]:
    input_tokens = usage.input_tokens if usage.input_tokens is not None else 0
    output_tokens = usage.output_tokens if usage.output_tokens is not None else 0
    cache_read = (
        usage.cache_read_input_tokens
        if usage.cache_read_input_tokens is not None
        else 0
    )
    cache_creation = (
        usage.cache_creation_input_tokens
        if usage.cache_creation_input_tokens is not None
        else 0
    )
    return (input_tokens, output_tokens, cache_read, cache_creation)


def _extract_anthropic_chunk_usage(
    chunk: RawMessageStreamEvent,
) -> Usage | MessageDeltaUsage | None:
    if chunk.type == "message_start":
        return chunk.message.usage if chunk.message else None
    elif chunk.type == "message_delta":
        return chunk.usage if hasattr(chunk, "usage") else None
    return None


def wrap_messages_create_sync(tracer: Tracer, client: Anthropic) -> None:
    original_func = client.messages.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return _wrap_streaming_sync(tracer, original_func)(*args, **kwargs)
        return _wrap_non_streaming_sync(tracer, original_func)(*args, **kwargs)

    setattr(client.messages, "create", dispatcher)


def _wrap_non_streaming_sync(
    tracer: Tracer, original_func: Callable[..., Message]
) -> Callable[..., Message]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Message) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        if result.usage:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_anthropic_tokens(result.usage)
            )
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
                span,
                AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                cache_creation,
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(result.usage),
            )

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            result.model,
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


def _wrap_streaming_sync(
    tracer: Tracer, original_func: Callable[..., Iterator[RawMessageStreamEvent]]
) -> Callable[..., Iterator[RawMessageStreamEvent]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(
        ctx: Dict[str, Any], result: Iterator[RawMessageStreamEvent]
    ) -> Iterator[RawMessageStreamEvent]:
        def traced_generator() -> Generator[RawMessageStreamEvent, None, None]:
            for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: RawMessageStreamEvent) -> None:
            span = ctx.get("span")
            if not span:
                return

            content = _extract_anthropic_content(chunk)
            if content:
                ctx["accumulated_content"] = (
                    ctx.get("accumulated_content", "") + content
                )

            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
                )
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
                    span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    cache_creation,
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(usage_data),
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


def wrap_messages_create_async(tracer: Tracer, client: AsyncAnthropic) -> None:
    original_func = client.messages.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return await _wrap_streaming_async(tracer, original_func)(*args, **kwargs)
        return await _wrap_non_streaming_async(tracer, original_func)(*args, **kwargs)

    setattr(client.messages, "create", dispatcher)


def _wrap_non_streaming_async(
    tracer: Tracer, original_func: Callable[..., Awaitable[Message]]
) -> Callable[..., Awaitable[Message]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Message) -> None:
        span = ctx.get("span")
        if not span:
            return

        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result)
        )

        if result.usage:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_anthropic_tokens(result.usage)
            )
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
                span,
                AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                cache_creation,
            )
            set_span_attribute(
                span,
                AttributeKeys.JUDGMENT_USAGE_METADATA,
                safe_serialize(result.usage),
            )

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            result.model,
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


def _wrap_streaming_async(
    tracer: Tracer,
    original_func: Callable[..., Awaitable[AsyncIterator[RawMessageStreamEvent]]],
) -> Callable[..., Awaitable[AsyncIterator[RawMessageStreamEvent]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        tracer.add_agent_attributes_to_span(ctx["span"])
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs)
        )
        ctx["model_name"] = kwargs.get("model", "")
        set_span_attribute(
            ctx["span"], AttributeKeys.GEN_AI_REQUEST_MODEL, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_hook(
        ctx: Dict[str, Any], result: AsyncIterator[RawMessageStreamEvent]
    ) -> AsyncIterator[RawMessageStreamEvent]:
        async def traced_generator() -> AsyncGenerator[RawMessageStreamEvent, None]:
            async for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: RawMessageStreamEvent) -> None:
            span = ctx.get("span")
            if not span:
                return

            content = _extract_anthropic_content(chunk)
            if content:
                ctx["accumulated_content"] = (
                    ctx.get("accumulated_content", "") + content
                )

            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
                )
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
                    span,
                    AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
                    cache_creation,
                )
                set_span_attribute(
                    span,
                    AttributeKeys.JUDGMENT_USAGE_METADATA,
                    safe_serialize(usage_data),
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
