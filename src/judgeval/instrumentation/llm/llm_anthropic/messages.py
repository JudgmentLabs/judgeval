from __future__ import annotations
from inspect import signature
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

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.wrappers import (
    immutable_wrap_sync,
    immutable_wrap_async,
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import (
        Message,
        Usage,
        MessageDeltaUsage,
        RawMessageStreamEvent,
    )


@dont_throw
def _accumulate_message(ctx: Dict[str, Any], chunk: RawMessageStreamEvent) -> None:
    # Use the same accumulator as messages.stream(), without consuming ahead.
    # A failed capture must never be published as a complete response.
    if ctx.get("capture_failed"):
        return
    try:
        from anthropic.lib.streaming._messages import accumulate_event

        if "accumulator_kwargs" not in ctx:
            # Anthropic 2.x moved tool JSON buffers out of content blocks.
            ctx["accumulator_kwargs"] = (
                {"json_bufs": {}}
                if "json_bufs" in signature(accumulate_event).parameters
                else {}
            )
        ctx["message"] = accumulate_event(
            event=chunk,
            current_snapshot=ctx.get("message"),
            **ctx["accumulator_kwargs"],
        )
        if chunk.type == "message_stop":
            ctx["message_complete"] = True
    except Exception:
        ctx["capture_failed"] = True
        raise


def _record_message(span: Any, message: Message, *, complete: bool) -> None:
    if complete:
        # Exclude SDK accumulation/parsing internals, retaining native wire fields.
        payload = message.model_dump(
            mode="json",
            exclude={"content": {"__all__": {"__json_buf", "parsed_output"}}},
        )
        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(payload))
    if message.usage:
        prompt, output, cache_read, cache_creation = _extract_anthropic_tokens(
            message.usage
        )
        for key, value in (
            (AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt),
            (AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS, output),
            (AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS, cache_read),
            (AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation),
        ):
            span.set_attribute(key, value)
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(message.usage)
        )
    span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, message.model)


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


def wrap_messages_create_sync(client: Anthropic) -> None:
    original_func = client.messages.create

    def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return _wrap_streaming_sync(original_func)(*args, **kwargs)
        return _wrap_non_streaming_sync(original_func)(*args, **kwargs)

    setattr(client.messages, "create", dispatcher)


def _wrap_non_streaming_sync(
    original_func: Callable[..., Message],
) -> Callable[..., Message]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "ANTHROPIC_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Message) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_anthropic_tokens(result.usage)
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
                safe_serialize(result.usage),
            )

        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, result.model)

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
    original_func: Callable[..., Iterator[RawMessageStreamEvent]],
) -> Callable[..., Iterator[RawMessageStreamEvent]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "ANTHROPIC_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

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

            _accumulate_message(ctx, chunk)

            if chunk.type == "message_start":
                span.set_attribute(
                    AttributeKeys.JUDGMENT_LLM_MODEL_NAME, chunk.message.model
                )
            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data and not ctx.get("capture_failed"):
                # The SDK merges cumulative usage without zeroing fields omitted
                # from message_delta (notably input and cache tokens).
                usage_data = ctx["message"].usage
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
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
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_data)
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span and ctx.get("message") is not None:
                _record_message(
                    span,
                    ctx["message"],
                    complete=bool(ctx.get("message_complete"))
                    and not ctx.get("capture_failed", False),
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
            span.end()

    return mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )


def wrap_messages_create_async(client: AsyncAnthropic) -> None:
    original_func = client.messages.create

    async def dispatcher(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return await _wrap_streaming_async(original_func)(*args, **kwargs)
        return await _wrap_non_streaming_async(original_func)(*args, **kwargs)

    setattr(client.messages, "create", dispatcher)


def _wrap_non_streaming_async(
    original_func: Callable[..., Awaitable[Message]],
) -> Callable[..., Awaitable[Message]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "ANTHROPIC_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

    def post_hook(ctx: Dict[str, Any], result: Message) -> None:
        span = ctx.get("span")
        if not span:
            return

        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, safe_serialize(result))

        if result.usage:
            prompt_tokens, completion_tokens, cache_read, cache_creation = (
                _extract_anthropic_tokens(result.usage)
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
                safe_serialize(result.usage),
            )

        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, result.model)

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
    original_func: Callable[..., Awaitable[AsyncIterator[RawMessageStreamEvent]]],
) -> Callable[..., Awaitable[AsyncIterator[RawMessageStreamEvent]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "ANTHROPIC_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )

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

            _accumulate_message(ctx, chunk)

            if chunk.type == "message_start":
                span.set_attribute(
                    AttributeKeys.JUDGMENT_LLM_MODEL_NAME, chunk.message.model
                )
            usage_data = _extract_anthropic_chunk_usage(chunk)
            if usage_data and not ctx.get("capture_failed"):
                # The SDK merges cumulative usage without zeroing fields omitted
                # from message_delta (notably input and cache tokens).
                usage_data = ctx["message"].usage
            if usage_data:
                prompt_tokens, completion_tokens, cache_read, cache_creation = (
                    _extract_anthropic_tokens(usage_data)
                )
                span.set_attribute(
                    AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS, prompt_tokens
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
                    AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_data)
                )

        def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
            span = ctx.get("span")
            if span and ctx.get("message") is not None:
                _record_message(
                    span,
                    ctx["message"],
                    complete=bool(ctx.get("message_complete"))
                    and not ctx.get("capture_failed", False),
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
            span.end()

    return mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )
