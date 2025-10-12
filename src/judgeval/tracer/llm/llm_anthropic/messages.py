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
    Optional,
    Tuple,
    Union,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    mutable_wrap_async,
    immutable_wrap_sync_generator,
    immutable_wrap_async_generator,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message


def _extract_anthropic_content(chunk: Any) -> str:
    if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "text"):
        return chunk.delta.text or ""

    if hasattr(chunk, "type") and chunk.type == "content_block_delta":
        if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "text"):
            return chunk.delta.text or ""
    return ""


def _extract_anthropic_tokens(usage_data: Any) -> Tuple[int, int, int, int]:
    prompt_tokens = getattr(usage_data, "input_tokens", 0) or 0
    completion_tokens = getattr(usage_data, "output_tokens", 0) or 0
    cache_read_input_tokens = getattr(usage_data, "cache_read_input_tokens", 0) or 0
    cache_creation_input_tokens = (
        getattr(usage_data, "cache_creation_input_tokens", 0) or 0
    )

    return (
        prompt_tokens,
        completion_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    )


def _extract_anthropic_chunk_usage(chunk: Any) -> Optional[Any]:
    if hasattr(chunk, "usage") and chunk.usage:
        return chunk.usage

    if hasattr(chunk, "type"):
        if (
            chunk.type == "message_start"
            and hasattr(chunk, "message")
            and chunk.message
        ):
            return getattr(chunk.message, "usage", None)
        elif chunk.type in ("message_delta", "message_stop"):
            return getattr(chunk, "usage", None)
    return None


def _format_anthropic_output(
    response: Any,
) -> Tuple[Optional[Union[str, list]], Optional[Any]]:
    message_content: Optional[Union[str, list]] = None
    usage_data: Optional[Any] = None

    try:
        usage_data = getattr(response, "usage", None)
        if hasattr(response, "content") and response.content:
            content_blocks = []
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    block_data = {
                        "type": "text",
                        "text": getattr(block, "text", ""),
                    }
                    if hasattr(block, "citations"):
                        block_data["citations"] = getattr(block, "citations", None)
                elif block_type == "tool_use":
                    block_data = {
                        "type": "tool_use",
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                elif block_type == "tool_result":
                    block_data = {
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "content": getattr(block, "content", None),
                    }
                else:
                    block_data = {"type": block_type}
                    for attr in [
                        "id",
                        "text",
                        "name",
                        "input",
                        "content",
                        "tool_use_id",
                        "citations",
                    ]:
                        if hasattr(block, attr):
                            block_data[attr] = getattr(block, attr)

                content_blocks.append(block_data)

            message_content = content_blocks if content_blocks else None
    except (AttributeError, IndexError, TypeError):
        pass

    return message_content, usage_data


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

        output, usage_data = _format_anthropic_output(result)
        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(output)
        )

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

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            getattr(result, "model", ctx["model_name"]),
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def _wrap_streaming_sync(
    tracer: Tracer, original_func: Callable[..., Iterator[Any]]
) -> Callable[..., Iterator[Any]]:
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

    def mutate_hook(ctx: Dict[str, Any], result: Iterator[Any]) -> Iterator[Any]:
        def traced_generator() -> Generator[Any, None, None]:
            for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
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

        def post_hook_inner(inner_ctx: Dict[str, Any], result: None) -> None:
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

        wrapped_generator = immutable_wrap_sync_generator(
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

        output, usage_data = _format_anthropic_output(result)
        set_span_attribute(
            span, AttributeKeys.GEN_AI_COMPLETION, safe_serialize(output)
        )

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

        set_span_attribute(
            span,
            AttributeKeys.GEN_AI_RESPONSE_MODEL,
            getattr(result, "model", ctx["model_name"]),
        )

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    def finally_hook(ctx: Dict[str, Any]) -> None:
        span = ctx.get("span")
        if span:
            span.end()

    return mutable_wrap_async(
        original_func,
        pre_hook=pre_hook,
        post_hook=post_hook,
        error_hook=error_hook,
        finally_hook=finally_hook,
    )


def _wrap_streaming_async(
    tracer: Tracer, original_func: Callable[..., Awaitable[AsyncIterator[Any]]]
) -> Callable[..., Awaitable[AsyncIterator[Any]]]:
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
        ctx: Dict[str, Any], result: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        async def traced_generator() -> AsyncGenerator[Any, None]:
            async for chunk in result:
                yield chunk

        def yield_hook(inner_ctx: Dict[str, Any], chunk: Any) -> None:
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

        wrapped_generator = immutable_wrap_async_generator(
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
