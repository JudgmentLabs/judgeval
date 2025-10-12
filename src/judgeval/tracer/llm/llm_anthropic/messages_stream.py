from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    AsyncGenerator,
)

from judgeval.tracer.keys import AttributeKeys
from judgeval.tracer.utils import set_span_attribute
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    immutable_wrap_sync_generator,
    immutable_wrap_async_generator,
)
from judgeval.tracer.llm.llm_anthropic.messages import (
    _extract_anthropic_tokens,
)

if TYPE_CHECKING:
    from judgeval.tracer import Tracer
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.lib.streaming import (
        MessageStreamManager,
        AsyncMessageStreamManager,
        MessageStream,
        AsyncMessageStream,
    )


def wrap_messages_stream_sync(tracer: Tracer, client: Anthropic) -> None:
    original_func = client.messages.stream

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
        ctx: Dict[str, Any], result: MessageStreamManager
    ) -> MessageStreamManager:
        original_manager = result

        class WrappedMessageStreamManager:
            def __init__(self, manager: MessageStreamManager):
                self._manager = manager

            def __enter__(self) -> MessageStream:
                stream = self._manager.__enter__()
                post_hook_enter_impl(stream)
                return stream

            def __exit__(self, exc_type, exc_val, exc_tb):
                result = self._manager.__exit__(exc_type, exc_val, exc_tb)
                post_hook_exit_impl()
                return result

            def __getattr__(self, name):
                return getattr(self._manager, name)

        def post_hook_enter_impl(stream: MessageStream) -> None:
            ctx["stream"] = stream
            original_text_stream = stream.text_stream

            def traced_text_stream() -> Generator[str, None, None]:
                for text_chunk in original_text_stream:
                    yield text_chunk

            def yield_hook(inner_ctx: Dict[str, Any], text_chunk: str) -> None:
                span = ctx.get("span")
                if span and text_chunk:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + text_chunk
                    )

            def post_hook_inner(inner_ctx: Dict[str, Any], result: None) -> None:
                pass

            def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
                span = ctx.get("span")
                if span:
                    span.record_exception(error)

            def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
                pass

            wrapped_text_stream = immutable_wrap_sync_generator(
                traced_text_stream,
                yield_hook=yield_hook,
                post_hook=post_hook_inner,
                error_hook=error_hook_inner,
                finally_hook=finally_hook_inner,
            )

            stream.text_stream = wrapped_text_stream()

        def post_hook_exit_impl() -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

                stream = ctx.get("stream")
                if stream:
                    try:
                        final_message = stream.get_final_message()
                        if hasattr(final_message, "usage") and final_message.usage:
                            usage_data = final_message.usage
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_anthropic_tokens(usage_data)
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
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

                        if hasattr(final_message, "model"):
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                getattr(final_message, "model", ctx["model_name"]),
                            )
                    except Exception:
                        pass

                span.end()

        return WrappedMessageStreamManager(original_manager)  # type: ignore[return-value]

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    wrapped = mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )

    setattr(client.messages, "stream", wrapped)


def wrap_messages_stream_async(tracer: Tracer, client: AsyncAnthropic) -> None:
    original_func = client.messages.stream

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
        ctx: Dict[str, Any], result: AsyncMessageStreamManager
    ) -> AsyncMessageStreamManager:
        original_manager = result

        class WrappedAsyncMessageStreamManager:
            def __init__(self, manager: AsyncMessageStreamManager):
                self._manager = manager

            async def __aenter__(self) -> AsyncMessageStream:
                stream = await self._manager.__aenter__()
                post_hook_aenter_impl(stream)
                return stream

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                result = await self._manager.__aexit__(exc_type, exc_val, exc_tb)
                await post_hook_aexit_impl()
                return result

            def __getattr__(self, name):
                return getattr(self._manager, name)

        def post_hook_aenter_impl(stream: AsyncMessageStream) -> None:
            ctx["stream"] = stream
            original_text_stream = stream.text_stream

            async def traced_text_stream() -> AsyncGenerator[str, None]:
                async for text_chunk in original_text_stream:
                    yield text_chunk

            def yield_hook(inner_ctx: Dict[str, Any], text_chunk: str) -> None:
                span = ctx.get("span")
                if span and text_chunk:
                    ctx["accumulated_content"] = (
                        ctx.get("accumulated_content", "") + text_chunk
                    )

            def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
                pass

            def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
                span = ctx.get("span")
                if span:
                    span.record_exception(error)

            def finally_hook_inner_sync(inner_ctx: Dict[str, Any]) -> None:
                pass

            wrapped_text_stream = immutable_wrap_async_generator(
                traced_text_stream,
                yield_hook=yield_hook,
                post_hook=post_hook_inner,
                error_hook=error_hook_inner,
                finally_hook=finally_hook_inner_sync,
            )

            stream.text_stream = wrapped_text_stream()

        async def post_hook_aexit_impl() -> None:
            span = ctx.get("span")
            if span:
                accumulated = ctx.get("accumulated_content", "")
                set_span_attribute(span, AttributeKeys.GEN_AI_COMPLETION, accumulated)

                stream = ctx.get("stream")
                if stream:
                    try:
                        final_message = await stream.get_final_message()
                        if hasattr(final_message, "usage") and final_message.usage:
                            usage_data = final_message.usage
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_anthropic_tokens(usage_data)
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
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

                        if hasattr(final_message, "model"):
                            set_span_attribute(
                                span,
                                AttributeKeys.GEN_AI_RESPONSE_MODEL,
                                getattr(final_message, "model", ctx["model_name"]),
                            )
                    except Exception:
                        pass

                span.end()

        return WrappedAsyncMessageStreamManager(original_manager)  # type: ignore[return-value]

    def error_hook(ctx: Dict[str, Any], error: Exception) -> None:
        span = ctx.get("span")
        if span:
            span.record_exception(error)

    wrapped = mutable_wrap_sync(
        original_func,
        pre_hook=pre_hook,
        mutate_hook=mutate_hook,
        error_hook=error_hook,
    )

    setattr(client.messages, "stream", wrapped)


if __name__ == "__main__":
    import os
    from anthropic import Anthropic, AsyncAnthropic
    from judgeval.tracer import Tracer

    tracer = Tracer(project_name="anthropic-stream-example")

    print("=" * 60)
    print("Sync Streaming Example")
    print("=" * 60)

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    wrap_messages_stream_sync(tracer, client)

    with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Write a haiku about Python"}],
        model="claude-sonnet-4-5",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    print("\n")

    print("\n" + "=" * 60)
    print("Async Streaming Example")
    print("=" * 60)

    import asyncio

    async def async_example():
        async_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        wrap_messages_stream_async(tracer, async_client)

        async with async_client.messages.stream(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            model="claude-sonnet-4-5",
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
        print("\n")

        print(await stream.get_final_message())

    asyncio.run(async_example())
