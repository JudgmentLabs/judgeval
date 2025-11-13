from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    AsyncGenerator,
)

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.v1.instrumentation.llm.llm_anthropic.messages import (
    _extract_anthropic_tokens,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.lib.streaming import (
        MessageStreamManager,
        AsyncMessageStreamManager,
        MessageStream,
        AsyncMessageStream,
    )


def wrap_messages_stream_sync(tracer: BaseTracer, client: Anthropic) -> None:
    original_func = client.messages.stream

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))

        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
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

            def post_hook_inner(inner_ctx: Dict[str, Any]) -> None:
                pass

            def error_hook_inner(inner_ctx: Dict[str, Any], error: Exception) -> None:
                span = ctx.get("span")
                if span:
                    span.record_exception(error)

            def finally_hook_inner(inner_ctx: Dict[str, Any]) -> None:
                pass

            wrapped_text_stream = immutable_wrap_sync_iterator(
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
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

                stream: MessageStream | None = ctx.get("stream")
                if stream:
                    try:
                        final_message = stream.get_final_message()
                        if final_message.usage:
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_anthropic_tokens(final_message.usage)
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                                cache_creation,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_METADATA,
                                safe_serialize(final_message.usage),
                            )

                        span.set_attribute(
                            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, final_message.model
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


def wrap_messages_stream_async(tracer: BaseTracer, client: AsyncAnthropic) -> None:
    original_func = client.messages.stream

    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = tracer.get_tracer().start_span(
            "ANTHROPIC_API_CALL", attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )

        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))

        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
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

            wrapped_text_stream = immutable_wrap_async_iterator(
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
                span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

                stream: AsyncMessageStream | None = ctx.get("stream")
                if stream:
                    try:
                        final_message = await stream.get_final_message()
                        if final_message.usage:
                            (
                                prompt_tokens,
                                completion_tokens,
                                cache_read,
                                cache_creation,
                            ) = _extract_anthropic_tokens(final_message.usage)
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS,
                                prompt_tokens,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS,
                                completion_tokens,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS,
                                cache_read,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS,
                                cache_creation,
                            )
                            span.set_attribute(
                                AttributeKeys.JUDGMENT_USAGE_METADATA,
                                safe_serialize(final_message.usage),
                            )

                        span.set_attribute(
                            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, final_message.model
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
