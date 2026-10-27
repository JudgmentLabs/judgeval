from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.utils.wrappers import (
    mutable_wrap_sync,
    immutable_wrap_sync_iterator,
    immutable_wrap_async_iterator,
)
from judgeval.instrumentation.llm.llm_anthropic.messages import _record_message
from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


def _start(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
    ctx["span"] = BaseTracer.start_span(
        "ANTHROPIC_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
    )
    ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
    ctx["span"].set_attribute(
        AttributeKeys.JUDGMENT_LLM_MODEL_NAME, kwargs.get("model", "")
    )


@dont_throw
def _error(ctx: Dict[str, Any], error: BaseException) -> None:
    if ctx.get("ended"):
        return
    ctx["failed"] = True
    span = ctx.get("span")
    if span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))


@dont_throw
def _finish(ctx: Dict[str, Any]) -> None:
    span = ctx.get("span")
    if not span or ctx.get("ended"):
        return
    ctx["ended"] = True
    try:
        stream = ctx.get("stream")
        if stream is not None and ctx.get("started"):
            # Reading the snapshot never drains the stream. get_final_message()
            # does, even on early context exit, so instrumentation must not call it.
            _record_message(
                span,
                stream.current_message_snapshot,
                complete=bool(ctx.get("complete")) and not ctx.get("failed", False),
            )
    finally:
        span.end()


@dont_throw
def _instrument_stream(ctx: Dict[str, Any], stream: Any, *, is_async: bool) -> None:
    ctx["stream"] = stream
    # All SDK consumption paths (events, text_stream, until_done and final-message
    # helpers) share this iterator. Return the original stream and events intact.
    iterator = stream._iterator

    def on_event(inner_ctx: Dict[str, Any], event: Any) -> None:
        ctx["started"] = True
        if event.type == "message_stop":
            ctx["complete"] = True

    if is_async:

        async def events_async():
            try:
                async for event in iterator:
                    yield event
            except BaseException as error:
                _error(ctx, error)
                raise

        stream._iterator = immutable_wrap_async_iterator(
            events_async, yield_hook=on_event
        )()
    else:

        def events_sync():
            try:
                yield from iterator
            except BaseException as error:
                _error(ctx, error)
                raise

        stream._iterator = immutable_wrap_sync_iterator(
            events_sync, yield_hook=on_event
        )()


def _creation_error(ctx: Dict[str, Any], error: Exception) -> None:
    _error(ctx, error)
    _finish(ctx)


def wrap_messages_stream_sync(client: Anthropic) -> None:
    def wrap_manager(ctx: Dict[str, Any], manager: Any) -> Any:
        class WrappedMessageStreamManager:
            def __enter__(self):
                try:
                    stream = manager.__enter__()
                except BaseException as error:
                    _error(ctx, error)
                    _finish(ctx)
                    raise
                _instrument_stream(ctx, stream, is_async=False)
                return stream

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_val is not None:
                    _error(ctx, exc_val)
                try:
                    return manager.__exit__(exc_type, exc_val, exc_tb)
                except BaseException as error:
                    _error(ctx, error)
                    raise
                finally:
                    _finish(ctx)

            def __getattr__(self, name):
                return getattr(manager, name)

        return WrappedMessageStreamManager()

    setattr(
        client.messages,
        "stream",
        mutable_wrap_sync(
            client.messages.stream,
            pre_hook=_start,
            mutate_hook=wrap_manager,
            error_hook=_creation_error,
        ),
    )


def wrap_messages_stream_async(client: AsyncAnthropic) -> None:
    def wrap_manager(ctx: Dict[str, Any], manager: Any) -> Any:
        class WrappedAsyncMessageStreamManager:
            async def __aenter__(self):
                try:
                    stream = await manager.__aenter__()
                except BaseException as error:
                    _error(ctx, error)
                    _finish(ctx)
                    raise
                _instrument_stream(ctx, stream, is_async=True)
                return stream

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                if exc_val is not None:
                    _error(ctx, exc_val)
                try:
                    return await manager.__aexit__(exc_type, exc_val, exc_tb)
                except BaseException as error:
                    _error(ctx, error)
                    raise
                finally:
                    _finish(ctx)

            def __getattr__(self, name):
                return getattr(manager, name)

        return WrappedAsyncMessageStreamManager()

    setattr(
        client.messages,
        "stream",
        mutable_wrap_sync(
            client.messages.stream,
            pre_hook=_start,
            mutate_hook=wrap_manager,
            error_hook=_creation_error,
        ),
    )
