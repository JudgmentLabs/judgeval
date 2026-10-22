from __future__ import annotations
import weakref
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
    cast,
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
from judgeval.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)
from judgeval.trace import BaseTracer

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


def _supports_stream_options() -> bool:
    try:
        import openai

        return version.parse(openai.__version__) >= version.parse("1.26.0")
    except Exception:
        return False


def _stream_options_unset(kwargs: Dict[str, Any]) -> bool:
    """True when the caller did not set ``stream_options``. SDK helpers such as
    ``chat.completions.stream()`` always pass the key, as the ``omit`` or
    ``NOT_GIVEN`` sentinel when it is unset."""
    if "stream_options" not in kwargs:
        return True
    import openai

    sentinels = tuple(
        t
        for t in (getattr(openai, "NotGiven", None), getattr(openai, "Omit", None))
        if isinstance(t, type)
    )
    return isinstance(kwargs["stream_options"], sentinels)


def _finish_stream_span(ctx: Dict[str, Any]) -> None:
    """End a streaming call's span once, keeping any text received so far
    (the stream may have been closed, dropped or failed before the end)."""
    span = ctx.get("span")
    if span is None or not span.is_recording():
        return
    if ctx.get("accumulated_content"):
        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, ctx["accumulated_content"])
    span.end()


class _TracedStream:
    """Iterates a ``Stream`` through the tracing hooks while keeping the SDK's
    stream API (``with``, ``.response``, ``.close()``) available. A stream that
    is dropped without being read to the end or closed ends its span when it is
    garbage-collected."""

    __slots__ = ("_stream", "_iterator", "_finish", "__weakref__")

    def __init__(
        self,
        stream: Any,
        iterator: Generator[Any, None, None],
        finish: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._iterator = iterator
        self._finish = finish
        weakref.finalize(self, finish)

    def __iter__(self) -> _TracedStream:
        return self

    def __next__(self) -> Any:
        return next(self._iterator)

    def __enter__(self) -> _TracedStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the stream and end the span, even if it was never read."""
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
        finally:
            try:
                self._iterator.close()
            except ValueError:
                # Another thread is reading; its loop ends the span once it
                # sees the closed stream.
                pass
            else:
                self._finish()

    def __getattr__(self, name: str) -> Any:
        if name in _TracedStream.__slots__:
            raise AttributeError(name)
        return getattr(self._stream, name)


class _AsyncTracedStream:
    """Async counterpart of ``_TracedStream`` for ``AsyncStream``."""

    __slots__ = ("_stream", "_iterator", "_finish", "__weakref__")

    def __init__(
        self,
        stream: Any,
        iterator: AsyncGenerator[Any, None],
        finish: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._iterator = iterator
        self._finish = finish
        weakref.finalize(self, finish)

    def __aiter__(self) -> _AsyncTracedStream:
        return self

    async def __anext__(self) -> Any:
        return await self._iterator.__anext__()

    async def __aenter__(self) -> _AsyncTracedStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the stream and end the span, even if it was never read."""
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                await close()
        finally:
            try:
                await self._iterator.aclose()
            except RuntimeError:
                # Another task is reading; its loop ends the span once it
                # sees the closed stream.
                pass
            else:
                self._finish()

    aclose = close

    def __getattr__(self, name: str) -> Any:
        if name in _AsyncTracedStream.__slots__:
            raise AttributeError(name)
        return getattr(self._stream, name)


class _TracedStreamManager:
    """Wraps the manager from ``chat.completions.stream()``. Leaving its block
    only closes the HTTP response, so also close the traced stream underneath
    to end its span."""

    __slots__ = ("_manager", "_stream")

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, *exc_info: Any) -> Any:
        try:
            return self._manager.__exit__(*exc_info)
        finally:
            raw_stream = getattr(self._stream, "_raw_stream", None)
            if isinstance(raw_stream, _TracedStream):
                raw_stream.close()

    def __getattr__(self, name: str) -> Any:
        if name in _TracedStreamManager.__slots__:
            raise AttributeError(name)
        return getattr(self._manager, name)


class _AsyncTracedStreamManager:
    """Async counterpart of ``_TracedStreamManager``."""

    __slots__ = ("_manager", "_stream")

    def __init__(self, manager: Any) -> None:
        self._manager = manager
        self._stream: Any = None

    async def __aenter__(self) -> Any:
        self._stream = await self._manager.__aenter__()
        return self._stream

    async def __aexit__(self, *exc_info: Any) -> Any:
        try:
            return await self._manager.__aexit__(*exc_info)
        finally:
            raw_stream = getattr(self._stream, "_raw_stream", None)
            if isinstance(raw_stream, _AsyncTracedStream):
                await raw_stream.close()

    def __getattr__(self, name: str) -> Any:
        if name in _AsyncTracedStreamManager.__slots__:
            raise AttributeError(name)
        return getattr(self._manager, name)


def wrap_chat_completions_stream_sync(client: OpenAI) -> None:
    original_func = getattr(client.chat.completions, "stream", None)
    if original_func is None:
        return

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        return _TracedStreamManager(result)

    setattr(
        client.chat.completions,
        "stream",
        mutable_wrap_sync(original_func, mutate_hook=mutate_hook),
    )


def wrap_chat_completions_stream_async(client: AsyncOpenAI) -> None:
    original_func = getattr(client.chat.completions, "stream", None)
    if original_func is None:
        return

    def mutate_hook(ctx: Dict[str, Any], result: Any) -> Any:
        return _AsyncTracedStreamManager(result)

    setattr(
        client.chat.completions,
        "stream",
        mutable_wrap_sync(original_func, mutate_hook=mutate_hook),
    )


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
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
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

            (
                prompt_tokens,
                completion_tokens,
                cache_read,
                _,
                _,
                _,
            ) = openai_tokens_converter(
                prompt_tokens,
                completion_tokens,
                cache_read,
                0,
                0,
                0,
                usage_data.total_tokens,
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
    original_func: Callable[..., Iterator[ChatCompletionChunk]],
) -> Callable[..., Iterator[ChatCompletionChunk]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_kwargs_hook(ctx: Dict[str, Any], kwargs: Any) -> Any:
        if _stream_options_unset(kwargs) and _supports_stream_options():
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

                (
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    _,
                    _,
                    _,
                ) = openai_tokens_converter(
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    0,
                    0,
                    0,
                    chunk.usage.total_tokens,
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
            _finish_stream_span(ctx)

        wrapped_generator = immutable_wrap_sync_iterator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return _TracedStream(
            result,
            cast("Generator[ChatCompletionChunk, None, None]", wrapped_generator()),
            lambda: _finish_stream_span(ctx),
        )

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
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
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

            (
                prompt_tokens,
                completion_tokens,
                cache_read,
                _,
                _,
                _,
            ) = openai_tokens_converter(
                prompt_tokens,
                completion_tokens,
                cache_read,
                0,
                0,
                0,
                usage_data.total_tokens,
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
    original_func: Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]],
) -> Callable[..., Awaitable[AsyncIterator[ChatCompletionChunk]]]:
    def pre_hook(ctx: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        ctx["span"] = BaseTracer.start_span(
            "OPENAI_API_CALL", {AttributeKeys.JUDGMENT_SPAN_KIND: "llm"}
        )
        ctx["span"].set_attribute(AttributeKeys.GEN_AI_PROMPT, safe_serialize(kwargs))
        ctx["model_name"] = kwargs.get("model", "")
        ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, ctx["model_name"]
        )
        ctx["accumulated_content"] = ""

    def mutate_kwargs_hook(ctx: Dict[str, Any], kwargs: Any) -> Any:
        if _stream_options_unset(kwargs) and _supports_stream_options():
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

                (
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    _,
                    _,
                    _,
                ) = openai_tokens_converter(
                    prompt_tokens,
                    completion_tokens,
                    cache_read,
                    0,
                    0,
                    0,
                    chunk.usage.total_tokens,
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
            _finish_stream_span(ctx)

        wrapped_generator = immutable_wrap_async_iterator(
            traced_generator,
            yield_hook=yield_hook,
            post_hook=post_hook_inner,
            error_hook=error_hook_inner,
            finally_hook=finally_hook_inner,
        )

        return _AsyncTracedStream(
            result,
            cast("AsyncGenerator[ChatCompletionChunk, None]", wrapped_generator()),
            lambda: _finish_stream_span(ctx),
        )

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
