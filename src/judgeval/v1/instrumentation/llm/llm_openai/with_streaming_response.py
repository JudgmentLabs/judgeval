from __future__ import annotations
import json
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    AsyncIterator,
)

from opentelemetry.trace import Status, StatusCode
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize
from judgeval.utils.decorators.dont_throw import dont_throw
from judgeval.v1.instrumentation.llm.llm_openai.utils import (
    openai_tokens_converter,
    set_cost_attribute,
)

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream
    from openai._response import (
        APIResponse,
        AsyncAPIResponse,
        ResponseContextManager,
        AsyncResponseContextManager,
    )
    from openai.types.chat import ChatCompletion, ChatCompletionChunk


@dont_throw
def _process_sse_line(ctx: Dict[str, Any], line: str) -> None:
    if not line.startswith("data: ") or line == "data: [DONE]":
        return
    try:
        data = json.loads(line[6:])
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    _process_chunk_dict(ctx, data)


@dont_throw
def _process_chunk_dict(ctx: Dict[str, Any], data: Dict[str, Any]) -> None:
    model = data.get("model")
    if model:
        ctx["model"] = model

    choices = data.get("choices", [])
    if choices and len(choices) > 0:
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            ctx["accumulated_content"] = ctx.get("accumulated_content", "") + content

    usage = data.get("usage")
    if usage and isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        ctx["usage_dict"] = usage


@dont_throw
def _process_chunk(ctx: Dict[str, Any], chunk: ChatCompletionChunk) -> None:
    if chunk.model:
        ctx["model"] = chunk.model

    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            ctx["accumulated_content"] = (
                ctx.get("accumulated_content", "") + delta.content
            )

    if chunk.usage:
        ctx["usage"] = chunk.usage


@dont_throw
def _process_completion(ctx: Dict[str, Any], result: ChatCompletion) -> None:
    if result.model:
        ctx["model"] = result.model

    if result.choices and len(result.choices) > 0:
        message = result.choices[0].message
        if message and message.content:
            ctx["accumulated_content"] = message.content

    if result.usage:
        ctx["usage"] = result.usage


@dont_throw
def _process_json_response(ctx: Dict[str, Any], data: Dict[str, Any]) -> None:
    model = data.get("model")
    if model:
        ctx["model"] = model

    choices = data.get("choices", [])
    if choices and len(choices) > 0:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content") if isinstance(message, dict) else None
        if content:
            ctx["accumulated_content"] = content

    usage = data.get("usage")
    if usage and isinstance(usage, dict):
        ctx["usage_dict"] = usage


@dont_throw
def _finalize_span(ctx: Dict[str, Any]) -> None:
    span = ctx.get("span")
    if not span:
        return

    accumulated = ctx.get("accumulated_content", "")
    if accumulated:
        span.set_attribute(AttributeKeys.GEN_AI_COMPLETION, accumulated)

    model = ctx.get("model")
    if model:
        span.set_attribute(AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model)

    usage = ctx.get("usage")
    if usage:
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or 0
        cache_read = 0
        if usage.prompt_tokens_details:
            cache_read = usage.prompt_tokens_details.cached_tokens or 0

        set_cost_attribute(span, usage)

        prompt_tokens, completion_tokens, cache_read, _ = openai_tokens_converter(
            prompt_tokens, completion_tokens, cache_read, 0, total_tokens
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
        span.set_attribute(AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0)
        span.set_attribute(AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage))
        return

    usage_dict = ctx.get("usage_dict")
    if usage_dict:
        prompt_tokens = usage_dict.get("prompt_tokens") or 0
        completion_tokens = usage_dict.get("completion_tokens") or 0
        total_tokens = usage_dict.get("total_tokens") or 0
        cache_read = 0
        prompt_details = usage_dict.get("prompt_tokens_details")
        if prompt_details and isinstance(prompt_details, dict):
            cache_read = prompt_details.get("cached_tokens") or 0

        prompt_tokens, completion_tokens, cache_read, _ = openai_tokens_converter(
            prompt_tokens, completion_tokens, cache_read, 0, total_tokens
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
        span.set_attribute(AttributeKeys.JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS, 0)
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA, safe_serialize(usage_dict)
        )


class _TracedAPIResponse:
    __slots__ = ("_response", "_ctx")

    def __init__(self, response: APIResponse[Any], ctx: Dict[str, Any]) -> None:
        self._response = response
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    def iter_lines(self) -> Iterator[str]:
        for line in self._response.iter_lines():
            _process_sse_line(self._ctx, line)
            yield line

    def iter_text(self, chunk_size: int | None = None) -> Iterator[str]:
        return self._response.iter_text(chunk_size)

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        return self._response.iter_bytes(chunk_size)

    def parse(self, *, to: type | None = None) -> Any:
        result = self._response.parse(to=to) if to else self._response.parse()
        if hasattr(result, "__iter__") and hasattr(result, "response"):
            return _TracedStream(result, self._ctx)
        _process_completion(self._ctx, result)
        return result

    def read(self) -> bytes:
        return self._response.read()

    def text(self) -> str:
        return self._response.text()

    def json(self) -> object:
        result = self._response.json()
        if isinstance(result, dict):
            _process_json_response(self._ctx, result)
        return result

    def close(self) -> None:
        return self._response.close()


class _TracedAsyncAPIResponse:
    __slots__ = ("_response", "_ctx")

    def __init__(self, response: AsyncAPIResponse[Any], ctx: Dict[str, Any]) -> None:
        self._response = response
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    async def iter_lines(self) -> AsyncIterator[str]:
        async for line in self._response.iter_lines():
            _process_sse_line(self._ctx, line)
            yield line

    async def iter_text(self, chunk_size: int | None = None) -> AsyncIterator[str]:
        async for chunk in self._response.iter_text(chunk_size):
            yield chunk

    async def iter_bytes(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        async for chunk in self._response.iter_bytes(chunk_size):
            yield chunk

    async def parse(self, *, to: type | None = None) -> Any:
        result = (
            await self._response.parse(to=to) if to else await self._response.parse()
        )
        if hasattr(result, "__aiter__") and hasattr(result, "response"):
            return _TracedAsyncStream(result, self._ctx)
        _process_completion(self._ctx, result)
        return result

    async def read(self) -> bytes:
        return await self._response.read()

    async def text(self) -> str:
        return await self._response.text()

    async def json(self) -> object:
        result = await self._response.json()
        if isinstance(result, dict):
            _process_json_response(self._ctx, result)
        return result

    async def close(self) -> None:
        return await self._response.close()


class _TracedStream:
    __slots__ = ("_stream", "_ctx")

    def __init__(
        self, stream: Stream[ChatCompletionChunk], ctx: Dict[str, Any]
    ) -> None:
        self._stream = stream
        self._ctx = ctx

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        for chunk in self._stream:
            _process_chunk(self._ctx, chunk)
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> _TracedStream:
        self._stream.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        return self._stream.__exit__(exc_type, exc_val, exc_tb)


class _TracedAsyncStream:
    __slots__ = ("_stream", "_ctx")

    def __init__(
        self, stream: AsyncStream[ChatCompletionChunk], ctx: Dict[str, Any]
    ) -> None:
        self._stream = stream
        self._ctx = ctx

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self._stream:
            _process_chunk(self._ctx, chunk)
            yield chunk

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> _TracedAsyncStream:
        await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        return await self._stream.__aexit__(exc_type, exc_val, exc_tb)


class _TracedResponseContextManager:
    __slots__ = ("_original", "_tracer", "_kwargs", "_ctx")

    def __init__(
        self,
        original: ResponseContextManager[APIResponse[Any]],
        tracer: BaseTracer,
        kwargs: Dict[str, Any],
    ) -> None:
        self._original = original
        self._tracer = tracer
        self._kwargs = kwargs
        self._ctx: Dict[str, Any] = {}

    def __enter__(self) -> _TracedAPIResponse:
        response = self._original.__enter__()
        self._start_span()
        return _TracedAPIResponse(response, self._ctx)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        try:
            return self._original.__exit__(exc_type, exc_val, exc_tb)
        finally:
            _finalize_span(self._ctx)
            span = self._ctx.get("span")
            if span:
                if exc_val:
                    span.record_exception(exc_val)
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

    @dont_throw
    def _start_span(self) -> None:
        self._ctx["span"] = self._tracer.get_tracer().start_span(
            "OPENAI_API_CALL",
            attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"},
        )
        self._ctx["span"].set_attribute(
            AttributeKeys.GEN_AI_PROMPT, safe_serialize(self._kwargs)
        )
        model_name = self._kwargs.get("model", "")
        self._ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model_name
        )
        self._ctx["accumulated_content"] = ""


class _TracedAsyncResponseContextManager:
    __slots__ = ("_original", "_tracer", "_kwargs", "_ctx")

    def __init__(
        self,
        original: AsyncResponseContextManager[AsyncAPIResponse[Any]],
        tracer: BaseTracer,
        kwargs: Dict[str, Any],
    ) -> None:
        self._original = original
        self._tracer = tracer
        self._kwargs = kwargs
        self._ctx: Dict[str, Any] = {}

    async def __aenter__(self) -> _TracedAsyncAPIResponse:
        response = await self._original.__aenter__()
        self._start_span()
        return _TracedAsyncAPIResponse(response, self._ctx)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Any:
        try:
            return await self._original.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            _finalize_span(self._ctx)
            span = self._ctx.get("span")
            if span:
                if exc_val:
                    span.record_exception(exc_val)
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

    @dont_throw
    def _start_span(self) -> None:
        self._ctx["span"] = self._tracer.get_tracer().start_span(
            "OPENAI_API_CALL",
            attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "llm"},
        )
        self._ctx["span"].set_attribute(
            AttributeKeys.GEN_AI_PROMPT, safe_serialize(self._kwargs)
        )
        model_name = self._kwargs.get("model", "")
        self._ctx["span"].set_attribute(
            AttributeKeys.JUDGMENT_LLM_MODEL_NAME, model_name
        )
        self._ctx["accumulated_content"] = ""


def wrap_with_streaming_response_sync(tracer: BaseTracer, client: OpenAI) -> None:
    original_create = client.chat.completions.with_streaming_response.create

    def wrapped_create(*args: Any, **kwargs: Any) -> _TracedResponseContextManager:
        original_cm = original_create(*args, **kwargs)
        return _TracedResponseContextManager(original_cm, tracer, kwargs)

    setattr(client.chat.completions.with_streaming_response, "create", wrapped_create)


def wrap_with_streaming_response_async(tracer: BaseTracer, client: AsyncOpenAI) -> None:
    original_create = client.chat.completions.with_streaming_response.create

    def wrapped_create(*args: Any, **kwargs: Any) -> _TracedAsyncResponseContextManager:
        original_cm = original_create(*args, **kwargs)
        return _TracedAsyncResponseContextManager(original_cm, tracer, kwargs)

    setattr(client.chat.completions.with_streaming_response, "create", wrapped_create)
