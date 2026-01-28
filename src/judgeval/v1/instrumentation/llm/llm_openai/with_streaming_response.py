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
from judgeval.v1.instrumentation.llm.llm_openai.utils import openai_tokens_converter

if TYPE_CHECKING:
    from judgeval.v1.tracer import BaseTracer
    from openai import OpenAI, AsyncOpenAI
    from openai._response import (
        APIResponse,
        AsyncAPIResponse,
        ResponseContextManager,
        AsyncResponseContextManager,
    )


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
        ctx["usage"] = usage


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
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or 0
        cache_read = 0
        prompt_details = usage.get("prompt_tokens_details")
        if prompt_details:
            cache_read = prompt_details.get("cached_tokens") or 0

        prompt_tokens, completion_tokens, cache_read, cache_creation = (
            openai_tokens_converter(
                prompt_tokens,
                completion_tokens,
                cache_read,
                0,
                total_tokens,
            )
        )

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
            0,
        )
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_METADATA,
            safe_serialize(usage),
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
        for chunk in self._response.iter_text(chunk_size):
            yield chunk

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        for chunk in self._response.iter_bytes(chunk_size):
            yield chunk


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
        self._start_span()
        response = self._original.__enter__()
        return _TracedAPIResponse(response, self._ctx)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            self._original.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._end_span(exc_val)

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

    @dont_throw
    def _end_span(self, error: BaseException | None) -> None:
        span = self._ctx.get("span")
        if not span:
            return
        _finalize_span(self._ctx)
        if error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))
        span.end()


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
        self._start_span()
        response = await self._original.__aenter__()
        return _TracedAsyncAPIResponse(response, self._ctx)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            await self._original.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self._end_span(exc_val)

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

    @dont_throw
    def _end_span(self, error: BaseException | None) -> None:
        span = self._ctx.get("span")
        if not span:
            return
        _finalize_span(self._ctx)
        if error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))
        span.end()


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
