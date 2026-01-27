from __future__ import annotations

import asyncio
import contextvars
from collections.abc import (
    Generator as ABCGenerator,
    AsyncGenerator as ABCAsyncGenerator,
)
from types import TracebackType
from typing import Any, Callable, Coroutine, Optional, TypeVar, overload

from opentelemetry.trace import Span, Status, StatusCode

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.tracer.isolated.tracer import JudgmentIsolatedTracer

R = TypeVar("R")


def _serialize(serializer: Callable[[Any], str], value: Any) -> Any:
    return value if isinstance(value, (str, int, float, bool)) else serializer(value)


class _ObservedGeneratorBase:
    __slots__ = (
        "_generator",
        "_span",
        "_serializer",
        "_tracer",
        "_context",
        "_closed",
        "_disable_generator_yield_span",
    )

    _generator: Any
    _span: Span
    _serializer: Callable[[Any], str]
    _tracer: JudgmentIsolatedTracer
    _context: contextvars.Context
    _closed: bool
    _disable_generator_yield_span: bool

    def __init__(
        self,
        generator: Any,
        span: Span,
        serializer: Callable[[Any], str],
        tracer: JudgmentIsolatedTracer,
        context: contextvars.Context,
        disable_generator_yield_span: bool = False,
    ) -> None:
        self._generator = generator
        self._span = span
        self._serializer = serializer
        self._tracer = tracer
        self._context = context
        self._closed = False
        self._disable_generator_yield_span = disable_generator_yield_span

    def _emit_yield_span(self, item: Any) -> None:
        if self._disable_generator_yield_span:
            return
        with self._tracer.use_span(self._span):
            span_name = str(getattr(self._span, "name", "generator_item"))
            with self._tracer.start_as_current_span(
                span_name,
                attributes={AttributeKeys.JUDGMENT_SPAN_KIND: "generator_item"},
            ) as child_span:
                child_span.set_attribute(
                    AttributeKeys.JUDGMENT_OUTPUT,
                    _serialize(self._serializer, item),
                )

    def _record_error(self, exc: BaseException) -> None:
        self._span.record_exception(exc)
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        self._finish()

    def _finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._span.set_attribute(AttributeKeys.JUDGMENT_SPAN_KIND, "generator")
        self._span.end()

    def __del__(self) -> None:
        self._finish()


class _ObservedSyncGenerator(_ObservedGeneratorBase, ABCGenerator[Any, Any, Any]):
    __slots__ = ()

    def __iter__(self) -> _ObservedSyncGenerator:
        return self

    def __next__(self) -> Any:
        return self.send(None)

    def send(self, value: Any) -> Any:
        if self._closed:
            raise StopIteration
        try:
            item = self._context.run(self._generator.send, value)
            self._emit_yield_span(item)
            return item
        except StopIteration:
            self._finish()
            raise
        except Exception as e:
            self._record_error(e)
            raise

    @overload
    def throw(
        self,
        __typ: type[BaseException],
        __val: object = ...,
        __tb: Optional[TracebackType] = ...,
    ) -> Any: ...

    @overload
    def throw(
        self,
        __typ: BaseException,
        __val: None = ...,
        __tb: Optional[TracebackType] = ...,
    ) -> Any: ...

    def throw(
        self,
        __typ: type[BaseException] | BaseException,
        __val: object = None,
        __tb: Optional[TracebackType] = None,
    ) -> Any:
        if self._closed:
            raise StopIteration
        try:
            if isinstance(__typ, type):
                item = self._context.run(self._generator.throw, __typ, __val, __tb)
            else:
                item = self._context.run(self._generator.throw, __typ, None, __tb)
            self._emit_yield_span(item)
            return item
        except StopIteration:
            self._finish()
            raise
        except Exception as e:
            self._record_error(e)
            raise

    def close(self) -> None:
        try:
            self._generator.close()
        finally:
            self._finish()


class _ObservedAsyncGenerator(_ObservedGeneratorBase, ABCAsyncGenerator[Any, Any]):
    __slots__ = ()

    def _create_task(self, coro: Coroutine[Any, Any, R]) -> asyncio.Task[R]:
        try:
            return asyncio.create_task(coro, context=self._context)
        except TypeError:
            return self._context.run(lambda: asyncio.create_task(coro))

    def __aiter__(self) -> _ObservedAsyncGenerator:
        return self

    async def __anext__(self) -> Any:
        return await self.asend(None)

    async def asend(self, value: Any) -> Any:
        if self._closed:
            raise StopAsyncIteration
        try:
            item = await self._create_task(self._generator.asend(value))
            self._emit_yield_span(item)
            return item
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as e:
            self._record_error(e)
            raise

    @overload
    async def athrow(
        self,
        __typ: type[BaseException],
        __val: object = ...,
        __tb: Optional[TracebackType] = ...,
    ) -> Any: ...

    @overload
    async def athrow(
        self,
        __typ: BaseException,
        __val: None = ...,
        __tb: Optional[TracebackType] = ...,
    ) -> Any: ...

    async def athrow(
        self,
        __typ: type[BaseException] | BaseException,
        __val: object = None,
        __tb: Optional[TracebackType] = None,
    ) -> Any:
        if self._closed:
            raise StopAsyncIteration
        try:
            if isinstance(__typ, type):
                item = await self._create_task(
                    self._generator.athrow(__typ, __val, __tb)
                )
            else:
                item = await self._create_task(
                    self._generator.athrow(__typ, None, __tb)
                )
            self._emit_yield_span(item)
            return item
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as e:
            self._record_error(e)
            raise

    async def aclose(self) -> None:
        try:
            await self._generator.aclose()
        finally:
            self._finish()
