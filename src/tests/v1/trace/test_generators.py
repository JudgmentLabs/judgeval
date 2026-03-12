"""Tests for _ObservedSyncGenerator and _ObservedAsyncGenerator."""

from __future__ import annotations

import asyncio
import contextvars
from unittest.mock import MagicMock

import pytest

from judgeval.utils.serialize import safe_serialize
from judgeval.v1.trace.generators import _ObservedSyncGenerator, _ObservedAsyncGenerator


def _make_observed_sync(gen, *, disable_yield_span=False):
    span = MagicMock()
    span.is_recording.return_value = True
    span.name = "test_gen"
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__ = MagicMock(
        return_value=MagicMock()
    )
    tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
    ctx = contextvars.copy_context()
    return _ObservedSyncGenerator(
        gen, span, safe_serialize, tracer, ctx, disable_yield_span
    ), span


class TestSyncGenerator:
    def test_yields_all_values(self):
        def gen():
            yield 1
            yield 2
            yield 3

        observed, span = _make_observed_sync(gen())
        assert list(observed) == [1, 2, 3]
        span.end.assert_called_once()

    def test_close_ends_span(self):
        def gen():
            yield 1
            yield 2

        observed, span = _make_observed_sync(gen())
        next(observed)
        observed.close()
        span.end.assert_called_once()

    def test_send_after_close_raises_stop(self):
        def gen():
            yield 1

        observed, _ = _make_observed_sync(gen())
        observed.close()
        with pytest.raises(StopIteration):
            observed.send(None)

    def test_exception_records_error(self):
        def gen():
            yield 1
            raise RuntimeError("fail")

        observed, span = _make_observed_sync(gen())
        assert next(observed) == 1
        with pytest.raises(RuntimeError, match="fail"):
            next(observed)
        span.record_exception.assert_called_once()
        span.end.assert_called_once()

    def test_throw_propagates(self):
        def gen():
            try:
                yield 1
            except ValueError:
                yield "caught"

        observed, _ = _make_observed_sync(gen())
        next(observed)
        result = observed.throw(ValueError, ValueError("test"))
        assert result == "caught"

    def test_disable_yield_span(self):
        def gen():
            yield 1

        observed, span = _make_observed_sync(gen(), disable_yield_span=True)
        list(observed)
        # Tracer should not create child spans when disabled


class TestAsyncGenerator:
    def test_yields_all_values(self):
        async def gen():
            yield "a"
            yield "b"

        span = MagicMock()
        span.is_recording.return_value = True
        span.name = "async_gen"
        tracer = MagicMock()
        tracer.start_as_current_span.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        tracer.start_as_current_span.return_value.__exit__ = MagicMock(
            return_value=False
        )
        ctx = contextvars.copy_context()
        observed = _ObservedAsyncGenerator(
            gen(), span, safe_serialize, tracer, ctx, False
        )

        async def consume():
            return [item async for item in observed]

        result = asyncio.run(consume())
        assert result == ["a", "b"]
        span.end.assert_called_once()

    def test_async_exception_records_error(self):
        async def gen():
            yield 1
            raise ValueError("async-fail")

        span = MagicMock()
        span.is_recording.return_value = True
        span.name = "async_gen"
        tracer = MagicMock()
        ctx = contextvars.copy_context()
        observed = _ObservedAsyncGenerator(
            gen(), span, safe_serialize, tracer, ctx, False
        )

        async def consume():
            return [item async for item in observed]

        with pytest.raises(ValueError, match="async-fail"):
            asyncio.run(consume())

        span.record_exception.assert_called_once()

    def test_aclose_ends_span(self):
        async def gen():
            yield 1
            yield 2

        span = MagicMock()
        span.is_recording.return_value = True
        span.name = "async_gen"
        tracer = MagicMock()
        ctx = contextvars.copy_context()
        observed = _ObservedAsyncGenerator(
            gen(), span, safe_serialize, tracer, ctx, False
        )

        async def run():
            await observed.__anext__()
            await observed.aclose()

        asyncio.run(run())
        span.end.assert_called_once()

    def test_send_after_close_raises_stop(self):
        async def gen():
            yield 1

        span = MagicMock()
        span.is_recording.return_value = True
        span.name = "async_gen"
        tracer = MagicMock()
        ctx = contextvars.copy_context()
        observed = _ObservedAsyncGenerator(
            gen(), span, safe_serialize, tracer, ctx, False
        )

        async def run():
            await observed.aclose()
            with pytest.raises(StopAsyncIteration):
                await observed.asend(None)

        asyncio.run(run())
