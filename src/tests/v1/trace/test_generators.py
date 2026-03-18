from __future__ import annotations

import contextvars

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.v1.trace.exporters.noop_judgment_span_exporter import (
    NoOpJudgmentSpanExporter,
)
from judgeval.v1.trace.generators import _ObservedSyncGenerator, _ObservedAsyncGenerator
from judgeval.utils.serialize import safe_serialize


class _CapturingExporter(NoOpJudgmentSpanExporter):
    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


@pytest.fixture
def otel_provider():
    exp = _CapturingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return provider, exp


def _make_sync_generator(items, provider, *, disable_yield_span=False, raise_at=None):
    otel_tracer = provider.get_tracer("test")
    span = otel_tracer.start_span("root-gen")

    def raw_gen():
        for i, item in enumerate(items):
            if raise_at is not None and i == raise_at:
                raise ValueError(f"error at {i}")
            yield item

    return _ObservedSyncGenerator(
        raw_gen(),
        span,
        safe_serialize,
        otel_tracer,
        contextvars.copy_context(),
        disable_yield_span,
    )


async def _make_async_generator(
    items, provider, *, disable_yield_span=False, raise_at=None
):
    otel_tracer = provider.get_tracer("test")
    span = otel_tracer.start_span("root-async-gen")

    async def raw_agen():
        for i, item in enumerate(items):
            if raise_at is not None and i == raise_at:
                raise ValueError(f"error at {i}")
            yield item

    return _ObservedAsyncGenerator(
        raw_agen(),
        span,
        safe_serialize,
        otel_tracer,
        contextvars.copy_context(),
        disable_yield_span,
    )


class TestSyncGenerator:
    def test_yields_all_items(self, otel_provider):
        provider, _ = otel_provider
        gen = _make_sync_generator([1, 2, 3], provider)
        assert list(gen) == [1, 2, 3]

    def test_span_ends_after_exhaustion(self, otel_provider):
        provider, exp = otel_provider
        gen = _make_sync_generator(["a", "b"], provider)
        list(gen)
        root = next(
            (
                s
                for s in exp.spans
                if s.name == "root-gen"
                and s.attributes.get("judgment.span_kind") == "generator"
            ),
            None,
        )
        assert root is not None
        assert root.end_time is not None

    def test_emits_yield_spans(self, otel_provider):
        provider, exp = otel_provider
        gen = _make_sync_generator([1, 2], provider)
        list(gen)
        assert any(
            s.name == "root-gen"
            and s.attributes.get("judgment.span_kind") == "generator_item"
            for s in exp.spans
        )

    def test_no_yield_spans_when_disabled(self, otel_provider):
        provider, exp = otel_provider
        gen = _make_sync_generator([1, 2], provider, disable_yield_span=True)
        list(gen)
        item_spans = [
            s
            for s in exp.spans
            if s.attributes.get("judgment.span_kind") == "generator_item"
        ]
        assert len(item_spans) == 0

    def test_exception_sets_error_status(self, otel_provider):
        provider, exp = otel_provider
        gen = _make_sync_generator([1, 2], provider, raise_at=1)
        with pytest.raises(ValueError):
            list(gen)
        root = next(
            (
                s
                for s in exp.spans
                if s.name == "root-gen"
                and s.attributes.get("judgment.span_kind") == "generator"
            ),
            None,
        )
        assert root is not None
        assert root.status.status_code.name == "ERROR"

    def test_stop_iteration_after_close(self, otel_provider):
        provider, _ = otel_provider
        gen = _make_sync_generator([1], provider)
        gen.close()
        with pytest.raises(StopIteration):
            next(gen)

    def test_send_value(self, otel_provider):
        provider, _ = otel_provider
        otel_tracer = provider.get_tracer("test")
        span = otel_tracer.start_span("root")

        def echo():
            v = yield "start"
            yield v

        gen = _ObservedSyncGenerator(
            echo(), span, safe_serialize, otel_tracer, contextvars.copy_context()
        )
        first = next(gen)
        assert first == "start"
        second = gen.send("echo-back")
        assert second == "echo-back"


class TestAsyncGenerator:
    @pytest.mark.asyncio
    async def test_yields_all_items(self, otel_provider):
        provider, _ = otel_provider
        gen = await _make_async_generator([10, 20, 30], provider)
        result = []
        async for item in gen:
            result.append(item)
        assert result == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_span_ends_after_exhaustion(self, otel_provider):
        provider, exp = otel_provider
        gen = await _make_async_generator(["x"], provider)
        async for _ in gen:
            pass
        root = next(
            (
                s
                for s in exp.spans
                if s.name == "root-async-gen"
                and s.attributes.get("judgment.span_kind") == "generator"
            ),
            None,
        )
        assert root is not None
        assert root.end_time is not None

    @pytest.mark.asyncio
    async def test_exception_sets_error_status(self, otel_provider):
        provider, exp = otel_provider
        gen = await _make_async_generator([1, 2], provider, raise_at=0)
        with pytest.raises(ValueError):
            async for _ in gen:
                pass
        root = next(
            (
                s
                for s in exp.spans
                if s.name == "root-async-gen"
                and s.attributes.get("judgment.span_kind") == "generator"
            ),
            None,
        )
        assert root is not None
        assert root.status.status_code.name == "ERROR"

    @pytest.mark.asyncio
    async def test_stop_async_iteration_after_close(self, otel_provider):
        provider, _ = otel_provider
        gen = await _make_async_generator([1], provider)
        await gen.aclose()
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
