"""Tests for BaseTracer static API — spans, attributes, observe decorator, context propagation."""

from __future__ import annotations

import asyncio

import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.trace.base_tracer import BaseTracer


# ------------------------------------------------------------------ #
#  Span creation                                                      #
# ------------------------------------------------------------------ #


class TestSpanCreation:
    def test_span_context_manager_creates_and_ends_span(
        self, tracer, collecting_exporter
    ):
        with BaseTracer.span("my-span"):
            pass

        assert any(s.name == "my-span" for s in collecting_exporter.spans)

    def test_span_records_exception_and_reraises(self, tracer, collecting_exporter):
        with pytest.raises(RuntimeError, match="kaboom"):
            with BaseTracer.span("err-span"):
                raise RuntimeError("kaboom")

        span = next(s for s in collecting_exporter.spans if s.name == "err-span")
        assert span.status.status_code.name == "ERROR"
        assert any(e.name == "exception" for e in span.events)

    def test_nested_spans_share_trace_id(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("parent"):
            with BaseTracer.start_as_current_span("child"):
                pass

        parent = next(s for s in collecting_exporter.spans if s.name == "parent")
        child = next(s for s in collecting_exporter.spans if s.name == "child")
        assert parent.context.trace_id == child.context.trace_id
        assert parent.context.span_id != child.context.span_id
        assert child.parent.span_id == parent.context.span_id


# ------------------------------------------------------------------ #
#  Attributes                                                         #
# ------------------------------------------------------------------ #


class TestAttributes:
    def test_set_attribute_on_current_span(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("attr-span"):
            BaseTracer.set_attribute("my.key", "my-value")

        span = next(s for s in collecting_exporter.spans if s.name == "attr-span")
        assert span.attributes["my.key"] == "my-value"

    def test_set_input_and_output(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("io-span"):
            BaseTracer.set_input({"question": "what?"})
            BaseTracer.set_output("answer")

        span = next(s for s in collecting_exporter.spans if s.name == "io-span")
        assert "question" in span.attributes[AttributeKeys.JUDGMENT_INPUT]
        assert "answer" in span.attributes[AttributeKeys.JUDGMENT_OUTPUT]

    def test_set_span_kind_helpers(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("llm"):
            BaseTracer.set_llm_span()
        with BaseTracer.start_as_current_span("tool"):
            BaseTracer.set_tool_span()

        llm = next(s for s in collecting_exporter.spans if s.name == "llm")
        tool = next(s for s in collecting_exporter.spans if s.name == "tool")
        assert llm.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "llm"
        assert tool.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"

    def test_set_attribute_noop_outside_span(self, tracer):
        # Should not raise
        BaseTracer.set_attribute("key", "value")

    def test_record_llm_metadata(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("meta"):
            BaseTracer.recordLLMMetadata(
                {"model": "gpt-4", "output_tokens": 100, "total_cost_usd": 0.01}
            )

        span = next(s for s in collecting_exporter.spans if s.name == "meta")
        assert span.attributes[AttributeKeys.JUDGMENT_LLM_MODEL_NAME] == "gpt-4"
        assert span.attributes[AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS] == 100
        assert span.attributes[AttributeKeys.JUDGMENT_USAGE_TOTAL_COST_USD] == 0.01


# ------------------------------------------------------------------ #
#  @observe decorator                                                  #
# ------------------------------------------------------------------ #


class TestObserveDecorator:
    def test_observe_sync_function(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

        span = next(s for s in collecting_exporter.spans if s.name == "add")
        assert AttributeKeys.JUDGMENT_INPUT in span.attributes
        assert AttributeKeys.JUDGMENT_OUTPUT in span.attributes

    def test_observe_async_function(self, tracer, collecting_exporter):
        @BaseTracer.observe
        async def greet(name):
            return f"hello {name}"

        result = asyncio.run(greet("world"))
        assert result == "hello world"

        span = next(s for s in collecting_exporter.spans if s.name == "greet")
        assert AttributeKeys.JUDGMENT_INPUT in span.attributes
        assert "hello world" in span.attributes[AttributeKeys.JUDGMENT_OUTPUT]

    def test_observe_with_custom_name(self, tracer, collecting_exporter):
        @BaseTracer.observe(span_name="custom-name")
        def fn():
            return 42

        fn()
        assert any(s.name == "custom-name" for s in collecting_exporter.spans)

    def test_observe_records_exception(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def fail():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            fail()

        span = next(s for s in collecting_exporter.spans if s.name == "fail")
        assert span.status.status_code.name == "ERROR"

    def test_observe_no_input_recording(self, tracer, collecting_exporter):
        @BaseTracer.observe(record_input=False)
        def fn(secret):
            return "ok"

        fn("password123")
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert AttributeKeys.JUDGMENT_INPUT not in span.attributes

    def test_observe_no_output_recording(self, tracer, collecting_exporter):
        @BaseTracer.observe(record_output=False)
        def fn():
            return "secret"

        fn()
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert AttributeKeys.JUDGMENT_OUTPUT not in span.attributes

    def test_observe_sync_generator(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def gen():
            yield 1
            yield 2
            yield 3

        result = list(gen())
        assert result == [1, 2, 3]

        # Parent generator span + child yield spans
        gen_spans = [s for s in collecting_exporter.spans if s.name == "gen"]
        assert len(gen_spans) >= 1
        parent = next(
            s
            for s in gen_spans
            if s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
        )
        assert parent is not None

    def test_observe_async_generator(self, tracer, collecting_exporter):
        @BaseTracer.observe
        async def agen():
            yield "a"
            yield "b"

        async def consume():
            return [item async for item in agen()]

        result = asyncio.run(consume())
        assert result == ["a", "b"]

    def test_observe_generator_with_exception(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def bad_gen():
            yield 1
            raise RuntimeError("gen-error")

        with pytest.raises(RuntimeError, match="gen-error"):
            list(bad_gen())

        gen_spans = [s for s in collecting_exporter.spans if s.name == "bad_gen"]
        errored = [s for s in gen_spans if s.status.status_code.name == "ERROR"]
        assert len(errored) >= 1


# ------------------------------------------------------------------ #
#  Context propagation (customer_id / session_id)                     #
# ------------------------------------------------------------------ #


class TestContextPropagation:
    def test_customer_id_propagates_to_children(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("root"):
            BaseTracer.set_customer_id("cust-42")
            with BaseTracer.start_as_current_span("child"):
                pass

        child = next(s for s in collecting_exporter.spans if s.name == "child")
        assert child.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-42"

    def test_session_id_propagates_to_children(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("root"):
            BaseTracer.set_session_id("sess-99")
            with BaseTracer.start_as_current_span("child"):
                pass

        child = next(s for s in collecting_exporter.spans if s.name == "child")
        assert child.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-99"

    def test_ids_do_not_leak_across_traces(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("trace1"):
            BaseTracer.set_customer_id("cust-1")

        with BaseTracer.start_as_current_span("trace2"):
            pass

        trace2 = next(s for s in collecting_exporter.spans if s.name == "trace2")
        assert AttributeKeys.JUDGMENT_CUSTOMER_ID not in (trace2.attributes or {})

    def test_customer_and_session_together(self, tracer, collecting_exporter):
        with BaseTracer.start_as_current_span("root"):
            BaseTracer.set_customer_id("cust-x")
            BaseTracer.set_session_id("sess-y")
            with BaseTracer.start_as_current_span("inner"):
                pass

        inner = next(s for s in collecting_exporter.spans if s.name == "inner")
        assert inner.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-x"
        assert inner.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-y"
