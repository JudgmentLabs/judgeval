from __future__ import annotations

import pytest

from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.v1.trace.base_tracer import BaseTracer


class TestObserveSync:
    def test_creates_span_with_function_name(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def my_func():
            return 42

        my_func()
        assert any(s.name == "my_func" for s in collecting_exporter.spans)

    def test_records_input(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def add(a, b):
            return a + b

        add(1, 2)
        span = next(s for s in collecting_exporter.spans if s.name == "add")
        assert "a" in span.attributes[AttributeKeys.JUDGMENT_INPUT]

    def test_records_output(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def greet(name):
            return f"hello {name}"

        greet("world")
        span = next(s for s in collecting_exporter.spans if s.name == "greet")
        assert "hello world" in span.attributes[AttributeKeys.JUDGMENT_OUTPUT]

    def test_span_name_override(self, tracer, collecting_exporter):
        @BaseTracer.observe(span_name="custom-name")
        def fn():
            return 1

        fn()
        assert any(s.name == "custom-name" for s in collecting_exporter.spans)

    def test_record_input_false(self, tracer, collecting_exporter):
        @BaseTracer.observe(record_input=False)
        def fn(x):
            return x

        fn("secret")
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert AttributeKeys.JUDGMENT_INPUT not in span.attributes

    def test_record_output_false(self, tracer, collecting_exporter):
        @BaseTracer.observe(record_output=False)
        def fn():
            return "secret"

        fn()
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert AttributeKeys.JUDGMENT_OUTPUT not in span.attributes

    def test_exception_recorded_and_reraised(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def boom():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            boom()

        span = next(s for s in collecting_exporter.spans if s.name == "boom")
        assert span.status.status_code.name == "ERROR"

    def test_span_type_attribute(self, tracer, collecting_exporter):
        @BaseTracer.observe(span_type="tool")
        def my_tool():
            return "ok"

        my_tool()
        span = next(s for s in collecting_exporter.spans if s.name == "my_tool")
        assert span.attributes[AttributeKeys.JUDGMENT_SPAN_KIND] == "tool"

    def test_returns_correct_value(self, tracer):
        @BaseTracer.observe
        def double(x):
            return x * 2

        assert double(5) == 10


class TestObserveAsync:
    @pytest.mark.asyncio
    async def test_async_creates_span(self, tracer, collecting_exporter):
        @BaseTracer.observe
        async def async_fn():
            return "result"

        await async_fn()
        assert any(s.name == "async_fn" for s in collecting_exporter.spans)

    @pytest.mark.asyncio
    async def test_async_records_output(self, tracer, collecting_exporter):
        @BaseTracer.observe
        async def fetch():
            return "data"

        await fetch()
        span = next(s for s in collecting_exporter.spans if s.name == "fetch")
        assert "data" in span.attributes[AttributeKeys.JUDGMENT_OUTPUT]

    @pytest.mark.asyncio
    async def test_async_exception_recorded(self, tracer, collecting_exporter):
        @BaseTracer.observe
        async def bad():
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError):
            await bad()

        span = next(s for s in collecting_exporter.spans if s.name == "bad")
        assert span.status.status_code.name == "ERROR"


class TestObserveGenerator:
    def test_generator_wrapped(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def gen():
            yield 1
            yield 2

        list(gen())
        assert any(s.name == "gen" for s in collecting_exporter.spans)

    def test_generator_output_marker(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def gen():
            yield "x"

        list(gen())
        root = next(
            s
            for s in collecting_exporter.spans
            if s.name == "gen"
            and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
        )
        assert root.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "<generator>"

    def test_generator_yields_all_values(self, tracer):
        @BaseTracer.observe
        def gen():
            yield 10
            yield 20
            yield 30

        assert list(gen()) == [10, 20, 30]


class TestObserveAsyncGenerator:
    @pytest.mark.asyncio
    async def test_sync_fn_returning_async_generator(self, tracer, collecting_exporter):
        @BaseTracer.observe(record_output=True)
        def make_agen():
            async def _inner():
                yield 1
                yield 2

            return _inner()

        agen = make_agen()
        results = [v async for v in agen]
        assert results == [1, 2]
        root = next(
            s
            for s in collecting_exporter.spans
            if s.name == "make_agen"
            and s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"
        )
        assert root.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "<async_generator>"


class TestFormatInputs:
    def test_var_positional_args(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def fn(*args):
            return args

        fn(1, 2, 3)
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert "args" in span.attributes[AttributeKeys.JUDGMENT_INPUT]

    def test_var_keyword_args(self, tracer, collecting_exporter):
        @BaseTracer.observe
        def fn(**kwargs):
            return kwargs

        fn(x=1, y=2)
        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert "kwargs" in span.attributes[AttributeKeys.JUDGMENT_INPUT]

    def test_format_inputs_exception_returns_empty(self, tracer, collecting_exporter):
        from unittest.mock import patch
        import judgeval.v1.trace.base_tracer as bt

        @BaseTracer.observe
        def fn(a):
            return a

        with patch.object(bt.inspect, "signature", side_effect=ValueError("bad sig")):
            fn(42)

        span = next(s for s in collecting_exporter.spans if s.name == "fn")
        assert span.attributes.get(AttributeKeys.JUDGMENT_INPUT) == "{}"


class TestFormatInputsDirect:
    def test_var_positional(self):
        from judgeval.v1.trace.base_tracer import _format_inputs

        def fn(*args):
            pass

        result = _format_inputs(fn, (1, 2, 3), {})
        assert result == {"args": (1, 2, 3)}

    def test_var_keyword(self):
        from judgeval.v1.trace.base_tracer import _format_inputs

        def fn(**kwargs):
            pass

        result = _format_inputs(fn, (), {"x": 1})
        assert result == {"kwargs": {"x": 1}}

    def test_positional_passed_as_kwarg(self):
        from judgeval.v1.trace.base_tracer import _format_inputs

        def fn(a, b):
            pass

        result = _format_inputs(fn, (), {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_exception_returns_empty(self):
        from unittest.mock import patch
        from judgeval.v1.trace.base_tracer import _format_inputs
        import judgeval.v1.trace.base_tracer as bt

        def fn(a):
            pass

        with patch.object(bt.inspect, "signature", side_effect=ValueError):
            result = _format_inputs(fn, (1,), {})
        assert result == {}
