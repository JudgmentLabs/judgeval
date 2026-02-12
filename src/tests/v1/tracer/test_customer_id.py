import asyncio
from typing import Tuple, Generator

import pytest
from unittest.mock import patch, MagicMock

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.exporters.in_memory_span_exporter import InMemorySpanExporter
from judgeval.v1.tracer.exporters.span_store import SpanStore
from judgeval.judgment_attribute_keys import AttributeKeys


@pytest.fixture
def tracer() -> Generator[Tuple[Tracer, SpanStore], None, None]:
    span_store = SpanStore()

    with patch.object(
        Tracer,
        "get_span_exporter",
        return_value=InMemorySpanExporter(span_store),
    ):
        mock_client = MagicMock()
        mock_client.base_url = "https://test.example.com/"

        t = Tracer(
            project_name="customer-id-test",
            project_id="test_project_id",
            enable_evaluation=False,
            enable_monitoring=True,
            api_client=mock_client,
            serializer=str,
            isolated=True,
            initialize=False,
        )

        yield t, span_store

        t.force_flush()
        t.shutdown()


def _final_spans(span_store: SpanStore):
    return [
        s
        for s in span_store.get_all()
        if s.attributes and s.attributes.get(AttributeKeys.JUDGMENT_UPDATE_ID) == 0
    ]


def test_customer_id_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="span_with_customer")
    def fn():
        t.set_customer_id("cust-123")
        return "ok"

    fn()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 1
    assert spans[0].attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-123"


def test_customer_id_parent_child_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="parent")
    def parent():
        t.set_customer_id("parent-cust")

        @t.observe(span_name="child")
        def child():
            return "child"

        return child()

    parent()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    for span in spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "parent-cust"


def test_customer_id_nested_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="level1")
    def level1():
        t.set_customer_id("nested-cust")

        @t.observe(span_name="level2")
        def level2():
            @t.observe(span_name="level3")
            def level3():
                return "deep"

            return level3()

        return level2()

    level1()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 3
    for span in spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "nested-cust"


def test_customer_id_does_not_persist_across_traces(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="trace1")
    def trace1():
        t.set_customer_id("trace1-cust")
        return "t1"

    @t.observe(span_name="trace2")
    def trace2():
        return "t2"

    trace1()
    trace2()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    for span in spans:
        if span.name == "trace1":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "trace1-cust"
            )
        elif span.name == "trace2":
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None


def test_customer_id_unrelated_span_unaffected(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="with_cust")
    def with_cust():
        t.set_customer_id("my-cust")

        @t.observe(span_name="child_cust")
        def child():
            return "c"

        return child()

    @t.observe(span_name="no_cust")
    def no_cust():
        return "n"

    with_cust()
    no_cust()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 3
    for span in spans:
        if span.name == "no_cust":
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None
        else:
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "my-cust"


def test_customer_id_with_exception(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="error_span")
    def failing():
        t.set_customer_id("err-cust")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        failing()

    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 1
    assert spans[0].attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "err-cust"


def test_customer_id_async_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    async def run():
        @t.observe(span_name="async_parent")
        async def async_parent():
            t.set_customer_id("async-cust")

            @t.observe(span_name="async_child")
            async def async_child():
                return "ac"

            return await async_child()

        return await async_parent()

    asyncio.run(run())
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    for span in spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "async-cust"


def test_customer_id_multiple_distinct_traces(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="t1")
    def t1():
        t.set_customer_id("cust-a")
        return "a"

    @t.observe(span_name="t2")
    def t2():
        t.set_customer_id("cust-b")
        return "b"

    @t.observe(span_name="t3")
    def t3():
        t.set_customer_id("cust-c")
        return "c"

    t1()
    t2()
    t3()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 3
    by_name = {s.name: s for s in spans}
    assert by_name["t1"].attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-a"
    assert by_name["t2"].attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-b"
    assert by_name["t3"].attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-c"
