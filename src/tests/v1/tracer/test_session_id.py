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
            project_name="session-id-test",
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


def test_session_id_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="span_with_session")
    def fn():
        t.set_session_id("sess-123")
        return "ok"

    fn()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 1
    assert spans[0].attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-123"


def test_session_id_parent_child_propagation(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="parent")
    def parent():
        t.set_session_id("parent-sess")

        @t.observe(span_name="child")
        def child():
            return "child"

        return child()

    parent()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    for span in spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "parent-sess"


def test_session_id_nested_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="level1")
    def level1():
        t.set_session_id("nested-sess")

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
        assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "nested-sess"


def test_session_id_does_not_persist_across_traces(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="trace1")
    def trace1():
        t.set_session_id("trace1-sess")
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
                span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "trace1-sess"
            )
        elif span.name == "trace2":
            assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) is None


def test_session_id_unrelated_span_unaffected(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="with_sess")
    def with_sess():
        t.set_session_id("my-sess")

        @t.observe(span_name="child_sess")
        def child():
            return "c"

        return child()

    @t.observe(span_name="no_sess")
    def no_sess():
        return "n"

    with_sess()
    no_sess()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 3
    for span in spans:
        if span.name == "no_sess":
            assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) is None
        else:
            assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "my-sess"


def test_session_id_with_exception(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    @t.observe(span_name="error_span")
    def failing():
        t.set_session_id("err-sess")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        failing()

    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 1
    assert spans[0].attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "err-sess"


def test_session_id_async_propagation(tracer: Tuple[Tracer, SpanStore]) -> None:
    t, span_store = tracer

    async def run():
        @t.observe(span_name="async_parent")
        async def async_parent():
            t.set_session_id("async-sess")

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
        assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "async-sess"


def test_session_id_and_customer_id_together(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="both_ids")
    def fn():
        t.set_customer_id("cust-1")
        t.set_session_id("sess-1")

        @t.observe(span_name="child_both")
        def child():
            return "ok"

        return child()

    fn()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    for span in spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "cust-1"
        assert span.attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-1"


def test_session_id_multiple_distinct_traces(
    tracer: Tuple[Tracer, SpanStore],
) -> None:
    t, span_store = tracer

    @t.observe(span_name="s1")
    def s1():
        t.set_session_id("sess-a")
        return "a"

    @t.observe(span_name="s2")
    def s2():
        t.set_session_id("sess-b")
        return "b"

    s1()
    s2()
    t.force_flush()

    spans = _final_spans(span_store)
    assert len(spans) == 2
    by_name = {s.name: s for s in spans}
    assert by_name["s1"].attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-a"
    assert by_name["s2"].attributes.get(AttributeKeys.JUDGMENT_SESSION_ID) == "sess-b"
