import pytest
import asyncio
from unittest.mock import patch, MagicMock
from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.exporters.in_memory_span_exporter import InMemorySpanExporter
from judgeval.v1.tracer.exporters.span_store import SpanStore
from judgeval.judgment_attribute_keys import AttributeKeys


@pytest.fixture
def tracer():
    span_store = SpanStore()

    with patch("judgeval.v1.tracer.base_tracer.resolve_project_id") as mock_resolve:
        mock_resolve.return_value = "test-project-id"

        with patch.object(
            Tracer,
            "get_span_exporter",
            return_value=InMemorySpanExporter(span_store),
        ):
            mock_client = MagicMock()
            mock_client.base_url = "https://test.example.com/"

            t = Tracer(
                project_name="test_project",
                project_id="test-project-id",
                enable_evaluation=False,
                enable_monitoring=True,
                api_client=mock_client,
                serializer=str,
                isolated=True,
                initialize=False,
            )

            yield t, span_store, mock_resolve

            t.shutdown()


def test_override_project_sets_attribute(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "override-project-id"

    @t.observe(span_name="test_span")
    def test_function():
        t.override_project("override-project")
        return "result"

    test_function()
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
        == "override-project-id"
    )


def test_override_project_propagates_to_children(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "parent-project-id"

    @t.observe(span_name="parent_span")
    def parent_function():
        t.override_project("parent-project")

        @t.observe(span_name="child_span")
        def child_function():
            return "child_result"

        return child_function()

    parent_function()
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 2

    for span in spans:
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
            == "parent-project-id"
        )


def test_override_project_deeply_nested(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "nested-project-id"

    @t.observe(span_name="level1_span")
    def level1_function():
        t.override_project("nested-project")

        @t.observe(span_name="level2_span")
        def level2_function():
            @t.observe(span_name="level3_span")
            def level3_function():
                return "level3_result"

            return level3_function()

        return level2_function()

    level1_function()
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 3

    for span in spans:
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
            == "nested-project-id"
        )


def test_override_project_not_on_non_root_span(tracer, caplog):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "child-project-id"

    @t.observe(span_name="parent_span")
    def parent_function():
        @t.observe(span_name="child_span")
        def child_function():
            t.override_project("child-project")
            return "child_result"

        return child_function()

    parent_function()
    t.force_flush()

    assert "non-root span" in caplog.text

    spans = span_store.get_all()
    for span in spans:
        if span.name == "child_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE) is None
            )


def test_override_project_outside_span_context(tracer, caplog):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "outside-project-id"

    t.override_project("outside-project")

    assert "outside of a span context" in caplog.text


def test_override_project_invalid_project(tracer, caplog):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = None

    @t.observe(span_name="test_span")
    def test_function():
        t.override_project("invalid-project")
        return "result"

    test_function()
    t.force_flush()

    assert "Failed to resolve project" in caplog.text

    spans = span_store.get_all()
    span = spans[0]
    assert span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE) is None


def test_override_project_does_not_persist_across_traces(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "trace1-project-id"

    @t.observe(span_name="trace1_span")
    def trace1_function():
        t.override_project("trace1-project")
        return "trace1_result"

    trace1_function()

    mock_resolve.return_value = "trace2-project-id"

    @t.observe(span_name="trace2_span")
    def trace2_function():
        return "trace2_result"

    trace2_function()
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 2

    for span in spans:
        if span.name == "trace1_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
                == "trace1-project-id"
            )
        elif span.name == "trace2_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE) is None
            )


def test_override_project_async_spans(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "async-project-id"

    async def async_parent():
        @t.observe(span_name="async_parent_span")
        async def async_parent_function():
            t.override_project("async-project")

            @t.observe(span_name="async_child_span")
            async def async_child_function():
                return "async_child_result"

            return await async_child_function()

        return await async_parent_function()

    asyncio.run(async_parent())
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 2

    for span in spans:
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
            == "async-project-id"
        )


def test_override_project_with_exception(tracer):
    t, span_store, mock_resolve = tracer
    mock_resolve.return_value = "exception-project-id"

    @t.observe(span_name="exception_span")
    def function_with_exception():
        t.override_project("exception-project")
        raise ValueError("Test exception")

    with pytest.raises(ValueError):
        function_with_exception()

    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE)
        == "exception-project-id"
    )


def test_no_override_has_no_attribute(tracer):
    t, span_store, mock_resolve = tracer

    @t.observe(span_name="no_override_span")
    def test_function():
        return "result"

    test_function()
    t.force_flush()

    spans = span_store.get_all()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get(AttributeKeys.JUDGMENT_PROJECT_ID_OVERRIDE) is None
