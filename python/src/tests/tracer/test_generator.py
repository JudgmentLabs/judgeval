# test_generator.py
import pytest
import asyncio
import contextvars
from unittest.mock import patch
from judgeval.tracer.exporters.utils import deduplicate_spans
from judgeval.tracer import Tracer
from judgeval.tracer.keys import AttributeKeys


class MockExporter:
    """Mock exporter that captures exported spans"""

    def __init__(self):
        self.exported_spans = []

    def export(self, spans):
        """Capture spans when they're exported"""
        self.exported_spans.extend(deduplicate_spans(spans))
        return True

    def shutdown(self):
        pass


@pytest.fixture
def tracer():
    """Create a tracer with mocked dependencies"""
    # Clear any existing singleton instance
    from judgeval.utils.meta import SingletonMeta
    from opentelemetry.trace import _TRACER_PROVIDER_SET_ONCE, _TRACER_PROVIDER

    if Tracer in SingletonMeta._instances:
        del SingletonMeta._instances[Tracer]

    # Reset the global tracer provider flag (OpenTelemetry internal)
    try:
        _TRACER_PROVIDER_SET_ONCE._done = False
        _TRACER_PROVIDER._default = None
    except Exception:
        pass  # If the internal API changes, just continue

    with (
        patch("judgeval.tracer.expect_api_key") as mock_api_key,
        patch("judgeval.tracer.expect_organization_id") as mock_org_id,
        patch("judgeval.tracer._resolve_project_id") as mock_project_id,
    ):
        mock_api_key.return_value = "test_api_key"
        mock_org_id.return_value = "test_org_id"
        mock_project_id.return_value = "test_project_id"

        tracer = Tracer(project_name="generator-test")
        mock_exporter = MockExporter()
        tracer.judgment_processor._batch_processor._exporter = mock_exporter

        yield tracer

        # Cleanup after test
        tracer.judgment_processor._batch_processor.force_flush()
        if Tracer in SingletonMeta._instances:
            del SingletonMeta._instances[Tracer]


def test_sync_generator_basic(tracer):
    """Test basic sync generator creates proper spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="sync_gen", disable_generator_yield_span=True)
    def sync_generator():
        yield 1
        yield 2
        yield 3

    result = list(sync_generator())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "sync_gen"
    assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"


def test_async_generator_basic(tracer):
    """Test basic async generator creates proper spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_gen", disable_generator_yield_span=True)
    async def async_generator():
        yield 1
        yield 2
        yield 3

    async def run_test():
        result = []
        async for item in async_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_gen"
    assert span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator"


def test_generator_context_preservation(tracer):
    """Test that context variables are preserved across generator iterations"""
    test_var = contextvars.ContextVar("test_var", default=None)
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_with_context", disable_generator_yield_span=True)
    def parent_function():
        test_var.set("TEST_VALUE")

        @tracer.observe(span_name="gen_with_context", disable_generator_yield_span=True)
        def generator_with_context():
            # Context should be preserved in each yield
            for i in range(3):
                assert test_var.get() == "TEST_VALUE", f"Context lost at iteration {i}"
                yield i

        return list(generator_with_context())

    result = parent_function()
    assert result == [0, 1, 2]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2


def test_async_generator_context_preservation(tracer):
    """Test that context variables are preserved across async generator iterations"""
    test_var = contextvars.ContextVar("test_var", default=None)
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(
        span_name="async_parent_with_context", disable_generator_yield_span=True
    )
    async def async_parent_function():
        test_var.set("ASYNC_TEST_VALUE")

        @tracer.observe(
            span_name="async_gen_with_context", disable_generator_yield_span=True
        )
        async def async_generator_with_context():
            for i in range(3):
                assert test_var.get() == "ASYNC_TEST_VALUE", (
                    f"Context lost at iteration {i}"
                )
                yield i

        result = []
        async for item in async_generator_with_context():
            result.append(item)
        return result

    result = asyncio.run(async_parent_function())
    assert result == [0, 1, 2]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2


def test_generator_with_customer_id(tracer):
    """Test that customer ID persists through generator execution"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_with_customer", disable_generator_yield_span=True)
    def parent_with_customer():
        tracer.set_customer_id("gen-customer")

        @tracer.observe(span_name="child_generator", disable_generator_yield_span=True)
        def child_generator():
            yield 1
            yield 2
            yield 3

        return list(child_generator())

    result = parent_with_customer()
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    # Both spans should have customer ID
    for span in mock_exporter.exported_spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "gen-customer"


def test_generator_exception_handling(tracer):
    """Test that exceptions in generators are properly handled"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="failing_generator", disable_generator_yield_span=True)
    def failing_generator():
        yield 1
        yield 2
        raise ValueError("Generator error")

    with pytest.raises(ValueError, match="Generator error"):
        list(failing_generator())

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "failing_generator"
    assert len(span.events) > 0


def test_async_generator_exception_handling(tracer):
    """Test that exceptions in async generators are properly handled"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(
        span_name="failing_async_generator", disable_generator_yield_span=True
    )
    async def failing_async_generator():
        yield 1
        yield 2
        raise ValueError("Async generator error")

    async def run_test():
        result = []
        async for item in failing_async_generator():
            result.append(item)

    with pytest.raises(ValueError, match="Async generator error"):
        asyncio.run(run_test())

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "failing_async_generator"
    assert len(span.events) > 0


def test_generator_partial_consumption(tracer):
    """Test that generator span closes even with partial consumption"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="partial_generator", disable_generator_yield_span=True)
    def partial_generator():
        yield 1
        yield 2
        yield 3
        yield 4
        yield 5

    gen = partial_generator()
    # Only consume first 2 items
    assert next(gen) == 1
    assert next(gen) == 2
    # Close generator early
    gen.close()

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "partial_generator"
    # Verify span was properly ended
    assert span.end_time is not None


def test_async_generator_partial_consumption(tracer):
    """Test that async generator span closes even with partial consumption"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(
        span_name="async_partial_generator", disable_generator_yield_span=True
    )
    async def async_partial_generator():
        yield 1
        yield 2
        yield 3
        yield 4
        yield 5

    async def run_test():
        gen = async_partial_generator()
        # Only consume first 2 items
        assert await gen.__anext__() == 1
        assert await gen.__anext__() == 2
        # Close generator early
        await gen.aclose()

    asyncio.run(run_test())

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_partial_generator"
    # Verify span was properly ended
    assert span.end_time is not None


def test_generator_parent_child_relationship(tracer):
    """Test that generator span has correct parent relationship"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_function", disable_generator_yield_span=True)
    def parent_function():
        @tracer.observe(span_name="child_generator", disable_generator_yield_span=True)
        def child_generator():
            yield 1
            yield 2

        return list(child_generator())

    parent_function()

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    parent_span = [
        s for s in mock_exporter.exported_spans if s.name == "parent_function"
    ][0]
    child_span = [
        s for s in mock_exporter.exported_spans if s.name == "child_generator"
    ][0]

    # Child should have parent's trace_id
    assert child_span.context.trace_id == parent_span.context.trace_id
    # Child's parent_id should be parent's span_id
    assert child_span.parent.span_id == parent_span.context.span_id


def test_sync_generator_with_child_spans(tracer):
    """Test sync generator with child spans enabled (default)"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_gen")
    def parent_generator():
        yield 1
        yield 2
        yield 3

    result = list(parent_generator())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()

    # Should have 1 parent + 3 child spans (one per yield)
    assert len(mock_exporter.exported_spans) == 4

    child_spans = [
        s
        for s in mock_exporter.exported_spans
        if s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator_item"
    ]

    assert len(child_spans) == 3

    # Verify each child span has output
    for child_span in child_spans:
        output = child_span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
        assert output is not None


def test_async_generator_with_child_spans(tracer):
    """Test async generator with child spans enabled (default)"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_parent_gen")
    async def async_parent_generator():
        yield "a"
        yield "b"
        yield "c"

    async def run_test():
        result = []
        async for item in async_parent_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == ["a", "b", "c"]

    tracer.judgment_processor._batch_processor.force_flush()

    # Should have 1 parent + 3 child spans (one per yield)
    assert len(mock_exporter.exported_spans) == 4

    child_spans = [
        s
        for s in mock_exporter.exported_spans
        if s.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "generator_item"
    ]

    assert len(child_spans) == 3

    # Verify each child span has output
    for child_span in child_spans:
        output = child_span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
        assert output is not None
