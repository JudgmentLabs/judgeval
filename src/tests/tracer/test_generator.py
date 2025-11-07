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
    """Test basic sync generator wrapping and output capture"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="sync_gen")
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
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2,3]"


def test_sync_generator_string_concatenation(tracer):
    """Test that string generators are properly concatenated"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="string_gen")
    def string_generator():
        yield "Hello"
        yield " "
        yield "World"

    result = "".join(string_generator())
    assert result == "Hello World"

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "string_gen"
    # Should concatenate strings
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == '"Hello World"'


def test_async_generator_basic(tracer):
    """Test basic async generator wrapping and output capture"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_gen")
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
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2,3]"


def test_async_generator_string_concatenation(tracer):
    """Test that async string generators are properly concatenated"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_string_gen")
    async def async_string_generator():
        yield "Async"
        yield " "
        yield "Test"

    async def run_test():
        result = ""
        async for item in async_string_generator():
            result += item
        return result

    result = asyncio.run(run_test())
    assert result == "Async Test"

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_string_gen"
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == '"Async Test"'


def test_generator_context_preservation(tracer):
    """Test that context variables are preserved across generator iterations"""
    test_var = contextvars.ContextVar("test_var", default=None)
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_with_context")
    def parent_function():
        test_var.set("TEST_VALUE")

        @tracer.observe(span_name="gen_with_context")
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

    @tracer.observe(span_name="async_parent_with_context")
    async def async_parent_function():
        test_var.set("ASYNC_TEST_VALUE")

        @tracer.observe(span_name="async_gen_with_context")
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

    @tracer.observe(span_name="parent_with_customer")
    def parent_with_customer():
        tracer.set_customer_id("gen-customer")

        @tracer.observe(span_name="child_generator")
        def child_generator():
            yield 1
            yield 2
            yield 3

        return list(child_generator())

    result = parent_with_customer()
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    # Both spans should have the customer ID
    for span in mock_exporter.exported_spans:
        assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "gen-customer"


def test_generator_exception_handling(tracer):
    """Test that exceptions in generators are properly handled"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="failing_generator")
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
    # Should have exception recorded
    assert len(span.events) > 0


def test_async_generator_exception_handling(tracer):
    """Test that exceptions in async generators are properly handled"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="failing_async_generator")
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


def test_nested_generators(tracer):
    """Test nested generator spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="outer_generator")
    def outer_generator():
        for i in range(2):

            @tracer.observe(span_name=f"inner_generator_{i}")
            def inner_generator():
                yield f"inner_{i}_a"
                yield f"inner_{i}_b"

            for item in inner_generator():
                yield item

    result = list(outer_generator())
    assert result == ["inner_0_a", "inner_0_b", "inner_1_a", "inner_1_b"]

    tracer.judgment_processor._batch_processor.force_flush()
    # Should have 1 outer + 2 inner generator spans
    assert len(mock_exporter.exported_spans) == 3


def test_generator_partial_consumption(tracer):
    """Test that generator span closes even with partial consumption"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="partial_generator")
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
    # Verify partial output was saved (only 2 items consumed before close)
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2]"
    # Verify span was properly ended
    assert span.end_time is not None


def test_generator_empty(tracer):
    """Test empty generator"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="empty_generator")
    def empty_generator():
        # Empty generator - immediately returns without yielding
        for _ in []:
            yield

    result = list(empty_generator())
    assert result == []

    tracer.judgment_processor._batch_processor.force_flush()
    for span in mock_exporter.exported_spans:
        print(span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT))
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "empty_generator"
    # Empty generator treated as empty string list by join logic
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == '""'


def test_generator_mixed_types(tracer):
    """Test generator with mixed return types"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="mixed_generator")
    def mixed_generator():
        yield 1
        yield "string"
        yield {"key": "value"}
        yield [1, 2, 3]

    result = list(mixed_generator())
    assert len(result) == 4

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "mixed_generator"
    # Output should be serialized
    output = span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
    assert output is not None


def test_generator_parent_child_relationship(tracer):
    """Test that generator span has correct parent relationship"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="parent_function")
    def parent_function():
        @tracer.observe(span_name="child_generator")
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


def test_async_generator_in_async_function(tracer):
    """Test async generator called from async function"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_parent")
    async def async_parent():
        @tracer.observe(span_name="async_child_gen")
        async def async_child_generator():
            yield "a"
            yield "b"
            yield "c"

        result = []
        async for item in async_child_generator():
            result.append(item)
        return result

    result = asyncio.run(async_parent())
    assert result == ["a", "b", "c"]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    parent_span = [s for s in mock_exporter.exported_spans if s.name == "async_parent"][
        0
    ]
    child_span = [
        s for s in mock_exporter.exported_spans if s.name == "async_child_gen"
    ][0]

    assert child_span.context.trace_id == parent_span.context.trace_id


def test_generator_with_input_capture(tracer):
    """Test that generator input is properly captured"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="gen_with_input")
    def generator_with_input(start, end):
        for i in range(start, end):
            yield i

    result = list(generator_with_input(5, 8))
    assert result == [5, 6, 7]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "gen_with_input"
    # Input should be captured
    input_attr = span.attributes.get(AttributeKeys.JUDGMENT_INPUT)
    assert input_attr is not None


def test_concurrent_generators(tracer):
    """Test multiple generators running concurrently (async)"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="concurrent_gen")
    async def concurrent_generator(id):
        for i in range(3):
            yield f"{id}_{i}"

    async def collect_items(gen):
        return [item async for item in gen]

    async def run_test():
        results = await asyncio.gather(
            collect_items(concurrent_generator("A")),
            collect_items(concurrent_generator("B")),
            collect_items(concurrent_generator("C")),
        )
        return results

    results = asyncio.run(run_test())
    assert len(results) == 3

    tracer.judgment_processor._batch_processor.force_flush()
    # Should have 3 generator spans
    assert len(mock_exporter.exported_spans) == 3

    span_names = [s.name for s in mock_exporter.exported_spans]
    assert span_names.count("concurrent_gen") == 3


def test_generator_span_timing(tracer):
    """Test that generator span duration covers entire iteration"""
    import time

    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    # Define generator outside to avoid decorator nesting issues
    def plain_timed_generator():
        yield 1
        time.sleep(0.1)
        yield 2
        time.sleep(0.1)
        yield 3

    # Wrap with observe
    timed_generator = tracer.observe(span_name="timed_generator")(plain_timed_generator)

    start_time = time.time()
    result = list(timed_generator())
    end_time = time.time()

    assert result == [1, 2, 3]
    actual_duration = end_time - start_time
    assert actual_duration >= 0.2  # Should take at least 0.2s

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Span duration should cover the entire execution
    span_duration_ns = span.end_time - span.start_time
    span_duration_s = span_duration_ns / 1e9
    assert span_duration_s >= 0.2


def test_generator_output_attribute_set(tracer):
    """Test that generator output is only set once at the end"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="output_test_gen")
    def output_test_generator():
        yield 1
        yield 2
        yield 3

    result = list(output_test_generator())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Output should be the aggregated result
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2,3]"
    # Span should only be ended once
    assert span.end_time is not None


def test_async_generator_with_customer_id(tracer):
    """Test that customer ID persists through async generator execution"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_parent_with_customer")
    async def async_parent_with_customer():
        tracer.set_customer_id("async-gen-customer")

        @tracer.observe(span_name="async_child_generator")
        async def async_child_generator():
            yield 1
            yield 2
            yield 3

        result = []
        async for item in async_child_generator():
            result.append(item)
        return result

    result = asyncio.run(async_parent_with_customer())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    # Both spans should have the customer ID
    for span in mock_exporter.exported_spans:
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
            == "async-gen-customer"
        )


def test_async_generator_partial_consumption(tracer):
    """Test that async generator span closes even with partial consumption"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_partial_generator")
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
    # Verify partial output was saved (only 2 items consumed before close)
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2]"
    # Verify span was properly ended
    assert span.end_time is not None


def test_async_generator_empty(tracer):
    """Test empty async generator"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_empty_generator")
    async def async_empty_generator():
        # Empty generator - immediately returns without yielding
        for _ in []:
            yield

    async def run_test():
        result = []
        async for item in async_empty_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == []

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_empty_generator"
    # Empty generator treated as empty string list by join logic
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == '""'


def test_async_generator_mixed_types(tracer):
    """Test async generator with mixed return types"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_mixed_generator")
    async def async_mixed_generator():
        yield 1
        yield "string"
        yield {"key": "value"}
        yield [1, 2, 3]

    async def run_test():
        result = []
        async for item in async_mixed_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert len(result) == 4

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_mixed_generator"
    # Output should be serialized
    output = span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT)
    assert output is not None


def test_async_generator_with_input_capture(tracer):
    """Test that async generator input is properly captured"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_gen_with_input")
    async def async_generator_with_input(start, end):
        for i in range(start, end):
            yield i

    async def run_test():
        result = []
        async for item in async_generator_with_input(5, 8):
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == [5, 6, 7]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert span.name == "async_gen_with_input"
    # Input should be captured
    input_attr = span.attributes.get(AttributeKeys.JUDGMENT_INPUT)
    assert input_attr is not None


def test_async_generator_span_timing(tracer):
    """Test that async generator span duration covers entire iteration"""
    import time

    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    # Define generator outside to avoid decorator nesting issues
    async def plain_async_timed_generator():
        yield 1
        await asyncio.sleep(0.1)
        yield 2
        await asyncio.sleep(0.1)
        yield 3

    # Wrap with observe
    async_timed_generator = tracer.observe(span_name="async_timed_generator")(
        plain_async_timed_generator
    )

    async def run_test():
        start_time = time.time()
        result = []
        async for item in async_timed_generator():
            result.append(item)
        end_time = time.time()
        return result, start_time, end_time

    result, start_time, end_time = asyncio.run(run_test())

    assert result == [1, 2, 3]
    actual_duration = end_time - start_time
    assert actual_duration >= 0.2  # Should take at least 0.2s

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Span duration should cover the entire execution
    span_duration_ns = span.end_time - span.start_time
    span_duration_s = span_duration_ns / 1e9
    assert span_duration_s >= 0.2


def test_async_generator_output_attribute_set(tracer):
    """Test that async generator output is only set once at the end"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_output_test_gen")
    async def async_output_test_generator():
        yield 1
        yield 2
        yield 3

    async def run_test():
        result = []
        async for item in async_output_test_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == [1, 2, 3]

    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Output should be the aggregated result
    assert span.attributes.get(AttributeKeys.JUDGMENT_OUTPUT) == "[1,2,3]"
    # Span should only be ended once
    assert span.end_time is not None


def test_async_nested_generators(tracer):
    """Test nested async generator spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="async_outer_generator")
    async def async_outer_generator():
        for i in range(2):

            @tracer.observe(span_name=f"async_inner_generator_{i}")
            async def async_inner_generator():
                yield f"inner_{i}_a"
                yield f"inner_{i}_b"

            async for item in async_inner_generator():
                yield item

    async def run_test():
        result = []
        async for item in async_outer_generator():
            result.append(item)
        return result

    result = asyncio.run(run_test())
    assert result == ["inner_0_a", "inner_0_b", "inner_1_a", "inner_1_b"]

    tracer.judgment_processor._batch_processor.force_flush()
    # Should have 1 outer + 2 inner generator spans
    assert len(mock_exporter.exported_spans) == 3
