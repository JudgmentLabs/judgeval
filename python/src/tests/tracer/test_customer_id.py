# test_customer_id.py
import pytest
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

    # Mock the utility functions to avoid API key/org ID requirements
    with (
        patch("judgeval.tracer.expect_api_key") as mock_api_key,
        patch("judgeval.tracer.expect_organization_id") as mock_org_id,
        patch("judgeval.tracer._resolve_project_id") as mock_project_id,
    ):
        # Set up the mocks
        mock_api_key.return_value = "test_api_key"
        mock_org_id.return_value = "test_org_id"
        mock_project_id.return_value = "test_project_id"

        # Create the real tracer with real processor
        tracer = Tracer(project_name="research-agent")

        # Replace ONLY the exporter, keep everything else
        mock_exporter = MockExporter()
        tracer.judgment_processor._batch_processor._exporter = mock_exporter

        yield tracer

        # Cleanup after test
        tracer.judgment_processor._batch_processor.force_flush()
        if Tracer in SingletonMeta._instances:
            del SingletonMeta._instances[Tracer]


def test_customer_id_propagation(tracer):
    # Get the mock exporter
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="test_span")
    def test_function():
        tracer.set_customer_id("test-customer")
        return "test_result"

    @tracer.observe(span_name="test_span2")
    def test_function2():
        tracer.set_customer_id("test-customer2")
        return "test_result2"

    @tracer.observe(span_name="test_span3")
    def test_function3():
        tracer.set_customer_id("test-customer3")
        return "test_result3"

    test_function()
    test_function2()
    test_function3()

    # Check that spans were exported
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 3

    # Check span attributes
    for span in mock_exporter.exported_spans:
        if span.name == "test_span":
            print(span.attributes)
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "test-customer"
            )
        elif span.name == "test_span2":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "test-customer2"
            )
        elif span.name == "test_span3":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "test-customer3"
            )
    # Force flush the batch processor to export spans immediately
    tracer.judgment_processor._batch_processor.force_flush()


def test_customer_id_parent_child_spans(tracer):
    """Test that customer ID persists in child spans but not in unrelated spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    # Parent span with customer ID
    @tracer.observe(span_name="parent_span")
    def parent_function():
        tracer.set_customer_id("parent-customer")

        # Child span should inherit customer ID
        @tracer.observe(span_name="child_span")
        def child_function():
            # Should have parent's customer ID
            return "child_result"

        return child_function()

    # Unrelated span without customer ID
    @tracer.observe(span_name="unrelated_span")
    def unrelated_function():
        # Should NOT have customer ID
        return "unrelated_result"

    parent_function()
    unrelated_function()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 3

    # Check span attributes
    span_names = [span.name for span in mock_exporter.exported_spans]
    assert "parent_span" in span_names
    assert "child_span" in span_names
    assert "unrelated_span" in span_names

    for span in mock_exporter.exported_spans:
        if span.name == "parent_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "parent-customer"
            )
        elif span.name == "child_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "parent-customer"
            )
        elif span.name == "unrelated_span":
            # Should NOT have customer ID
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None


def test_customer_id_async_spans(tracer):
    """Test customer ID with async spans"""
    import asyncio

    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    async def async_parent():
        @tracer.observe(span_name="async_parent_span")
        async def async_parent_function():
            tracer.set_customer_id("async-customer")

            @tracer.observe(span_name="async_child_span")
            async def async_child_function():
                return "async_child_result"

            return await async_child_function()

        return await async_parent_function()

    async def async_unrelated():
        @tracer.observe(span_name="async_unrelated_span")
        async def async_unrelated_function():
            return "async_unrelated_result"

        return await async_unrelated_function()

    # Run async functions
    async def run_tests():
        await async_parent()
        await async_unrelated()

    asyncio.run(run_tests())

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 3

    # Check span attributes
    span_names = [span.name for span in mock_exporter.exported_spans]
    assert "async_parent_span" in span_names
    assert "async_child_span" in span_names
    assert "async_unrelated_span" in span_names

    for span in mock_exporter.exported_spans:
        if span.name == "async_parent_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "async-customer"
            )
        elif span.name == "async_child_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "async-customer"
            )
        elif span.name == "async_unrelated_span":
            # Should NOT have customer ID
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None


def test_customer_id_nested_spans(tracer):
    """Test customer ID with deeply nested spans"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="level1_span")
    def level1_function():
        tracer.set_customer_id("nested-customer")

        @tracer.observe(span_name="level2_span")
        def level2_function():
            @tracer.observe(span_name="level3_span")
            def level3_function():
                return "level3_result"

            return level3_function()

        return level2_function()

    level1_function()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 3

    # All nested spans should have the same customer ID
    for span in mock_exporter.exported_spans:
        assert (
            span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "nested-customer"
        )


def test_customer_id_multiple_traces(tracer):
    """Test that customer ID doesn't persist across different traces"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    # First trace
    @tracer.observe(span_name="trace1_span")
    def trace1_function():
        tracer.set_customer_id("trace1-customer")
        return "trace1_result"

    trace1_function()

    # Second trace (should not have customer ID)
    @tracer.observe(span_name="trace2_span")
    def trace2_function():
        # Should NOT have customer ID from trace1
        return "trace2_result"

    trace2_function()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 2

    # Check span attributes
    for span in mock_exporter.exported_spans:
        if span.name == "trace1_span":
            assert (
                span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
                == "trace1-customer"
            )
        elif span.name == "trace2_span":
            # Should NOT have customer ID
            assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None


def test_customer_id_concurrent_traces(tracer):
    """Test customer ID with concurrent traces (threading)"""
    import threading

    mock_exporter = tracer.judgment_processor._batch_processor._exporter
    results = []

    def trace_with_customer_id(customer_id, trace_name):
        @tracer.observe(span_name=f"{trace_name}_span")
        def trace_function():
            tracer.set_customer_id(customer_id)
            return f"{trace_name}_result"

        result = trace_function()
        results.append((customer_id, trace_name, result))

    # Create multiple threads with different customer IDs
    threads = []
    for i in range(10):
        thread = threading.Thread(
            target=trace_with_customer_id,
            args=(f"concurrent-customer-{i}", f"trace{i}"),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 10

    # Each span should have its own specific customer ID
    span_customer_map = {}
    for span in mock_exporter.exported_spans:
        customer_id = span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID)
        span_customer_map[span.name] = customer_id
        assert customer_id is not None

    # Verify specific customer IDs for each trace
    for i in range(10):
        assert span_customer_map[f"trace{i}_span"] == f"concurrent-customer-{i}"


def test_customer_id_exception_handling(tracer):
    """Test customer ID behavior when exceptions occur"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="exception_span")
    def function_with_exception():
        tracer.set_customer_id("exception-customer")
        raise ValueError("Test exception")

    # Should raise exception but still set customer ID
    with pytest.raises(ValueError):
        function_with_exception()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    assert (
        span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "exception-customer"
    )
    # Should also have exception information
    assert len(span.events) > 0


def test_customer_id_multiple_set_calls(tracer):
    """Test behavior when set_customer_id is called multiple times"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="multiple_set_span")
    def function_with_multiple_sets():
        # First call
        tracer.set_customer_id("first-customer")
        # Second call (should be ignored based on current implementation)
        tracer.set_customer_id("second-customer")
        return "result"

    function_with_multiple_sets()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = [
        span
        for span in mock_exporter.exported_spans
        if span.name == "multiple_set_span"
    ][0]
    # Should have the first customer ID (not the second)
    assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "first-customer"


def test_customer_id_empty_string(tracer):
    """Test behavior with empty string customer ID"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="empty_string_span")
    def function_with_empty_string():
        tracer.set_customer_id("")
        return "result"

    function_with_empty_string()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Should not have customer ID
    assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) is None


def test_customer_id_special_characters(tracer):
    """Test behavior with special characters in customer ID"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    special_customer_id = "customer-123_@#$%^&*()[]{}|\\:;\"'<>,.?/~`"

    @tracer.observe(span_name="special_chars_span")
    def function_with_special_chars():
        tracer.set_customer_id(special_customer_id)
        return "result"

    function_with_special_chars()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Should have the special character customer ID
    assert (
        span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == special_customer_id
    )


def test_customer_id_unicode_characters(tracer):
    """Test behavior with unicode characters in customer ID"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    unicode_customer_id = "客户-123_🚀_测试"

    @tracer.observe(span_name="unicode_span")
    def function_with_unicode():
        tracer.set_customer_id(unicode_customer_id)
        return "result"

    function_with_unicode()

    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = mock_exporter.exported_spans[0]
    # Should have the unicode customer ID
    assert (
        span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == unicode_customer_id
    )


def test_customer_id_context_var_persistence(tracer):
    """Test that customer ID persists in ContextVar across function calls"""
    mock_exporter = tracer.judgment_processor._batch_processor._exporter

    @tracer.observe(span_name="context_var_span")
    def function_with_context_var():
        tracer.set_customer_id("context-customer")

        # Check that ContextVar is set
        assert tracer.customer_id.get() == "context-customer"
        return "result"

    def tester():
        function_with_context_var()
        assert not tracer.customer_id.get()

    tester()
    # Force flush and check
    tracer.judgment_processor._batch_processor.force_flush()
    assert len(mock_exporter.exported_spans) == 1

    span = [
        span for span in mock_exporter.exported_spans if span.name == "context_var_span"
    ][0]
    assert span.attributes.get(AttributeKeys.JUDGMENT_CUSTOMER_ID) == "context-customer"
