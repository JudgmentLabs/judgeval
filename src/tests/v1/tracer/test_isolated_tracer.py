"""
Test suite for isolated tracer functionality.

This ensures that isolated tracers properly maintain separate contexts
and don't leak trace_ids or span_ids across different operations or projects.

Reproduces and validates the fix for the bug where the same trace_id and span_id
appeared across different operations at different timestamps.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.utils.serialize import safe_serialize


@pytest.fixture
def mock_client():
    """Create a mock JudgmentSyncClient."""
    client = MagicMock(spec=JudgmentSyncClient)
    client.organization_id = "test-org-id"
    client.api_key = "test-api-key"
    client.base_url = "https://api.test.com/"
    return client


@pytest.fixture
def serializer():
    """Return the serializer function."""
    return safe_serialize


def test_isolated_tracer_sequential_operations_unique_trace_ids(
    mock_client, serializer
):
    """
    Test that sequential operations with the same isolated tracer generate unique trace_ids.

    This reproduces the customer bug where:
    - Same trace_id: 56c5f4017500d610567f56f4699b4062
    - Same span_id: 5747d44f730131a9
    - Appeared across different timestamps (Dec 8, Dec 9, Dec 15, Dec 23)
    - Different operations: verify_evidence_v1, audit_evidence_v1

    Root cause: Stale ended spans persisting in ContextVarsRuntimeContext.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="Prod - Verifiers",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        trace_ids = []
        span_ids = []

        # Simulate multiple operations over time (like the customer's case)
        operations = [
            "verify_evidence_v1",
            "audit_evidence_v1",
            "verify_evidence_v1",  # Same operation again
            "audit_evidence_v1",  # Same operation again
        ]

        for operation_name in operations:
            with tracer.span(operation_name) as span:
                ctx = span.get_span_context()
                trace_ids.append(ctx.trace_id)
                span_ids.append(ctx.span_id)

        # All trace_ids must be unique (each operation should create a new root trace)
        assert len(set(trace_ids)) == 4, (
            f"Expected 4 unique trace_ids, got {len(set(trace_ids))}. "
            f"trace_ids: {[format(t, '032x') for t in trace_ids]}"
        )

        # All span_ids must be unique
        assert len(set(span_ids)) == 4, (
            f"Expected 4 unique span_ids, got {len(set(span_ids))}. "
            f"span_ids: {[format(s, '016x') for s in span_ids]}"
        )


def test_isolated_tracer_parent_child_same_trace(mock_client, serializer):
    """
    Test that parent-child spans share the same trace_id but have different span_ids.

    This ensures that legitimate parent-child relationships still work correctly
    while fixing the bug where independent operations shared trace_ids.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        parent_ctx = None
        child_ctx = None

        with tracer.span("parent_operation") as parent:
            parent_ctx = parent.get_span_context()

            with tracer.span("child_operation") as child:
                child_ctx = child.get_span_context()

        # Parent and child should share the same trace_id
        assert parent_ctx.trace_id == child_ctx.trace_id, (
            "Parent and child spans should share the same trace_id"
        )

        # But have different span_ids
        assert parent_ctx.span_id != child_ctx.span_id, (
            "Parent and child spans should have different span_ids"
        )


def test_isolated_tracer_stale_span_detection(mock_client, serializer):
    """
    Test that stale (ended) spans in context don't cause new spans to inherit their trace_id.

    This directly tests the fix: checking if a span is still recording before reusing context.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        first_trace_id = None

        # First operation - span ends when context exits
        with tracer.span("first_operation") as span:
            first_trace_id = span.get_span_context().trace_id
            # Span is recording here
            assert span.is_recording()

        # After exiting, span should no longer be recording
        assert not span.is_recording()

        # Second operation - should create NEW root span, not child of first
        with tracer.span("second_operation") as span:
            second_trace_id = span.get_span_context().trace_id

            # Should be a different trace_id because first span has ended
            assert second_trace_id != first_trace_id, (
                "Sequential operations should have different trace_ids. "
                f"Got same trace_id: {format(first_trace_id, '032x')}"
            )


def test_multiple_isolated_tracers_same_provider_share_context(mock_client, serializer):
    """
    Test that multiple tracer instances from the same provider share runtime context.

    This is intentional - allows parent-child relationships across different parts
    of the application using the same provider.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        # Create tracer with isolated provider
        tracer1 = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        # Get the same provider - this simulates getting tracer multiple times
        provider = tracer1.tracer_provider
        tracer2_instance = provider.get_tracer("judgeval")

        parent_trace_id = None

        # Start a span with first tracer instance
        with tracer1.span("parent_operation") as parent:
            parent_trace_id = parent.get_span_context().trace_id

            # Create child span using the second tracer instance from same provider
            # This should work because they share the runtime context
            child_span = tracer2_instance.start_span("child_operation")
            child_trace_id = child_span.get_span_context().trace_id
            child_span.end()

            # Child should inherit parent's trace_id (same provider)
            assert child_trace_id == parent_trace_id, (
                "Tracers from same provider should share context for parent-child relationships"
            )


def test_isolated_tracer_exception_handling(mock_client, serializer):
    """
    Test that exceptions are properly recorded on spans in isolated mode.

    This validates the fix where start_as_current_span now delegates to use_span
    for proper exception handling.
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        span_ref = None

        # Test that exceptions are recorded
        with pytest.raises(ValueError):
            with tracer.span("failing_operation") as span:
                span_ref = span
                raise ValueError("Test error")

        # Span should have recorded the exception
        # Note: We can't easily verify span.events without accessing internals,
        # but we verify the span ended and is no longer recording
        assert not span_ref.is_recording(), "Span should be ended after exception"


def test_isolated_tracer_no_global_context_pollution(mock_client, serializer):
    """
    Test that isolated tracers don't affect the global OpenTelemetry context.

    This ensures isolated=True actually isolates the tracer from global state.
    """
    from opentelemetry import context as otel_context

    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        # Get global context before creating isolated tracer
        global_context_before = otel_context.get_current()

        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        # Create a span with isolated tracer
        with tracer.span("isolated_operation"):
            # Get isolated context
            isolated_context = tracer.get_context()

            # Get global context while span is active
            global_context_during = otel_context.get_current()

            # Isolated context should be different from global context
            assert isolated_context is not global_context_during, (
                "Isolated context should not be the same as global context"
            )

            # Global context should be unchanged
            assert global_context_during is global_context_before, (
                "Isolated tracer should not modify global context"
            )


def test_isolated_tracer_concurrent_operations_pattern(mock_client, serializer):
    """
    Test a realistic pattern: multiple sequential root operations in the same thread.

    This simulates a web server handling multiple requests sequentially
    in the same worker thread/process (common in sync frameworks).
    """
    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="web_service",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        results = []

        # Simulate handling 5 different requests sequentially
        for request_id in range(5):
            with tracer.span(f"handle_request_{request_id}") as span:
                ctx = span.get_span_context()
                results.append(
                    {
                        "request_id": request_id,
                        "trace_id": format(ctx.trace_id, "032x"),
                        "span_id": format(ctx.span_id, "016x"),
                    }
                )

        # Verify all trace_ids are unique
        trace_ids = [r["trace_id"] for r in results]
        assert len(set(trace_ids)) == 5, (
            f"Expected 5 unique trace_ids, got {len(set(trace_ids))}"
        )

        # Verify all span_ids are unique
        span_ids = [r["span_id"] for r in results]
        assert len(set(span_ids)) == 5, (
            f"Expected 5 unique span_ids, got {len(set(span_ids))}"
        )

        # Print results for debugging
        print("\nRequest handling results:")
        for r in results:
            print(
                f"  Request {r['request_id']}: trace_id={r['trace_id']}, span_id={r['span_id']}"
            )


def test_isolated_tracer_get_context_returns_provider_context(mock_client, serializer):
    """
    Test that tracer.get_context() correctly returns the provider's isolated context.

    This validates the architectural fix where we use the provider's public method
    instead of reaching into private attributes.
    """
    from judgeval.v1.tracer.judgment_tracer_provider import JudgmentTracerProvider

    with patch("judgeval.v1.utils.resolve_project_id", return_value="project-1"):
        tracer = Tracer(
            project_name="test_project",
            enable_evaluation=False,
            enable_monitoring=False,  # Disable to avoid network calls in tests
            api_client=mock_client,
            serializer=serializer,
            isolated=True,
        )

        # Get context via tracer's public method
        tracer_context = tracer.get_context()

        # Get context via provider's public method
        provider = tracer.tracer_provider
        assert isinstance(provider, JudgmentTracerProvider)
        provider_context = provider.get_isolated_current_context()

        # They should be the same object
        assert tracer_context is provider_context, (
            "tracer.get_context() should return the same context as provider.get_isolated_current_context()"
        )
