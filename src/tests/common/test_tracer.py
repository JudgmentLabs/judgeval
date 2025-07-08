import pytest
from unittest.mock import Mock, patch
from uuid import uuid4
from requests import Response

from judgeval.common.tracer import (
    Tracer,
    wrap,
    current_span_var,
    current_trace_var,
    TraceClient,
)
from judgeval.common.exceptions import JudgmentAPIError
from judgeval.data.trace import TraceSpan


@pytest.fixture
def tracer(mocker):
    """Provide a configured tracer instance"""

    # Create the mock response for trace saving (POST)
    mock_post_response = mocker.Mock(spec=Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "message": "Trace saved successfully",
        "trace_id": "test-trace-id",
        "ui_results_url": "http://example.com/results",
    }

    # Create mocks for POST requests
    mock_post = mocker.patch("judgeval.utils.requests.requests.post", autospec=True)
    mock_post.return_value = mock_post_response

    with patch("judgeval.common.tracer.BackgroundSpanService"):
        yield Tracer(api_key=str(uuid4()), organization_id="test_org")


@pytest.fixture
def trace_client(tracer):
    """Provide a trace client instance"""
    # Create a new trace client directly
    trace_id = str(uuid4())
    with patch("judgeval.common.tracer.BackgroundSpanService"):
        trace_client = TraceClient(
            tracer=tracer,
            trace_id=trace_id,
            name="test_trace",
            project_name="test_project",
        )
    # Set the trace context
    token = current_trace_var.set(trace_client)

    try:
        # Create a root span without recording any data
        with trace_client.span("root_span", span_type="test"):
            yield trace_client
    finally:
        # Clean up the trace context
        current_trace_var.reset(token)


def test_tracer_requires_api_key():
    """Test that Tracer requires an API key"""
    # Clear any existing singleton instance first
    Tracer._instance = None

    with pytest.raises(ValueError):
        tracer = Tracer(api_key=None)
        print(tracer.api_key)


def test_trace_span_to_dict():
    """Test TraceSpan serialization"""
    # Test basic serialization
    span = TraceSpan(
        span_id="test-span-1",
        trace_id="test-trace-id",
        depth=1,
        created_at=0,
        duration=0.5,
        inputs={"arg": 1},
        output="result",
        function="test_func",
        span_type="test-span",
        parent_span_id="test-parent-span-id",
    )
    data = span.model_dump()
    assert data["span_type"] == "test-span"
    assert data["trace_id"] == "test-trace-id"
    assert data["depth"] == 1
    assert data["created_at"] == "1970-01-01T00:00:00+00:00"
    assert data["duration"] == 0.5
    assert data["inputs"] == {"arg": 1}
    assert data["output"] == "result"
    assert data["function"] == "test_func"
    assert data["span_id"] == "test-span-1"
    assert data["parent_span_id"] == "test-parent-span-id"
    assert not data["has_evaluation"]  # Verify default value

    # Test with has_evaluation set to True
    span.has_evaluation = True
    data = span.model_dump()
    assert data["has_evaluation"]  # Verify updated value


def test_trace_client_span(trace_client):
    """Test span context manager"""
    # The trace_client fixture starts with a trace "test_trace" and its 'enter' entry
    initial_spans_count = len(trace_client.trace_spans)
    assert initial_spans_count == 1  # Should only have the 'enter' for "test_trace"

    parent_before_span = current_span_var.get()  # Should be the span_id of test_trace

    with trace_client.span("test_span"):
        # Inside the span, the current span var should be updated to the new span_id
        current_span_id = current_span_var.get()
        assert current_span_id is not None
        # Check the 'enter' entry for the new span
        new_span = trace_client.trace_spans[-1]
        assert new_span.function == "test_span"
        assert new_span.span_id == current_span_id
        assert (
            new_span.parent_span_id == parent_before_span
        )  # Check parent relationship
        assert new_span.depth == 1  # Depth relative to parent

    # After the span, the context var should be reset
    assert current_span_var.get() == parent_before_span

    # Check total spans (1 parent span + 1 child span)
    assert len(trace_client.trace_spans) == initial_spans_count + 1


def test_trace_client_nested_spans(trace_client):
    """Test nested spans maintain proper depth recorded in trace_spans"""
    root_span_id = current_span_var.get()  # From the fixture

    with trace_client.span("outer") as outer_span:
        outer_span_id = current_span_var.get()
        # Check 'enter' entry for 'outer' span
        outer_span = trace_client.trace_spans[-1]
        assert outer_span.span_type == "span"
        assert outer_span.function == "outer"
        assert outer_span.span_id == outer_span_id
        assert outer_span.parent_span_id == root_span_id
        assert outer_span.depth == 1  # Depth is 0(root) + 1

        with trace_client.span("inner") as inner_span:
            inner_span_id = current_span_var.get()
            # Check 'enter' entry for 'inner' span
            inner_span = trace_client.trace_spans[-1]
            assert inner_span.span_type == "span"
            assert inner_span.function == "inner"
            assert inner_span.span_id == inner_span_id
            assert inner_span.parent_span_id == outer_span_id
            assert inner_span.depth == 2  # Depth is 1(outer) + 1


@patch("judgeval.utils.requests.requests.post")
def test_save_trace(mock_post, trace_client):
    """Test saving trace data"""
    # Configure mock response properly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "success"}'
    mock_response.json.return_value = {
        "ui_results_url": "http://example.com/results",
        "trace_id": trace_client.trace_id,
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    with trace_client.span("test_span"):
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")

    trace_id, data = trace_client.save()
    assert mock_post.called
    assert data["trace_id"] == trace_client.trace_id


def test_wrap_unsupported_client(tracer):
    """Test wrapping unsupported client type"""

    class UnsupportedClient:
        pass

    with pytest.raises(ValueError):
        wrap(UnsupportedClient())


def test_tracer_invalid_api_key(mocker):
    """Test that Tracer handles invalid API keys"""
    # Clear the singleton instance first
    Tracer._instance = None

    # Now when Tracer tries to initialize JudgmentClient, it will receive our mocked result
    with patch(
        "judgeval.common.tracer.validate_api_key",
        return_value=(False, "Invalid API key"),
    ):
        with pytest.raises(
            JudgmentAPIError,
            match="Issue with passed in Judgment API key: Invalid API key",
        ):
            Tracer(api_key="invalid_key", organization_id="test_org")


def test_observe_decorator(tracer):
    """Test the @tracer.observe decorator"""

    @tracer.observe
    def test_function(x, y):
        return x + y

    result = test_function(1, 2)
    assert result == 3


def test_observe_decorator_with_error(tracer):
    """Test decorator error handling"""

    @tracer.observe
    def failing_function():
        raise ValueError("Test error")

    with tracer.trace("test_trace"):
        with pytest.raises(ValueError):
            failing_function()


def test_async_evaluate_sets_has_evaluation_flag(trace_client):
    """Test that async_evaluate sets has_evaluation flag on the span"""
    from judgeval.scorers import AnswerCorrectnessScorer
    from judgeval.data import Example

    # Create a span and get its span_id
    with trace_client.span("test_evaluation_span"):
        current_span_id = current_span_var.get()

        # Get the actual span object
        test_span = trace_client.span_id_to_span[current_span_id]

        # Verify has_evaluation is initially False
        assert not test_span.has_evaluation

        # Create a mock example and scorer
        example = Example(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
        )
        scorers = [AnswerCorrectnessScorer(threshold=0.9)]

        # Call async_evaluate
        trace_client.async_evaluate(
            scorers=scorers,
            example=example,
            model="gpt-4o-mini",
            span_id=current_span_id,
        )

        # Verify has_evaluation is now True
        assert test_span.has_evaluation
