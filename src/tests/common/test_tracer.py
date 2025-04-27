import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from openai import OpenAI
from anthropic import Anthropic
import requests

from judgeval.common.tracer import Tracer, TraceEntry, wrap, current_span_var
from judgeval.judgment_client import JudgmentClient
from judgeval.common.exceptions import JudgmentAPIError

@pytest.fixture
def tracer(mocker):
    """Provide a configured tracer instance"""
    
    # Create the mock response for trace saving (POST)
    mock_post_response = mocker.Mock(spec=requests.Response)
    mock_post_response.status_code = 200
    mock_post_response.json.return_value = {
        "message": "Trace saved successfully",
        "trace_id": "test-trace-id"
    }
    
    # Create mocks for POST requests
    
    mock_post = mocker.patch('requests.post', autospec=True)
    mock_post.return_value = mock_post_response
    
    # Mock the JudgmentClient
    mock_judgment_client = mocker.Mock(spec=JudgmentClient)
    mocker.patch('judgeval.common.tracer.JudgmentClient', return_value=mock_judgment_client)
    
    yield Tracer(api_key=str(uuid4()), organization_id="test_org")

@pytest.fixture
def trace_client(tracer):
    """Provide a trace client instance"""
    with tracer.trace("test_trace") as client:
        yield client

def test_tracer_singleton(mocker):
    """Test that Tracer maintains singleton pattern"""
    # Clear any existing singleton instance first
    Tracer._instance = None
    
    # Mock the JudgmentClient
    mock_judgment_client = mocker.Mock(spec=JudgmentClient)
    mocker.patch('judgeval.common.tracer.JudgmentClient', return_value=mock_judgment_client)
    
    tracer1 = Tracer(api_key=str(uuid4()), organization_id="test_org")
    tracer2 = Tracer(api_key=str(uuid4()), organization_id="test_org")
    assert tracer1 is tracer2
    assert tracer1.api_key == tracer2.api_key

def test_tracer_requires_api_key():
    """Test that Tracer requires an API key"""
    # Clear any existing singleton instance first
    Tracer._instance = None
    
    with pytest.raises(ValueError):
        tracer = Tracer(api_key=None)
        print(tracer.api_key)

def test_trace_entry_print(capsys):
    """Test TraceEntry print formatting"""
    entries = [
        TraceEntry(type="enter", function="test_func", span_id="test-span-1", depth=1, message="test", created_at=0),
        TraceEntry(type="exit", function="test_func", span_id="test-span-1", depth=1, message="test", created_at=0, duration=0.5),
        TraceEntry(type="output", function="test_func", span_id="test-span-1", depth=1, message="test", created_at=0, output="result"),
        TraceEntry(type="input", function="test_func", span_id="test-span-1", depth=1, message="test", created_at=0, inputs={"arg": 1}),
    ]
    
    expected_outputs = [
        "  → test_func (id: test-span-1) (trace: test)\n",
        "  ← test_func (id: test-span-1) (0.500s)\n",
        "  Output (for id: test-span-1): result\n",
        "  Input (for id: test-span-1): {'arg': 1}\n",
    ]
    
    for entry, expected in zip(entries, expected_outputs):
        entry.print_entry()
        captured = capsys.readouterr()
        assert captured.out == expected

def test_trace_entry_to_dict():
    """Test TraceEntry serialization"""
    # Test basic serialization
    entry = TraceEntry(
        type="enter",
        function="test_func",
        span_id="test-span-1",
        depth=1,
        message="test",
        created_at=0
    )
    data = entry.to_dict()
    assert data["type"] == "enter"
    assert data["function"] == "test_func"
    assert data["span_id"] == "test-span-1"

    # Test with non-serializable output
    class NonSerializable:
        pass

    non_serializable = NonSerializable()
    entry = TraceEntry(
        type="output",
        function="test_func",
        span_id="test-span-2",
        depth=1,
        message="test",
        created_at=0,
        output=non_serializable
    )
    
    data = entry.to_dict()
    assert data["output"] == non_serializable.__str__()

def test_trace_client_span(trace_client):
    """Test span context manager"""
    # The trace_client fixture starts with a trace "test_trace" and its 'enter' entry
    initial_entries_count = len(trace_client.entries)
    assert initial_entries_count == 1 # Should only have the 'enter' for "test_trace"

    parent_before_span = current_span_var.get() # Should be the span_id of test_trace

    with trace_client.span("test_span") as span:
        # Inside the span, the current span var should be updated to the new span_id
        current_span_id = current_span_var.get()
        assert current_span_id is not None
        # Check the 'enter' entry for the new span
        enter_entry = trace_client.entries[-1]
        assert enter_entry.type == "enter"
        assert enter_entry.function == "test_span"
        assert enter_entry.span_id == current_span_id
        assert enter_entry.parent_span_id == parent_before_span # Check parent relationship
        assert enter_entry.depth == 1 # Depth relative to parent

    # After the span, the context var should be reset
    assert current_span_var.get() == parent_before_span

    # Check the 'exit' entry
    exit_entry = trace_client.entries[-1]
    assert exit_entry.type == "exit"
    assert exit_entry.function == "test_span"
    assert exit_entry.span_id == current_span_id
    assert exit_entry.depth == 1 # Depth after exiting is the same as the entry depth for that span

    # Check total entries (1 enter root + 1 enter span + 1 exit span)
    assert len(trace_client.entries) == initial_entries_count + 2

def test_trace_client_nested_spans(trace_client):
    """Test nested spans maintain proper depth recorded in entries"""
    root_span_id = current_span_var.get() # From the fixture

    with trace_client.span("outer") as outer_span:
        outer_span_id = current_span_var.get()
        # Check 'enter' entry for 'outer' span
        outer_enter_entry = trace_client.entries[-1]
        assert outer_enter_entry.type == "enter"
        assert outer_enter_entry.function == "outer"
        assert outer_enter_entry.span_id == outer_span_id
        assert outer_enter_entry.parent_span_id == root_span_id
        assert outer_enter_entry.depth == 1 # Depth is 0(root) + 1

        with trace_client.span("inner") as inner_span:
            inner_span_id = current_span_var.get()
            # Check 'enter' entry for 'inner' span
            inner_enter_entry = trace_client.entries[-1]
            assert inner_enter_entry.type == "enter"
            assert inner_enter_entry.function == "inner"
            assert inner_enter_entry.span_id == inner_span_id
            assert inner_enter_entry.parent_span_id == outer_span_id
            assert inner_enter_entry.depth == 2 # Depth is 1(outer) + 1

        # Check 'exit' entry for 'inner' span
        inner_exit_entry = trace_client.entries[-1]
        assert inner_exit_entry.type == "exit"
        assert inner_exit_entry.function == "inner"
        assert inner_exit_entry.span_id == inner_span_id
        assert inner_exit_entry.depth == 2 # Depth when exiting inner is inner's entry depth

    # Check 'exit' entry for 'outer' span
    outer_exit_entry = trace_client.entries[-1]
    assert outer_exit_entry.type == "exit"
    assert outer_exit_entry.function == "outer"
    assert outer_exit_entry.span_id == outer_span_id
    assert outer_exit_entry.depth == 1 # Depth when exiting outer is outer's entry depth

def test_record_input_output(trace_client):
    """Test recording inputs and outputs"""
    with trace_client.span("test_span") as span:
        span_id = current_span_var.get()
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")
    
    # Filter entries to only include those for the current span
    entries = [e for e in trace_client.entries if e.span_id == span_id]
    assert [e.type for e in entries] == ["enter", "input", "output", "exit"]
    
    # Verify the input and output entries have the correct span_id
    input_entry = next(e for e in entries if e.type == "input")
    output_entry = next(e for e in entries if e.type == "output")
    assert input_entry.span_id == span_id
    assert output_entry.span_id == span_id
    assert input_entry.inputs == {"arg": 1}
    assert output_entry.output == "result"

def test_condense_trace(trace_client):
    """Test trace condensing functionality"""
    # Store the base depth from the enter event
    base_depth = 0
    span_id = "test-span-1"
    entries = [
        {
            "type": "enter",
            "function": "test_func",
            "span_id": span_id,
            "trace_id": trace_client.trace_id,
            "depth": base_depth,
            "created_at": "2024-01-01T00:00:01.000000",
            "parent_span_id": None
        },
        {
            "type": "input",
            "function": "test_func",
            "span_id": span_id,
            "trace_id": trace_client.trace_id,
            "depth": base_depth + 1,
            "created_at": "2024-01-01T00:00:01.100000",
            "inputs": {"x": 1}
        },
        {
            "type": "output",
            "function": "test_func",
            "span_id": span_id,
            "trace_id": trace_client.trace_id,
            "depth": base_depth + 1,
            "created_at": "2024-01-01T00:00:01.200000",
            "output": "result"
        },
        {
            "type": "exit",
            "function": "test_func",
            "span_id": span_id,
            "trace_id": trace_client.trace_id,
            "depth": base_depth,
            "created_at": "2024-01-01T00:00:02.000000"
        },
    ]
    
    condensed, evals = trace_client.condense_trace(entries)
    print(f"{condensed=}")
    # Test that the condensed entry's depth matches the enter event's depth
    assert len(condensed) == 1
    assert condensed[0]["function"] == "test_func"
    assert condensed[0]["span_id"] == span_id
    assert condensed[0]["depth"] == entries[0]["depth"]  # Should match the input event's depth
    assert condensed[0]["inputs"] == {"x": 1}
    assert condensed[0]["output"] == "result"
    assert condensed[0]["duration"] == 1.0

@patch('requests.post')
def test_save_trace(mock_post, trace_client):
    """Test saving trace data"""
    # Configure mock response properly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "success"}'
    mock_response.json.return_value = {"ui_results_url": "http://example.com/results"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    
    with trace_client.span("test_span"):
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")
    
    trace_id, data = trace_client.save()
    assert mock_post.called
    assert data["trace_id"] == trace_client.trace_id

@patch('requests.post')
def test_wrap_openai(mock_post, tracer):
    """Test wrapping OpenAI client"""
    # Configure mock response properly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "success"}'
    mock_post.return_value = mock_response
    
    client = OpenAI()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="test response"))]
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    client.chat.completions.create = MagicMock(return_value=mock_completion)
    
    wrapped_client = wrap(client)
    
    with tracer.trace("test_trace"):
        response = wrapped_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}]
        )
    
    assert response == mock_completion

@patch('requests.post')
def test_wrap_anthropic(mock_post, tracer):
    """Test wrapping Anthropic client"""
    # Configure mock response properly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "success"}'
    mock_post.return_value = mock_response
    
    client = Anthropic()
    mock_completion = MagicMock()
    mock_completion.content = [MagicMock(text="test response")]
    mock_completion.usage = MagicMock(input_tokens=10, output_tokens=20)
    client.messages.create = MagicMock(return_value=mock_completion)
    
    wrapped_client = wrap(client)
    
    with tracer.trace("test_trace"):
        response = wrapped_client.messages.create(
            model="claude-3",
            messages=[{"role": "user", "content": "test"}]
        )
    
    assert response == mock_completion

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
    JudgmentClient._instances = {}  # Clear JudgmentClient singleton too
    
    # Directly patch the _validate_api_key method in JudgmentClient
    mocker.patch('judgeval.judgment_client.JudgmentClient._validate_api_key',
                return_value=(False, "API key is invalid"))
    
    # Now when Tracer tries to initialize JudgmentClient, it will receive our mocked result
    with pytest.raises(JudgmentAPIError, match="Issue with passed in Judgment API key: API key is invalid"):
        Tracer(api_key="invalid_key", organization_id="test_org")

def test_observe_decorator(tracer):
    """Test the @tracer.observe decorator"""
    @tracer.observe
    def test_function(x, y):
        return x + y
    
    with tracer.trace("test_trace"):
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

@patch('requests.post')
def test_wrap_openai_responses_api(mock_post, tracer):
    """Test wrapping OpenAI responses API"""
    # Configure mock response properly
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = '{"message": "success"}'
    mock_post.return_value = mock_response
    
    client = OpenAI()
    # Create mock for responses.create method
    mock_responses_result = MagicMock()
    mock_responses_result.output = [MagicMock(type="text", text="test response")]
    mock_responses_result.usage = MagicMock(prompt_tokens=15, completion_tokens=25, total_tokens=40)
    
    # Mock the responses.create method
    client.responses = MagicMock()
    original_mock = MagicMock(return_value=mock_responses_result)
    client.responses.create = original_mock
    
    wrapped_client = wrap(client)
    
    # Test that the responses.create method is wrapped correctly
    with tracer.trace("test_response_api"):
        response = wrapped_client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        )
    
    # Verify the response is correctly passed through
    assert response == mock_responses_result
    
    # Verify that the original mock was called - check by examining call count
    assert original_mock.call_count == 1, "responses.create should have been called exactly once"
