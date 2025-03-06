import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4
from openai import OpenAI
from anthropic import Anthropic
import requests

from judgeval.common.tracer import Tracer, TraceEntry, wrap
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
        TraceEntry(type="enter", function="test_func", depth=1, message="test", timestamp=0),
        TraceEntry(type="exit", function="test_func", depth=1, message="test", timestamp=0, duration=0.5),
        TraceEntry(type="output", function="test_func", depth=1, message="test", timestamp=0, output="result"),
        TraceEntry(type="input", function="test_func", depth=1, message="test", timestamp=0, inputs={"arg": 1}),
    ]
    
    expected_outputs = [
        "  → test_func (trace: test)\n",
        "  ← test_func (0.500s)\n",
        "  Output: result\n",
        "  Input: {'arg': 1}\n",
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
        depth=1,
        message="test",
        timestamp=0
    )
    data = entry.to_dict()
    assert data["type"] == "enter"
    assert data["function"] == "test_func"

    # Test with non-serializable output
    class NonSerializable:
        pass

    non_serializable = NonSerializable()
    entry = TraceEntry(
        type="output",
        function="test_func",
        depth=1,
        message="test",
        timestamp=0,
        output=non_serializable
    )
    
    data = entry.to_dict()
    assert data["output"] == non_serializable.__str__()

def test_trace_client_span(trace_client):
    """Test span context manager"""
    initial_entries = len(trace_client.entries)  # Get initial count
    
    with trace_client.span("test_span") as span:
        assert trace_client._current_span == "test_span"
        assert len(trace_client.entries) == initial_entries + 1  # Compare to initial count
    
    assert len(trace_client.entries) == initial_entries + 2  # Account for both enter and exit
    assert trace_client.entries[-1].type == "exit"
    assert trace_client._current_span == "test_trace"

def test_trace_client_nested_spans(trace_client):
    """Test nested spans maintain proper depth"""
    with trace_client.span("outer"):
        assert trace_client.tracer.depth == 2  # 1 for trace + 1 for span
        with trace_client.span("inner"):
            assert trace_client.tracer.depth == 3
        assert trace_client.tracer.depth == 2
    assert trace_client.tracer.depth == 1

def test_record_input_output(trace_client):
    """Test recording inputs and outputs"""
    with trace_client.span("test_span"):
        trace_client.record_input({"arg": 1})
        trace_client.record_output("result")
    
    # Filter entries to only include those for the current span
    entries = [e.type for e in trace_client.entries if e.function == "test_span"]
    assert entries == ["enter", "input", "output", "exit"]

def test_condense_trace(trace_client):
    """Test trace condensing functionality"""
    # Store the base depth from the enter event
    base_depth = 0
    entries = [
        {"type": "enter", "function": "test_func", "depth": base_depth, "timestamp": 1.0},
        {"type": "input", "function": "test_func", "depth": base_depth + 1, "timestamp": 1.1, "inputs": {"x": 1}},
        {"type": "output", "function": "test_func", "depth": base_depth + 1, "timestamp": 1.2, "output": "result"},
        {"type": "exit", "function": "test_func", "depth": base_depth, "timestamp": 2.0},
    ]
    
    condensed = trace_client.condense_trace(entries)
    print(f"{condensed=}")
    # Test that the condensed entry's depth matches the enter event's depth
    assert len(condensed) == 1
    assert condensed[0]["function"] == "test_func"
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
    
    # Create the mock response for invalid API key
    mock_post_response = mocker.Mock(spec=requests.Response)
    mock_post_response.status_code = 401
    mock_post_response.json.return_value = {"detail": "API key is invalid"}
    mock_post_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    
    # Create mock for POST request
    mock_post = mocker.patch('requests.post', autospec=True)
    mock_post.return_value = mock_post_response
    
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
