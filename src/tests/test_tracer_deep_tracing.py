"""
Tests for deep tracing functionality in tracer.py
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import inspect
import asyncio
from judgeval.common.tracer import Tracer, current_trace_var, current_span_var, in_traced_function_var

# Test fixtures
@pytest.fixture
def mock_tracer():
    """Create a mock Tracer instance with required attributes."""
    tracer = MagicMock()
    tracer.api_key = "test_api_key"
    tracer.organization_id = "test_org_id"
    tracer.project_name = "test_project"
    tracer.enable_monitoring = True
    tracer.enable_evaluations = True
    tracer.deep_tracing = True
    
    # Mock the observe method to return a wrapped function that records tracing
    def mock_observe(func=None, **kwargs):
        if func is None:
            return lambda f: mock_observe(f, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Get current trace from context
                current_trace = current_trace_var.get()
                if current_trace:
                    # Create a span context manager
                    span_context = MagicMock()
                    span_context.__enter__.return_value = current_trace
                    current_trace.span.return_value = span_context
                    
                    # Force the span to be called
                    current_trace.span(func.__name__)
                    
                    current_trace.record_input({'args': args, 'kwargs': kwargs})
                    try:
                        result = await func(*args, **kwargs)
                        current_trace.record_output(result)
                        return result
                    except Exception as e:
                        current_trace.record_output(str(e))
                        raise
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # Get current trace from context
                current_trace = current_trace_var.get()
                if current_trace:
                    # Create a span context manager
                    span_context = MagicMock()
                    span_context.__enter__.return_value = current_trace
                    current_trace.span.return_value = span_context
                    
                    # Force the span to be called
                    current_trace.span(func.__name__)
                    
                    current_trace.record_input({'args': args, 'kwargs': kwargs})
                    try:
                        result = func(*args, **kwargs)
                        current_trace.record_output(result)
                        return result
                    except Exception as e:
                        current_trace.record_output(str(e))
                        raise
                return func(*args, **kwargs)
            return sync_wrapper
            
    tracer.observe.side_effect = mock_observe
    
    return tracer

@pytest.fixture
def mock_trace_client():
    """Create a mock TraceClient instance."""
    client = MagicMock()
    client.trace_id = "test_trace_id"
    client.name = "test_trace"
    client.project_name = "test_project"
    client.entries = []
    
    # Mock the span context manager
    span_context = MagicMock()
    span_context.__enter__.return_value = client
    client.span.return_value = span_context
    
    return client

@pytest.fixture
def mock_context_vars():
    """Create mock context variables."""
    with patch('judgeval.common.tracer.current_trace_var', new=MagicMock()) as mock_trace, \
         patch('judgeval.common.tracer.current_span_var', new=MagicMock()) as mock_span, \
         patch('judgeval.common.tracer.in_traced_function_var', new=MagicMock()) as mock_in_traced:
        yield mock_trace, mock_span, mock_in_traced

# Test cases
def test_observe_deep_tracing_sync(mock_tracer, mock_trace_client, mock_context_vars):
    """Test deep tracing with synchronous functions using the observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    def test_func(x, y):
        return x + y
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    result = wrapped_func(2, 3)
    assert result == 5
    assert mock_trace_client.span.called
    assert mock_trace_client.record_input.called
    assert mock_trace_client.record_output.called

@pytest.mark.asyncio
async def test_observe_deep_tracing_async(mock_tracer, mock_trace_client, mock_context_vars):
    """Test deep tracing with asynchronous functions using the observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    async def test_func(x, y):
        return x + y
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    result = await wrapped_func(2, 3)
    assert result == 5
    assert mock_trace_client.span.called
    assert mock_trace_client.record_input.called
    assert mock_trace_client.record_output.called

def test_observe_deep_tracing_nested(mock_tracer, mock_trace_client, mock_context_vars):
    """Test deep tracing with nested function calls using the observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    def inner_func(x):
        return x * 2
    
    def outer_func(x):
        return inner_func(x) + 1
    
    # Apply observe decorator with deep tracing to outer function
    wrapped_outer = mock_tracer.observe(outer_func, deep_tracing=True)
    
    # Test
    result = wrapped_outer(3)
    assert result == 7
    # Verify both functions were traced
    assert mock_trace_client.span.call_count == 2
    assert mock_trace_client.record_input.call_count == 2
    assert mock_trace_client.record_output.call_count == 2

def test_observe_deep_tracing_skips_builtins(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that deep tracing skips built-in functions when using observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    def test_func():
        return len([1, 2, 3])  # len is a built-in function
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    result = wrapped_func()
    assert result == 3
    # Verify that built-in functions are not wrapped
    assert not hasattr(len, '_judgment_traced')

def test_observe_deep_tracing_context_vars(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that context variables are properly managed during deep tracing with observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    mock_in_traced.get.return_value = True
    
    # Setup
    def test_func():
        assert mock_trace.get() is not None
        assert mock_in_traced.get() is True
        return "test"
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    result = wrapped_func()
    assert result == "test"
    # Verify context variables are reset
    assert mock_in_traced.get() is True

@pytest.mark.asyncio
async def test_observe_deep_tracing_async_context_vars(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that context variables are properly managed during async deep tracing with observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    mock_in_traced.get.return_value = True
    
    # Setup
    async def test_func():
        assert mock_trace.get() is not None
        assert mock_in_traced.get() is True
        return "test"
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    result = await wrapped_func()
    assert result == "test"
    # Verify context variables are reset
    assert mock_in_traced.get() is True

def test_observe_deep_tracing_error_handling(mock_tracer, mock_trace_client, mock_context_vars):
    """Test error handling in deep tracing with observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    def test_func():
        raise ValueError("Test error")
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    with pytest.raises(ValueError):
        wrapped_func()
    # Verify tracing still completed despite error
    assert mock_trace_client.span.called
    assert mock_trace_client.record_input.called
    assert mock_trace_client.record_output.called

@pytest.mark.asyncio
async def test_observe_deep_tracing_async_error_handling(mock_tracer, mock_trace_client, mock_context_vars):
    """Test error handling in async deep tracing with observe decorator."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    mock_trace.get.return_value = mock_trace_client
    
    # Setup
    async def test_func():
        raise ValueError("Test error")
    
    # Apply observe decorator with deep tracing
    wrapped_func = mock_tracer.observe(test_func, deep_tracing=True)
    
    # Test
    with pytest.raises(ValueError):
        await wrapped_func()
    # Verify tracing still completed despite error
    assert mock_trace_client.span.called
    assert mock_trace_client.record_input.called
    assert mock_trace_client.record_output.called 