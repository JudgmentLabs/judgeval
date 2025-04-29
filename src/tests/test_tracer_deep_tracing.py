"""
Tests for deep tracing functionality in tracer.py
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import inspect
import asyncio
import uuid
import functools
from judgeval.common.tracer import Tracer, current_trace_var, current_span_var, in_traced_function_var, _create_deep_tracing_wrapper

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
    
    # Create a more realistic observe method that stores custom attributes
    def mock_observe(func=None, *, name=None, span_type="span", deep_tracing=True, **kwargs):
        if func is None:
            return lambda f: mock_observe(f, name=name, span_type=span_type, deep_tracing=deep_tracing, **kwargs)
        
        # Store custom attributes on the function
        span_name = name or func.__name__
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Get current trace from context
                current_trace = mock_trace_var.get()
                
                # If no trace context, just call the function
                if not current_trace:
                    return await func(*args, **kwargs)
                
                # Create a span for this function call
                current_trace.span(span_name, span_type=span_type)
                
                # Record function call
                current_trace.record_input({'args': str(args), 'kwargs': kwargs})
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Record result
                current_trace.record_output(result)
                
                return result
            
            # Store custom attributes on the wrapper
            async_wrapper._judgment_span_name = span_name
            async_wrapper._judgment_span_type = span_type
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Get current trace from context
                current_trace = mock_trace_var.get()
                
                # If no trace context, just call the function
                if not current_trace:
                    return func(*args, **kwargs)
                
                # Create a span for this function call
                current_trace.span(span_name, span_type=span_type)
                
                # Record function call
                current_trace.record_input({'args': str(args), 'kwargs': kwargs})
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Record result
                current_trace.record_output(result)
                
                return result
            
            # Store custom attributes on the wrapper
            sync_wrapper._judgment_span_name = span_name
            sync_wrapper._judgment_span_type = span_type
            
            return sync_wrapper
    
    tracer.observe = mock_observe
    
    # Mock the _apply_deep_tracing method to simulate real behavior
    def mock_apply_deep_tracing(func, span_type="span"):
        # Create a simple module mock with the function
        module = MagicMock()
        module.__name__ = "test_module"
        
        # Add the function to the module
        setattr(module, func.__name__, func)
        
        # Create a traced version of the function
        traced_func = _create_deep_tracing_wrapper(func, tracer, "span")
        
        # Replace with traced version
        setattr(module, func.__name__, traced_func)
        
        return module, {func.__name__: func}
    
    tracer._apply_deep_tracing = mock_apply_deep_tracing
    
    return tracer

# Global mock for context vars
mock_trace_var = MagicMock()
mock_span_var = MagicMock()
mock_in_traced_var = MagicMock()

@pytest.fixture
def mock_trace_client():
    """Create a mock TraceClient instance."""
    client = MagicMock()
    client.trace_id = str(uuid.uuid4())
    client.name = "test_trace"
    client.project_name = "test_project"
    client.span_calls = []
    
    # Mock the span method to return a context manager and record calls
    def mock_span(name, span_type="span"):
        # Record the span call
        client.span_calls.append((name, span_type))
        
        # Create a span context manager
        span_context = MagicMock()
        span_context.__enter__ = MagicMock(return_value=span_context)
        span_context.__exit__ = MagicMock(return_value=None)
        
        return span_context
    
    client.span = mock_span
    
    # Mock record methods
    client.record_input = MagicMock()
    client.record_output = MagicMock()
    
    return client

@pytest.fixture
def mock_context_vars(mock_trace_client):
    """Create mock context variables with the trace client."""
    global mock_trace_var, mock_span_var, mock_in_traced_var
    
    with patch('judgeval.common.tracer.current_trace_var', mock_trace_var), \
         patch('judgeval.common.tracer.current_span_var', mock_span_var), \
         patch('judgeval.common.tracer.in_traced_function_var', mock_in_traced_var):
        
        # Set up the current trace
        mock_trace_var.get = MagicMock(return_value=mock_trace_client)
        mock_trace_var.set = MagicMock()
        mock_trace_var.reset = MagicMock()
        
        # Set up in_traced_function
        mock_in_traced_var.get = MagicMock(return_value=False)
        mock_in_traced_var.set = MagicMock()
        mock_in_traced_var.reset = MagicMock()
        
        yield mock_trace_var, mock_span_var, mock_in_traced_var

# Test cases
def test_custom_span_name_and_type(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that custom span name and type are respected in deep tracing."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Define a function with custom span name and type
    @mock_tracer.observe(name="custom_name", span_type="custom_type")
    def test_func():
        return "result"
    
    # Verify custom attributes are stored on the function
    assert hasattr(test_func, '_judgment_span_name')
    assert hasattr(test_func, '_judgment_span_type')
    assert test_func._judgment_span_name == "custom_name"
    assert test_func._judgment_span_type == "custom_type"
    
    # Call the function
    result = test_func()
    
    # Verify the result
    assert result == "result"
    
    # Verify the span was created with custom name and type
    assert len(mock_trace_client.span_calls) > 0
    assert mock_trace_client.span_calls[0] == ("custom_name", "custom_type")

def test_nested_functions_with_custom_spans(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that nested functions with custom span names and types are respected."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Define nested functions with custom attributes
    @mock_tracer.observe(name="outer_custom", span_type="outer_type")
    def outer_func():
        return inner_func()
    
    @mock_tracer.observe(name="inner_custom", span_type="inner_type")
    def inner_func():
        return "inner_result"
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Call the outer function
    result = outer_func()
    
    # Verify the result
    assert result == "inner_result"
    
    # Verify both spans were created with their custom names and types
    assert len(mock_trace_client.span_calls) == 2
    assert ("outer_custom", "outer_type") in mock_trace_client.span_calls
    assert ("inner_custom", "inner_type") in mock_trace_client.span_calls

@pytest.mark.asyncio
async def test_async_functions_with_custom_spans(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that async functions with custom span names and types are respected."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Define async functions with custom attributes
    @mock_tracer.observe(name="async_custom", span_type="async_type")
    async def async_func():
        return "async_result"
    
    # Call the async function
    result = await async_func()
    
    # Verify the result
    assert result == "async_result"
    
    # Verify the span was created with custom name and type
    assert len(mock_trace_client.span_calls) > 0
    assert ("async_custom", "async_type") in mock_trace_client.span_calls

def test_deep_tracing_with_custom_spans(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that deep tracing respects custom span names and types."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Define a module with functions
    def module_func1():
        return "result1"
    
    def module_func2():
        return "result2"
    
    # Add custom attributes to the functions
    module_func1._judgment_span_name = "custom_module_func1"
    module_func1._judgment_span_type = "module_type1"
    
    module_func2._judgment_span_name = "custom_module_func2"
    module_func2._judgment_span_type = "module_type2"
    
    # Create deep tracing wrappers
    wrapped_func1 = _create_deep_tracing_wrapper(module_func1, mock_tracer)
    wrapped_func2 = _create_deep_tracing_wrapper(module_func2, mock_tracer)
    
    # Call the wrapped functions
    result1 = wrapped_func1()
    result2 = wrapped_func2()
    
    # Verify the results
    assert result1 == "result1"
    assert result2 == "result2"
    
    # Verify the spans were created with custom names and types
    assert len(mock_trace_client.span_calls) == 2
    span_names = [name for name, _ in mock_trace_client.span_calls]
    span_types = [type_ for _, type_ in mock_trace_client.span_calls]
    
    assert "custom_module_func1" in span_names
    assert "custom_module_func2" in span_names
    assert "module_type1" in span_types or "span" in span_types
    assert "module_type2" in span_types or "span" in span_types

def test_error_handling_with_custom_spans(mock_tracer, mock_trace_client, mock_context_vars):
    """Test error handling with custom span names and types."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Define a function with custom span name and type that raises an error
    @mock_tracer.observe(name="error_custom", span_type="error_type")
    def error_func():
        raise ValueError("Test error")
    
    # Call the function and expect an error
    with pytest.raises(ValueError):
        error_func()
    
    # Verify the span was created with custom name and type
    assert len(mock_trace_client.span_calls) > 0
    assert ("error_custom", "error_type") in mock_trace_client.span_calls

@pytest.mark.asyncio
async def test_async_error_handling_with_custom_spans(mock_tracer, mock_trace_client, mock_context_vars):
    """Test async error handling with custom span names and types."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Define an async function with custom span name and type that raises an error
    @mock_tracer.observe(name="async_error_custom", span_type="async_error_type")
    async def async_error_func():
        raise ValueError("Test async error")
    
    # Call the function and expect an error
    with pytest.raises(ValueError):
        await async_error_func()
    
    # Verify the span was created with custom name and type
    assert len(mock_trace_client.span_calls) > 0
    assert ("async_error_custom", "async_error_type") in mock_trace_client.span_calls

def test_multiple_decorated_functions_in_same_trace(mock_tracer, mock_trace_client, mock_context_vars):
    """Test that multiple decorated functions in the same trace maintain their custom attributes."""
    mock_trace, mock_span, mock_in_traced = mock_context_vars
    
    # Reset span calls before test
    mock_trace_client.span_calls = []
    
    # Define multiple functions with different custom attributes
    @mock_tracer.observe(name="func1_name", span_type="func1_type")
    def func1():
        return func2()
    
    @mock_tracer.observe(name="func2_name", span_type="func2_type")
    def func2():
        return func3()
    
    @mock_tracer.observe(name="func3_name", span_type="func3_type")
    def func3():
        return "result"
    
    # Call the first function which calls the others
    result = func1()
    
    # Verify the result
    assert result == "result"
    
    # Verify all spans were created with their custom names and types
    assert len(mock_trace_client.span_calls) == 3
    assert ("func1_name", "func1_type") in mock_trace_client.span_calls
    assert ("func2_name", "func2_type") in mock_trace_client.span_calls
    assert ("func3_name", "func3_type") in mock_trace_client.span_calls