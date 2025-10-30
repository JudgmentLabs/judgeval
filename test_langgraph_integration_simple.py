"""
Simple test to verify LangGraph callback handler integration.

This test verifies the callback handler's logic without requiring
a fully initialized tracer or API keys.
"""

import sys
import warnings
from typing import Any, Dict
from uuid import uuid4, UUID

from langchain_core.outputs import LLMResult, Generation
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import get_current_span

# Import after setting up environment
import os
os.environ["JUDGMENT_API_KEY"] = "test-key"  
os.environ["JUDGMENT_ORG_ID"] = "test-org"

from judgeval.tracer import Tracer
from judgeval.integrations.langgraph import Langgraph, LangGraphCallbackHandler


# Track any warnings about SpanContext
span_context_warnings = []

def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """Capture warnings about SpanContext."""
    warning_str = str(message)
    if "SpanContext" in warning_str or "invalid" in warning_str.lower():
        span_context_warnings.append(warning_str)
        print(f"⚠️  SPAN CONTEXT WARNING: {warning_str}")

warnings.showwarning = custom_warning_handler


def test_callback_handler_methods():
    """Test that all callback handler methods work without errors."""
    print("\n" + "="*80)
    print("TEST 1: Callback Handler Methods")
    print("="*80)
    
    # Set up a basic TracerProvider
    resource = Resource.create({"service.name": "test-service"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    # Create a minimal tracer
    class MockTracer:
        def get_tracer(self):
            return trace.get_tracer(__name__)
        
        def get_current_agent_context(self):
            from contextvars import ContextVar
            return ContextVar("test", default=None)
    
    mock_tracer = MockTracer()
    handler = LangGraphCallbackHandler(tracer=mock_tracer, verbose=True)
    
    print("✓ Callback handler created")
    
    # Test chain callbacks
    print("\nTesting chain callbacks...")
    chain_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "TestChain"},
        inputs={"query": "test"},
        run_id=chain_run_id,
    )
    print("  ✓ on_chain_start")
    
    handler.on_chain_end(
        outputs={"result": "test result"},
        run_id=chain_run_id,
    )
    print("  ✓ on_chain_end")
    
    # Test tool callbacks
    print("\nTesting tool callbacks...")
    tool_run_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "get_stock_price"},
        input_str="AAPL",
        run_id=tool_run_id,
    )
    print("  ✓ on_tool_start")
    
    handler.on_tool_end(
        output="$175.50",
        run_id=tool_run_id,
    )
    print("  ✓ on_tool_end")
    
    # Test LLM callbacks
    print("\nTesting LLM callbacks...")
    llm_run_id = uuid4()
    handler.on_llm_start(
        serialized={"name": "gpt-4o-mini"},
        prompts=["What is the weather?"],
        run_id=llm_run_id,
        invocation_params={"temperature": 0.7},
    )
    print("  ✓ on_llm_start")
    
    result = LLMResult(generations=[[Generation(text="It's sunny")]])
    handler.on_llm_end(
        response=result,
        run_id=llm_run_id,
    )
    print("  ✓ on_llm_end")
    
    # Test error handling
    print("\nTesting error handling...")
    error_run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "ErrorChain"},
        inputs={"query": "test"},
        run_id=error_run_id,
    )
    handler.on_chain_error(
        error=Exception("Test error"),
        run_id=error_run_id,
    )
    print("  ✓ Error handling")
    
    print("\n✓ All callback handler methods work correctly")


def test_span_tracking():
    """Test that the handler properly tracks spans."""
    print("\n" + "="*80)
    print("TEST 2: Span Tracking")
    print("="*80)
    
    resource = Resource.create({"service.name": "test-service"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    class MockTracer:
        def get_tracer(self):
            return trace.get_tracer(__name__)
        
        def get_current_agent_context(self):
            from contextvars import ContextVar
            return ContextVar("test", default=None)
    
    mock_tracer = MockTracer()
    handler = LangGraphCallbackHandler(tracer=mock_tracer, verbose=False)
    
    # Verify spans are tracked
    run_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "TestChain"},
        inputs={"test": "input"},
        run_id=run_id,
    )
    
    run_id_str = str(run_id)
    assert run_id_str in handler._run_spans, "Span should be tracked"
    assert run_id_str in handler._context_tokens, "Context token should be tracked"
    print("✓ Span is tracked after start")
    
    handler.on_chain_end(
        outputs={"result": "output"},
        run_id=run_id,
    )
    
    assert run_id_str not in handler._run_spans, "Span should be cleaned up"
    assert run_id_str not in handler._context_tokens, "Context token should be cleaned up"
    print("✓ Span is cleaned up after end")
    
    print("\n✓ Span tracking works correctly")


def test_context_propagation():
    """Test that context is properly propagated."""
    print("\n" + "="*80)
    print("TEST 3: Context Propagation")
    print("="*80)
    
    resource = Resource.create({"service.name": "test-service"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    class MockTracer:
        def get_tracer(self):
            return trace.get_tracer(__name__)
        
        def get_current_agent_context(self):
            from contextvars import ContextVar
            return ContextVar("test", default=None)
    
    mock_tracer = MockTracer()
    handler = LangGraphCallbackHandler(tracer=mock_tracer, verbose=False)
    
    otel_tracer = mock_tracer.get_tracer()
    
    # Create a parent span
    with otel_tracer.start_as_current_span("parent") as parent_span:
        parent_context = parent_span.get_span_context()
        parent_trace_id = parent_context.trace_id
        
        print(f"Parent trace_id: {format(parent_trace_id, '032x')}")
        
        # Start a chain event (which should create a child span)
        run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "TestChain"},
            inputs={"query": "test"},
            run_id=run_id,
        )
        
        # Get the span that was created
        run_id_str = str(run_id)
        if run_id_str in handler._run_spans:
            child_span = handler._run_spans[run_id_str]
            child_context = child_span.get_span_context()
            child_trace_id = child_context.trace_id
            
            print(f"Child trace_id:  {format(child_trace_id, '032x')}")
            
            if child_trace_id == parent_trace_id:
                print("✓ Child span has same trace_id as parent (correct!)")
            else:
                print("✗ Child span has different trace_id (context not propagated)")
                sys.exit(1)
        else:
            print("✗ Child span not created")
            sys.exit(1)
        
        # End the chain
        handler.on_chain_end(
            outputs={"result": "test"},
            run_id=run_id,
        )
    
    print("\n✓ Context propagation works correctly")


def test_nested_spans():
    """Test nested span relationships."""
    print("\n" + "="*80)
    print("TEST 4: Nested Spans")
    print("="*80)
    
    resource = Resource.create({"service.name": "test-service"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    
    class MockTracer:
        def get_tracer(self):
            return trace.get_tracer(__name__)
        
        def get_current_agent_context(self):
            from contextvars import ContextVar
            return ContextVar("test", default=None)
    
    mock_tracer = MockTracer()
    handler = LangGraphCallbackHandler(tracer=mock_tracer, verbose=False)
    
    otel_tracer = mock_tracer.get_tracer()
    
    # Create root -> chain -> tool hierarchy
    with otel_tracer.start_as_current_span("root") as root_span:
        root_trace_id = root_span.get_span_context().trace_id
        
        # Start chain
        chain_run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "AgentChain"},
            inputs={"query": "test"},
            run_id=chain_run_id,
        )
        
        # Start tool within chain
        tool_run_id = uuid4()
        handler.on_tool_start(
            serialized={"name": "get_stock_price"},
            input_str="AAPL",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )
        
        # Check tool span has same trace_id
        tool_span = handler._run_spans.get(str(tool_run_id))
        if tool_span:
            tool_trace_id = tool_span.get_span_context().trace_id
            if tool_trace_id == root_trace_id:
                print("✓ Tool span maintains root trace_id")
            else:
                print("✗ Tool span lost trace context")
                sys.exit(1)
        
        # End tool
        handler.on_tool_end(
            output="$175.50",
            run_id=tool_run_id,
        )
        
        # End chain
        handler.on_chain_end(
            outputs={"result": "Price is $175.50"},
            run_id=chain_run_id,
        )
    
    print("\n✓ Nested spans work correctly")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LANGGRAPH CALLBACK HANDLER INTEGRATION TEST")
    print("="*80)
    print("\nVerifying:")
    print("1. Callback handler methods work without errors")
    print("2. Spans are properly tracked and cleaned up")
    print("3. Context is propagated correctly")
    print("4. Nested operations maintain trace context")
    print("5. No SpanContext warnings appear")
    
    try:
        test_callback_handler_methods()
        test_span_tracking()
        test_context_propagation()
        test_nested_spans()
        
        # Check for SpanContext warnings
        print("\n" + "="*80)
        print("CHECKING FOR SPAN CONTEXT WARNINGS")
        print("="*80)
        
        if span_context_warnings:
            print(f"\n❌ FOUND {len(span_context_warnings)} WARNING(S):")
            for warning in span_context_warnings:
                print(f"  - {warning}")
            sys.exit(1)
        else:
            print("\n✓ No SpanContext warnings detected")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nIntegration Status:")
        print("  ✓ LangGraph callback handler functional")
        print("  ✓ OpenTelemetry spans created correctly")
        print("  ✓ Context propagation working")
        print("  ✓ Nested operations maintain trace context")
        print("  ✓ NO SPAN CONTEXT ERRORS")
        print("\nThe LangGraph integration is ready to use!")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
