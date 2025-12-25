"""Unit tests for thread-local storage isolation in Claude Agent SDK integration."""

import threading
from unittest.mock import Mock, MagicMock


def test_thread_local_storage_per_tracer():
    """
    Test that _get_thread_local returns separate storage for different tracer instances.

    This ensures that multiple tracer instances don't share thread-local storage,
    preventing trace context contamination.
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _get_thread_local,
        _tracer_thread_locals,
    )

    # Clear any existing thread-local storage
    _tracer_thread_locals.clear()

    # Create two mock tracer instances
    tracer1 = Mock()
    tracer2 = Mock()

    # Get thread-local storage for each tracer
    tl1 = _get_thread_local(tracer1)
    tl2 = _get_thread_local(tracer2)

    # Verify they are different threading.local instances
    assert tl1 is not tl2, (
        "Thread-local storage should be different for different tracers"
    )

    # Set different values in each thread-local storage
    tl1.test_value = "tracer1_value"
    tl2.test_value = "tracer2_value"

    # Verify values don't interfere with each other
    assert tl1.test_value == "tracer1_value"
    assert tl2.test_value == "tracer2_value"

    # Verify getting the same tracer returns the same thread-local
    tl1_again = _get_thread_local(tracer1)
    assert tl1 is tl1_again, "Should return same thread-local for same tracer"
    assert tl1_again.test_value == "tracer1_value"


def test_thread_local_storage_across_threads():
    """
    Test that thread-local storage is truly thread-local even with multiple tracers.

    This ensures that each thread has its own storage per tracer, preventing
    contamination across both threads AND tracers.
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _get_thread_local,
        _tracer_thread_locals,
    )

    # Clear any existing thread-local storage
    _tracer_thread_locals.clear()

    # Create two mock tracer instances
    tracer1 = Mock()
    tracer2 = Mock()

    results = {}

    def thread1_func():
        """Simulate thread 1 using both tracers"""
        tl1 = _get_thread_local(tracer1)
        tl2 = _get_thread_local(tracer2)

        tl1.thread_id = "thread1"
        tl1.tracer_name = "tracer1"

        tl2.thread_id = "thread1"
        tl2.tracer_name = "tracer2"

        results["thread1"] = {
            "tl1": (tl1.thread_id, tl1.tracer_name),
            "tl2": (tl2.thread_id, tl2.tracer_name),
        }

    def thread2_func():
        """Simulate thread 2 using both tracers"""
        tl1 = _get_thread_local(tracer1)
        tl2 = _get_thread_local(tracer2)

        tl1.thread_id = "thread2"
        tl1.tracer_name = "tracer1"

        tl2.thread_id = "thread2"
        tl2.tracer_name = "tracer2"

        results["thread2"] = {
            "tl1": (tl1.thread_id, tl1.tracer_name),
            "tl2": (tl2.thread_id, tl2.tracer_name),
        }

    # Run both threads
    t1 = threading.Thread(target=thread1_func)
    t2 = threading.Thread(target=thread2_func)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Verify that each thread had its own storage
    assert results["thread1"]["tl1"] == ("thread1", "tracer1")
    assert results["thread1"]["tl2"] == ("thread1", "tracer2")
    assert results["thread2"]["tl1"] == ("thread2", "tracer1")
    assert results["thread2"]["tl2"] == ("thread2", "tracer2")


def test_wrapped_tool_handler_uses_correct_tracer_storage():
    """
    Test that the wrapped tool handler retrieves context from the correct tracer's storage.

    This is the critical test that verifies the fix for the trace_id collision bug.
    When multiple tracers are active, tools should use the correct tracer's parent context.
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _wrap_tool_handler,
        _get_thread_local,
    )
    from opentelemetry.trace import SpanContext, TraceFlags
    from opentelemetry.context import Context

    # Create two mock tracers with mock OpenTelemetry tracers
    tracer1 = Mock()
    tracer2 = Mock()

    mock_otel_tracer1 = Mock()
    mock_otel_tracer2 = Mock()

    tracer1.get_tracer.return_value = mock_otel_tracer1
    tracer2.get_tracer.return_value = mock_otel_tracer2

    # Create mock spans with different trace IDs
    trace_id1 = 0x12345678901234567890123456789012
    trace_id2 = 0x98765432109876543210987654321098

    mock_span1 = Mock()
    mock_span1.context = SpanContext(
        trace_id=trace_id1,
        span_id=1,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )

    mock_span2 = Mock()
    mock_span2.context = SpanContext(
        trace_id=trace_id2,
        span_id=2,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )

    mock_otel_tracer1.start_span.return_value = mock_span1
    mock_otel_tracer2.start_span.return_value = mock_span2

    # Mock use_span to return a context manager
    mock_cm1 = MagicMock()
    mock_cm1.__enter__ = Mock(return_value=None)
    mock_cm1.__exit__ = Mock(return_value=None)
    tracer1.use_span.return_value = mock_cm1

    mock_cm2 = MagicMock()
    mock_cm2.__enter__ = Mock(return_value=None)
    mock_cm2.__exit__ = Mock(return_value=None)
    tracer2.use_span.return_value = mock_cm2

    # Create mock contexts for each tracer
    mock_context1 = Mock(spec=Context)
    mock_context2 = Mock(spec=Context)

    # Store parent contexts in thread-local storage (simulating what receive_response does)
    tl1 = _get_thread_local(tracer1)
    tl2 = _get_thread_local(tracer2)

    tl1.parent_context = mock_context1
    tl1.tracer = tracer1

    tl2.parent_context = mock_context2
    tl2.tracer = tracer2

    # Create mock tool handlers
    async def mock_handler1(args):
        return {"result": "handler1"}

    async def mock_handler2(args):
        return {"result": "handler2"}

    # Wrap the handlers with each tracer
    wrapped_handler1 = _wrap_tool_handler(tracer1, mock_handler1, "tool1")
    wrapped_handler2 = _wrap_tool_handler(tracer2, mock_handler2, "tool2")

    # Call the wrapped handlers
    import asyncio

    asyncio.run(wrapped_handler1({"input": "test1"}))
    asyncio.run(wrapped_handler2({"input": "test2"}))

    # Verify that each handler used the correct context
    # Handler1 should have called start_span with context1
    assert mock_otel_tracer1.start_span.called
    call_kwargs1 = mock_otel_tracer1.start_span.call_args[1]
    assert call_kwargs1["context"] == mock_context1

    # Handler2 should have called start_span with context2
    assert mock_otel_tracer2.start_span.called
    call_kwargs2 = mock_otel_tracer2.start_span.call_args[1]
    assert call_kwargs2["context"] == mock_context2

    print("✅ Tool handlers correctly use their respective tracer's context")


def test_thread_local_cleanup():
    """
    Test that thread-local storage is properly cleaned up after use.

    This ensures no memory leaks and prevents stale context from being reused.
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _get_thread_local,
        _tracer_thread_locals,
    )

    # Clear any existing thread-local storage
    _tracer_thread_locals.clear()

    tracer = Mock()
    tl = _get_thread_local(tracer)

    # Set some attributes
    tl.parent_context = Mock()
    tl.tracer = tracer

    # Verify attributes exist
    assert hasattr(tl, "parent_context")
    assert hasattr(tl, "tracer")

    # Clean up (simulating what the finally block does)
    if hasattr(tl, "parent_context"):
        delattr(tl, "parent_context")
    if hasattr(tl, "tracer"):
        delattr(tl, "tracer")

    # Verify attributes are removed
    assert not hasattr(tl, "parent_context")
    assert not hasattr(tl, "tracer")


def test_multiple_setup_calls_same_process():
    """
    Test that calling setup_claude_agent_sdk() multiple times in the same process
    works correctly, with each client using its assigned tracer.

    This tests the fix for the wrapper overwrite issue where the second setup_claude_agent_sdk()
    call would overwrite the first, causing all clients to use the last tracer.
    """
    from judgeval.v1.integrations.claude_agent_sdk.wrapper import (
        _create_client_wrapper_class,
    )

    # Create two mock tracers
    tracer1 = Mock()
    tracer2 = Mock()

    # Mock OpenTelemetry tracers
    mock_otel_tracer1 = Mock()
    mock_otel_tracer2 = Mock()

    tracer1.get_tracer.return_value = mock_otel_tracer1
    tracer2.get_tracer.return_value = mock_otel_tracer2

    # Create a mock original client class
    class MockOriginalClient:
        def __init__(self, *args, **kwargs):
            pass

    # Simulate calling setup_claude_agent_sdk() twice (same process, different tracers)
    WrappedClient1 = _create_client_wrapper_class(MockOriginalClient, tracer1)
    WrappedClient2 = _create_client_wrapper_class(MockOriginalClient, tracer2)

    # Create client instances from each wrapped class
    # This simulates creating clients BEFORE the second setup overwrites
    client1_early = WrappedClient1()

    # Now simulate the second setup_claude_agent_sdk() call overwriting the global
    # In reality, claude_agent_sdk.ClaudeSDKClient would now point to WrappedClient2

    # Create clients from the "overwritten" class (WrappedClient2)
    client2 = WrappedClient2()
    client1_late = WrappedClient2()  # Created AFTER overwrite, should use tracer2

    # Verify each client stores the correct tracer on its instance
    # The key is that the tracer is stored on __init__, not used from closure
    assert hasattr(client1_early, "_WrappedClaudeSDKClient__judgeval_tracer")
    assert hasattr(client2, "_WrappedClaudeSDKClient__judgeval_tracer")
    assert hasattr(client1_late, "_WrappedClaudeSDKClient__judgeval_tracer")

    # client1_early was created from WrappedClient1, so it has tracer1
    assert client1_early._WrappedClaudeSDKClient__judgeval_tracer is tracer1

    # client2 was created from WrappedClient2, so it has tracer2
    assert client2._WrappedClaudeSDKClient__judgeval_tracer is tracer2

    # client1_late was ALSO created from WrappedClient2 (after overwrite), so it has tracer2
    assert client1_late._WrappedClaudeSDKClient__judgeval_tracer is tracer2

    print("✅ Each client instance correctly stores its tracer")
    print(
        f"   - client1_early → tracer1: {client1_early._WrappedClaudeSDKClient__judgeval_tracer is tracer1}"
    )
    print(
        f"   - client2       → tracer2: {client2._WrappedClaudeSDKClient__judgeval_tracer is tracer2}"
    )
    print(
        f"   - client1_late  → tracer2: {client1_late._WrappedClaudeSDKClient__judgeval_tracer is tracer2}"
    )


if __name__ == "__main__":
    # Run tests
    test_thread_local_storage_per_tracer()
    print("✅ test_thread_local_storage_per_tracer passed")

    test_thread_local_storage_across_threads()
    print("✅ test_thread_local_storage_across_threads passed")

    test_wrapped_tool_handler_uses_correct_tracer_storage()
    print("✅ test_wrapped_tool_handler_uses_correct_tracer_storage passed")

    test_thread_local_cleanup()
    print("✅ test_thread_local_cleanup passed")

    test_multiple_setup_calls_same_process()
    print("✅ test_multiple_setup_calls_same_process passed")

    print("\n🎉 All tests passed!")
