import os
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
import pytest
import litellm

from judgeval.common.tracer import Tracer
from judgeval.integrations.litellm_integration import JudgevalLitellmCallbackHandler


# Global handler to ensure only one instance
_GLOBAL_LITELLM_HANDLER = None


def get_or_create_litellm_handler(tracer):
    """Get or create a single LiteLLM callback handler"""
    global _GLOBAL_LITELLM_HANDLER

    # Clear any existing callbacks first
    if hasattr(litellm, 'callbacks'):
        existing_handlers = [cb for cb in litellm.callbacks if isinstance(
            cb, JudgevalLitellmCallbackHandler)]
        if existing_handlers:
            print(
                f"Found {len(existing_handlers)} existing LiteLLM handlers, clearing them")
            litellm.callbacks = [cb for cb in litellm.callbacks if not isinstance(
                cb, JudgevalLitellmCallbackHandler)]

    # Create new handler if needed
    if _GLOBAL_LITELLM_HANDLER is None or _GLOBAL_LITELLM_HANDLER.tracer != tracer:
        print("Creating new LiteLLM callback handler")
        _GLOBAL_LITELLM_HANDLER = JudgevalLitellmCallbackHandler(tracer)
    else:
        print("Reusing existing LiteLLM callback handler")

    # Ensure it's registered
    if not hasattr(litellm, 'callbacks'):
        litellm.callbacks = []

    if _GLOBAL_LITELLM_HANDLER not in litellm.callbacks:
        litellm.callbacks.append(_GLOBAL_LITELLM_HANDLER)
        print(
            f"Registered handler. Total LiteLLM callbacks: {len(litellm.callbacks)}")

    return _GLOBAL_LITELLM_HANDLER


@pytest.fixture(scope="session")
def setup_litellm_handler():
    """Setup and cleanup LiteLLM handlers for the test session"""
    # Clear any existing handlers at start
    if hasattr(litellm, 'callbacks'):
        litellm.callbacks = [cb for cb in litellm.callbacks if not isinstance(
            cb, JudgevalLitellmCallbackHandler)]

    yield

    # Cleanup at end
    global _GLOBAL_LITELLM_HANDLER
    if hasattr(litellm, 'callbacks'):
        litellm.callbacks = [cb for cb in litellm.callbacks if not isinstance(
            cb, JudgevalLitellmCallbackHandler)]
    _GLOBAL_LITELLM_HANDLER = None
    print("Cleaned up LiteLLM handlers")


class MockLLMAgent:
    """Mock agent that uses LiteLLM for testing"""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    @property
    def name(self):
        return "test-agent"

    def generate_response(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate a response using LiteLLM"""
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        return response.choices[0].message.content


def create_mock_response(content: str = "Test response from LLM"):
    """Create a mock LiteLLM response object"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    mock_response.model = "gpt-4o"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    return mock_response


@pytest.fixture
def tracer():
    """Create a tracer instance for testing"""
    api_key = os.getenv("JUDGMENT_API_KEY")
    org_id = os.getenv("JUDGMENT_ORG_ID")

    if not api_key:
        pytest.skip("JUDGMENT_API_KEY environment variable not set")
    if not org_id:
        pytest.skip("JUDGMENT_ORG_ID environment variable not set")

    return Tracer(
        api_key=api_key,
        organization_id=org_id,
        project_name="test-litellm-integration"
    )


@pytest.fixture
def mock_agent(tracer):
    """Create a mock agent for testing"""
    return MockLLMAgent(tracer)


@pytest.fixture
def litellm_handler(tracer, setup_litellm_handler):
    """Create LiteLLM callback handler (ensure only one exists)"""
    handler = get_or_create_litellm_handler(tracer)
    return handler


def test_litellm_callback_handler_creation(tracer, setup_litellm_handler):
    """Test that the callback handler can be created and registered"""
    handler = get_or_create_litellm_handler(tracer)

    assert handler.tracer == tracer
    assert handler._current_span_id is None
    assert handler._current_trace_client is None

    # Verify it's registered with LiteLLM
    assert handler in litellm.callbacks

    # Verify only one handler of our type exists
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(
        our_handlers) == 1, f"Expected 1 handler, found {len(our_handlers)}"


def test_span_creation_and_updates(tracer, litellm_handler):
    """Test that spans are created and updated correctly"""

    # Verify we're using the same handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(
        our_handlers) == 1, f"Expected 1 handler, found {len(our_handlers)}"

    # Create a trace context
    with tracer.trace("test-trace") as trace_client:

        # Mock the LiteLLM completion call
        mock_response = create_mock_response("Hello from LiteLLM!")
        start_time = time.time()
        end_time = start_time + 2.5

        # Simulate the callback lifecycle
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={}
        )

        # Verify span was created
        assert litellm_handler._current_span_id is not None
        assert litellm_handler._current_trace_client is not None
        span_id = litellm_handler._current_span_id

        # Check span exists in trace
        assert span_id in trace_client.span_id_to_span
        span = trace_client.span_id_to_span[span_id]

        # Verify initial span properties
        assert span.function == "LiteLLM-gpt-4o"
        assert span.span_type == "llm"
        assert span.inputs["model"] == "gpt-4o"
        assert span.inputs["messages"] == [
            {"role": "user", "content": "Hello"}]
        assert span.duration is None  # Should be None initially
        assert span.output is None    # Should be None initially

        # Simulate successful completion
        litellm_handler.log_success_event(
            kwargs={},
            response_obj=mock_response,
            start_time=start_time,
            end_time=end_time
        )

        # Verify span was updated
        updated_span = trace_client.span_id_to_span[span_id]
        assert updated_span.duration == 2.5
        assert updated_span.output == "Hello from LiteLLM!"

        # Verify handler cleaned up
        assert litellm_handler._current_span_id is None
        assert litellm_handler._current_trace_client is None


def test_error_handling(tracer, litellm_handler):
    """Test that errors are handled correctly"""

    # Verify single handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(our_handlers) == 1

    with tracer.trace("test-error-trace") as trace_client:

        # Simulate the callback lifecycle with error
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={}
        )

        span_id = litellm_handler._current_span_id
        start_time = time.time()
        end_time = start_time + 1.0

        # Simulate failure
        litellm_handler.log_failure_event(
            kwargs={"exception": Exception("API Error")},
            response_obj=None,
            start_time=start_time,
            end_time=end_time
        )

        # Verify error was recorded
        updated_span = trace_client.span_id_to_span[span_id]
        assert updated_span.duration == 1.0
        assert "API Error" in str(updated_span.output)


def test_real_litellm_call_with_agent(tracer, mock_agent, setup_litellm_handler):
    """Test the mock agent with real LiteLLM integration"""

    # Get or create handler (ensures only one)
    handler = get_or_create_litellm_handler(tracer)

    # Verify single handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(
        our_handlers) == 1, f"Expected 1 handler, found {len(our_handlers)}"

    @tracer.observe(name="agent-generate")
    def agent_generate_with_tracing(prompt: str) -> str:
        return mock_agent.generate_response(prompt)

    # Execute the agent method with a real LiteLLM call
    result = agent_generate_with_tracing("What is 2+2?")

    # Verify we got a real response
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"LiteLLM Response: {result}")


def test_save_coordination(tracer, litellm_handler):
    """Test that save coordination works properly"""

    # Verify single handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(our_handlers) == 1

    # Track save calls
    save_calls = []
    deferred_save_executed = False

    def mock_save(overwrite=False):
        save_calls.append(("save", overwrite))
        # Check if we should defer
        with tracer._save_lock:
            if not tracer._safe_to_save:
                # Store the actual trace_client object, not a string
                tracer._deferred_save_pending = True
                tracer._deferred_save_args = (
                    trace_client, overwrite)  # Use actual trace_client
                return "test-trace-id", {}
        return "test-trace-id", {}

    def mock_perform_actual_save(overwrite=False):
        nonlocal deferred_save_executed
        save_calls.append(("actual_save", overwrite))
        deferred_save_executed = True
        return "test-trace-id", {}

    with tracer.trace("test-coordination") as trace_client:

        # Patch the save methods
        original_save = trace_client.save
        original_actual_save = trace_client._perform_actual_save
        trace_client.save = mock_save
        trace_client._perform_actual_save = mock_perform_actual_save

        try:
            # Start LiteLLM operation
            litellm_handler.log_pre_api_call(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                kwargs={}
            )

            # Verify _safe_to_save is False
            assert not tracer._safe_to_save

            # Simulate trace save attempt (this should be deferred)
            trace_client.save()

            # Verify save was called
            assert len(save_calls) >= 1
            assert save_calls[0] == ("save", False)

            # Complete LiteLLM operation
            mock_response = create_mock_response()
            litellm_handler.log_success_event(
                kwargs={},
                response_obj=mock_response,
                start_time=time.time(),
                end_time=time.time() + 1
            )

            # Verify _safe_to_save is True
            assert tracer._safe_to_save

        finally:
            # Restore original methods
            trace_client.save = original_save
            trace_client._perform_actual_save = original_actual_save


def test_multiple_llm_calls_same_trace(tracer, litellm_handler):
    """Test multiple LiteLLM calls within the same trace"""

    # Verify single handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(our_handlers) == 1

    with tracer.trace("test-multiple-calls") as trace_client:

        # First LLM call
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "First call"}],
            kwargs={}
        )
        first_span_id = litellm_handler._current_span_id

        litellm_handler.log_success_event(
            kwargs={},
            response_obj=create_mock_response("First response"),
            start_time=time.time(),
            end_time=time.time() + 1
        )

        # Second LLM call
        litellm_handler.log_pre_api_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Second call"}],
            kwargs={}
        )
        second_span_id = litellm_handler._current_span_id

        litellm_handler.log_success_event(
            kwargs={},
            response_obj=create_mock_response("Second response"),
            start_time=time.time(),
            end_time=time.time() + 2
        )

        # Verify both spans exist and are different
        assert first_span_id != second_span_id
        assert first_span_id in trace_client.span_id_to_span
        assert second_span_id in trace_client.span_id_to_span

        # Verify span details
        first_span = trace_client.span_id_to_span[first_span_id]
        second_span = trace_client.span_id_to_span[second_span_id]

        assert first_span.function == "LiteLLM-gpt-4o"
        assert second_span.function == "LiteLLM-gpt-3.5-turbo"
        assert first_span.output == "First response"
        assert second_span.output == "Second response"


def test_real_llm_call_with_trace_saving(tracer, setup_litellm_handler):
    """Test with real LiteLLM call and trace saving"""

    # Get or create handler (ensures only one)
    handler = get_or_create_litellm_handler(tracer)

    # Verify single handler
    our_handlers = [cb for cb in litellm.callbacks if isinstance(
        cb, JudgevalLitellmCallbackHandler)]
    assert len(
        our_handlers) == 1, f"Expected 1 handler, found {len(our_handlers)}"

    # Track spans created
    spans_created = []

    @tracer.observe(name="real-llm-test")
    def make_real_llm_call():
        # Get current trace to monitor spans
        current_trace = tracer.get_current_trace()
        initial_span_count = len(
            current_trace.span_id_to_span) if current_trace else 0

        response = litellm.completion(
            model="gpt-4o-mini",  # Use cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello from LiteLLM integration test'"}],
            max_tokens=20
        )

        # Give the callback a moment to complete
        # LiteLLM callbacks are executed after the response is returned
        time.sleep(0.1)  # 100ms should be enough for callback to complete

        # Check spans after LiteLLM call and callback completion
        if current_trace:
            final_span_count = len(current_trace.span_id_to_span)
            llm_spans = [span for span in current_trace.span_id_to_span.values()
                         if span.function.startswith("LiteLLM-")]
            spans_created.extend(llm_spans)
            print(f"Spans in trace after LiteLLM call: {len(llm_spans)}")
            for span in llm_spans:
                print(
                    f"  - {span.function}: duration={span.duration}, output={span.output}")

        return response.choices[0].message.content

    # Execute the test
    result = make_real_llm_call()

    # Verify result
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"Real LiteLLM Response: {result}")

    # Verify spans were created during execution
    assert len(
        spans_created) >= 1, f"Expected at least 1 LiteLLM span, got {len(spans_created)}"

    llm_span = spans_created[0]
    assert llm_span.function.startswith("LiteLLM-gpt-4o-mini")

    # Now duration and output should be set since we waited for callback
    assert llm_span.duration is not None, f"Expected duration to be set, got {llm_span.duration}"
    assert llm_span.output is not None, f"Expected output to be set, got {llm_span.output}"
    assert isinstance(llm_span.output, str)
    assert llm_span.span_type == "llm"

    print(f"✅ LiteLLM span verification passed!")
    print(f"   Duration: {llm_span.duration}s")
    print(f"   Output: {llm_span.output}")

    # Optional: Check if traces were saved (may not always work due to save coordination)
    if len(tracer.traces) >= 1:
        print(f"Traces saved: {len(tracer.traces)}")
        trace_data = tracer.traces[-1]
        llm_spans_in_saved = [span for span in trace_data.get("entries", [])
                              if span.get("function", "").startswith("LiteLLM-")]
        print(f"LiteLLM spans in saved trace: {len(llm_spans_in_saved)}")
    else:
        print("No traces saved (may be expected due to save coordination)")


if __name__ == "__main__":
    # Run a simple test
    api_key = os.getenv("JUDGMENT_API_KEY")
    org_id = os.getenv("JUDGMENT_ORG_ID")

    if not api_key or not org_id:
        print("Please set JUDGMENT_API_KEY and JUDGMENT_ORG_ID environment variables")
        exit(1)

    tracer = Tracer(
        api_key=api_key,
        organization_id=org_id,
        project_name="litellm-test"
    )

    handler = get_or_create_litellm_handler(tracer)
    print("LiteLLM integration test setup successful!")
    print(f"Handler created: {handler}")
    print(f"Tracer ready: {tracer}")
    print(f"Total LiteLLM callbacks: {len(litellm.callbacks)}")
