"""
Failure isolation tests for the v1 SDK.

These tests verify that SDK failures do not crash or impact customer code.
The SDK should be completely transparent - if it fails, customer code
should continue working normally.

Run with: pytest src/tests/reliability/test_isolation.py -v
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch

from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.tracer_factory import TracerFactory
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.v1.data.example import Example
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer
from judgeval.exceptions import JudgmentAPIError


@pytest.mark.reliability
class TestAPIFailureIsolation:
    """Test that API failures don't crash customer code."""

    def test_api_timeout_doesnt_crash_user_code(
        self, mock_client_with_timeout: MagicMock
    ):
        """
        API timeout during async_evaluate should not crash user function.

        The function should complete normally and return its result.
        """
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client_with_timeout)
            tracer = factory.create(
                project_name="timeout-test",
                enable_monitoring=True,
                enable_evaluation=True,
                isolated=True,
            )

        @tracer.observe(span_type="function")
        def customer_function():
            # Simulate async_evaluate being called
            tracer.async_evaluate(
                scorer=AnswerRelevancyScorer(),
                example=Example(name="test").create(input="q", output="a"),
            )
            return "business_result"

        # This should NOT raise, even though API call will timeout
        result = customer_function()

        assert result == "business_result", (
            "Customer function result was affected by SDK failure"
        )

    def test_api_500_doesnt_crash_user_code(self, mock_client_with_error: MagicMock):
        """
        HTTP 500 from API should not crash user function.
        """
        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client_with_error)
            tracer = factory.create(
                project_name="error-test",
                enable_monitoring=True,
                enable_evaluation=True,
                isolated=True,
            )

        @tracer.observe(span_type="function")
        def customer_function():
            tracer.async_evaluate(
                scorer=AnswerRelevancyScorer(),
                example=Example(name="test").create(input="q", output="a"),
            )
            return "business_result"

        result = customer_function()

        assert result == "business_result", (
            "Customer function result was affected by HTTP 500 error"
        )

    def test_network_error_doesnt_crash_user_code(self, mock_client: MagicMock):
        """
        Network connection errors should not crash user function.
        """
        # Configure mock to raise connection error on async_evaluate
        mock_client.add_to_run_eval_queue_examples.side_effect = httpx.ConnectError(
            "Connection refused"
        )

        with patch(
            "judgeval.v1.utils.resolve_project_id", return_value="test_project_id"
        ):
            factory = TracerFactory(mock_client)
            tracer = factory.create(
                project_name="network-error-test",
                enable_monitoring=True,
                enable_evaluation=True,
                isolated=True,
            )

        @tracer.observe(span_type="function")
        def customer_function():
            tracer.async_evaluate(
                scorer=AnswerRelevancyScorer(),
                example=Example(name="test").create(input="q", output="a"),
            )
            return "business_result"

        result = customer_function()

        assert result == "business_result", (
            "Customer function affected by network error"
        )


@pytest.mark.reliability
class TestInitializationFailureIsolation:
    """Test that initialization failures are handled gracefully."""

    def test_project_resolution_failure_allows_degradation(
        self, mock_client_project_failure: MagicMock
    ):
        """
        Failed project resolution should allow graceful degradation.

        Tracing should be disabled but user code should still work.
        """
        with patch("judgeval.v1.utils.resolve_project_id", return_value=None):
            factory = TracerFactory(mock_client_project_failure)
            tracer = factory.create(
                project_name="nonexistent-project",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

        @tracer.observe(span_type="function")
        def customer_function():
            return "business_result"

        # Should work even though project resolution failed
        result = customer_function()

        assert result == "business_result", (
            "Customer function affected by project resolution failure"
        )

    def test_invalid_credentials_handled_gracefully(self):
        """
        Invalid credentials should not crash, just disable functionality.
        """
        invalid_client = MagicMock(spec=JudgmentSyncClient)
        invalid_client.api_key = ""  # Invalid
        invalid_client.organization_id = ""  # Invalid
        invalid_client.base_url = "http://test.com/"
        invalid_client.projects_resolve.side_effect = JudgmentAPIError(
            401, "Unauthorized", None
        )

        with patch("judgeval.v1.utils.resolve_project_id", return_value=None):
            factory = TracerFactory(invalid_client)
            tracer = factory.create(
                project_name="test",
                enable_monitoring=True,
                enable_evaluation=False,
                isolated=True,
            )

        @tracer.observe(span_type="function")
        def customer_function():
            return "business_result"

        result = customer_function()

        assert result == "business_result"


@pytest.mark.reliability
class TestSerializationFailureIsolation:
    """Test that serialization failures don't crash customer code."""

    def test_set_attribute_with_unserializable_data(self, tracer: Tracer):
        """
        set_attribute with unserializable data should not crash.
        """

        class UnserializableObject:
            def __repr__(self):
                raise Exception("Cannot repr this object")

        with tracer.span("test-span"):
            # This should not raise
            tracer.set_attribute("bad_key", UnserializableObject())

        # If we got here, the test passed

    def test_observe_with_unserializable_return(self, tracer: Tracer):
        """
        @observe on function returning unserializable data should not crash.
        """

        class UnserializableResult:
            def __init__(self):
                self.value = "test"

        @tracer.observe(span_type="function")
        def function_with_bad_return():
            return UnserializableResult()

        # Should not raise, should handle serialization gracefully
        result = function_with_bad_return()

        assert result.value == "test"

    def test_observe_with_unserializable_input(self, tracer: Tracer):
        """
        @observe on function with unserializable input should not crash.
        """

        class UnserializableInput:
            pass

        @tracer.observe(span_type="function")
        def function_with_bad_input(data):
            return "processed"

        result = function_with_bad_input(UnserializableInput())

        assert result == "processed"


@pytest.mark.reliability
class TestExceptionIsolation:
    """Test that SDK exceptions don't mask user exceptions."""

    def test_user_exception_propagates_correctly(self, tracer: Tracer):
        """
        User exceptions should propagate unchanged through @observe.
        """

        class CustomUserException(Exception):
            pass

        @tracer.observe(span_type="function")
        def function_that_raises():
            raise CustomUserException("User error")

        with pytest.raises(CustomUserException) as exc_info:
            function_that_raises()

        assert str(exc_info.value) == "User error"

    def test_user_exception_type_preserved(self, tracer: Tracer):
        """
        The exact type of user exception should be preserved.
        """

        @tracer.observe(span_type="function")
        def raise_value_error():
            raise ValueError("Bad value")

        @tracer.observe(span_type="function")
        def raise_type_error():
            raise TypeError("Bad type")

        @tracer.observe(span_type="function")
        def raise_runtime_error():
            raise RuntimeError("Runtime issue")

        with pytest.raises(ValueError):
            raise_value_error()

        with pytest.raises(TypeError):
            raise_type_error()

        with pytest.raises(RuntimeError):
            raise_runtime_error()

    def test_nested_user_exceptions_propagate(self, tracer: Tracer):
        """
        User exceptions in nested traced functions should propagate correctly.
        """

        @tracer.observe(span_type="function")
        def inner_function():
            raise ValueError("Inner error")

        @tracer.observe(span_type="function")
        def outer_function():
            return inner_function()

        with pytest.raises(ValueError) as exc_info:
            outer_function()

        assert "Inner error" in str(exc_info.value)


@pytest.mark.reliability
class TestAsyncEvaluateIsolation:
    """Test that async_evaluate failures are isolated."""

    def test_async_evaluate_exception_isolated(
        self, tracer_with_evaluation: Tracer, mock_client: MagicMock
    ):
        """
        Exception in async_evaluate should not affect user code.
        """
        # Make async_evaluate fail
        mock_client.add_to_run_eval_queue_examples.side_effect = Exception(
            "Eval failed"
        )

        @tracer_with_evaluation.observe(span_type="function")
        def customer_function():
            tracer_with_evaluation.async_evaluate(
                scorer=AnswerRelevancyScorer(),
                example=Example(name="test").create(input="q", output="a"),
            )
            return "business_result"

        # Should complete normally
        result = customer_function()
        assert result == "business_result"

    def test_multiple_async_evaluate_failures_isolated(
        self, tracer_with_evaluation: Tracer, mock_client: MagicMock
    ):
        """
        Multiple async_evaluate failures should all be isolated.
        """
        mock_client.add_to_run_eval_queue_examples.side_effect = Exception(
            "Eval failed"
        )

        @tracer_with_evaluation.observe(span_type="function")
        def customer_function():
            for i in range(10):
                tracer_with_evaluation.async_evaluate(
                    scorer=AnswerRelevancyScorer(),
                    example=Example(name=f"test-{i}").create(
                        input=f"q{i}", output=f"a{i}"
                    ),
                )
            return "all_done"

        result = customer_function()
        assert result == "all_done"
