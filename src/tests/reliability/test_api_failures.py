"""
API failure resilience tests for the v1 SDK.

These tests verify that SDK API interactions do not block customer code
and failures are handled gracefully.

Run with: pytest src/tests/reliability/test_api_failures.py -v
"""

import time
from unittest.mock import patch

import pytest
import httpx

from judgeval.exceptions import JudgmentAPIError
from judgeval.v1.data.example import Example
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer
from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.processors.judgment_span_processor import JudgmentSpanProcessor
from opentelemetry.sdk.trace.export import SpanExportResult


def _get_span_processor(tracer: Tracer) -> JudgmentSpanProcessor:
    processor = tracer._tracer_provider._active_span_processor
    for span_processor in getattr(processor, "_span_processors", []):
        if isinstance(span_processor, JudgmentSpanProcessor):
            return span_processor
    raise AssertionError("JudgmentSpanProcessor not found on tracer provider")


@pytest.mark.reliability
class TestAPIFailureResilience:
    """Test SDK behavior when backend API is slow or failing."""

    def test_evaluation_enqueue_with_slow_api(self, tracer_with_evaluation: Tracer):
        """
        Evaluation enqueue should not block customer code when API is slow.
        """
        scorer = AnswerRelevancyScorer()
        example = Example(name="test").create(input="q", output="a")

        def slow_api_call(*args, **kwargs):
            time.sleep(0.2)
            return {"success": True}

        with patch.object(
            tracer_with_evaluation.api_client,
            "add_to_run_eval_queue_examples",
            side_effect=slow_api_call,
        ):
            with tracer_with_evaluation.span("test-span"):
                start = time.perf_counter()
                tracer_with_evaluation.async_evaluate(scorer, example)
                duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 100, (
            f"Customer code blocked for {duration_ms:.1f}ms "
            "waiting for slow API (max: 100ms)"
        )

    def test_evaluation_enqueue_with_api_down(self, tracer_with_evaluation: Tracer):
        """
        Evaluation failures should be isolated and not raise to customer code.
        """
        scorer = AnswerRelevancyScorer()
        example = Example(name="test").create(input="q", output="a")

        with patch.object(
            tracer_with_evaluation.api_client,
            "add_to_run_eval_queue_examples",
            side_effect=Exception("API is down"),
        ):
            with tracer_with_evaluation.span("test-span"):
                tracer_with_evaluation.async_evaluate(scorer, example)

    def test_tag_with_api_timeout(self, tracer: Tracer):
        """
        Tagging should not block when API times out.
        """
        with patch.object(
            tracer.api_client, "_request", side_effect=httpx.TimeoutException("timeout")
        ):
            with tracer.span("test-span"):
                start = time.perf_counter()
                tracer.tag(["test-tag"])
                duration_ms = (time.perf_counter() - start) * 1000

        assert duration_ms < 50, (
            f"tag() blocked for {duration_ms:.1f}ms on timeout (max: 50ms)"
        )

    def test_trace_export_with_network_errors(self, tracer: Tracer):
        """
        Export should retry on transient network errors.
        """
        span_processor = _get_span_processor(tracer)
        exporter = span_processor._span_exporter

        call_count = {"count": 0}

        def flaky_export(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise httpx.NetworkError("Network failure")
            return SpanExportResult.SUCCESS

        with patch.object(exporter._delegate, "export", side_effect=flaky_export):
            for i in range(10):
                with tracer.span(f"test-{i}"):
                    pass

            tracer.force_flush(timeout_millis=10000)

        assert call_count["count"] >= 3, "Exporter did not retry on failures"

    def test_api_rate_limiting(self, tracer: Tracer):
        """
        SDK should back off when API returns 429 rate limit.
        """
        call_count = {"count": 0}

        def rate_limited_request(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise JudgmentAPIError(429, "Rate limit exceeded", None)
            return {"success": True}

        with patch.object(
            tracer.api_client, "_request", side_effect=rate_limited_request
        ):
            with tracer.span("test-span"):
                tracer.tag(["test"])

            time.sleep(0.5)

        assert call_count["count"] >= 3, "Rate limit retries were not attempted"
