"""
Crash resistance tests for the v1 SDK.

These tests verify that SDK errors never crash customer code.

Run with: pytest src/tests/reliability/test_crash_resistance.py -v
"""

import statistics
import time
from unittest.mock import patch
import pytest
import httpx

from judgeval.v1.data.example import Example
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer
from judgeval.v1.tracer.tracer import Tracer
from judgeval.v1.tracer.processors.judgment_span_processor import JudgmentSpanProcessor


def _get_span_processor(tracer: Tracer) -> JudgmentSpanProcessor:
    processor = tracer._tracer_provider._active_span_processor
    for span_processor in getattr(processor, "_span_processors", []):
        if isinstance(span_processor, JudgmentSpanProcessor):
            return span_processor
    raise AssertionError("JudgmentSpanProcessor not found on tracer provider")


@pytest.mark.reliability
class TestCrashResistance:
    """Test that SDK failures never crash customer applications."""

    def test_sdk_never_crashes_customer_code(self, tracer_with_evaluation: Tracer):
        """
        Inject various SDK failures and verify customer code continues.
        """
        scorer = AnswerRelevancyScorer()
        example = Example(name="test").create(input="q", output="a")

        with patch.object(
            tracer_with_evaluation.api_client,
            "add_to_run_eval_queue_examples",
            side_effect=Exception("API failed"),
        ):

            @tracer_with_evaluation.observe(span_type="function")
            def function_with_eval():
                tracer_with_evaluation.async_evaluate(scorer, example)
                return "success"

            result = function_with_eval()
            assert result == "success"

        def serializer_raises(_value):
            raise TypeError("Cannot serialize")

        with patch.object(tracer_with_evaluation, "serializer", serializer_raises):

            @tracer_with_evaluation.observe(span_type="function")
            def function_with_bad_attribute():
                tracer_with_evaluation.set_attribute("key", object())
                return "success"

            result = function_with_bad_attribute()
            assert result == "success"

        span_processor = _get_span_processor(tracer_with_evaluation)
        exporter = span_processor._span_exporter

        with patch.object(
            exporter._delegate, "export", side_effect=httpx.NetworkError("fail")
        ):

            @tracer_with_evaluation.observe(span_type="function")
            def function_with_export_failure():
                return "success"

            result = function_with_export_failure()
            assert result == "success"

    def test_oom_during_serialization(self, tracer: Tracer):
        """
        Large payloads should be handled without crashing customer code.
        """
        huge_data = "x" * 10_000_000  # 10MB

        @tracer.observe(span_type="function")
        def function_with_huge_payload():
            tracer.set_attribute("huge", huge_data)
            return "success"

        result = function_with_huge_payload()
        assert result == "success"

    def test_latency_with_background_pressure(self, tracer: Tracer):
        """
        Background export pressure should not affect foreground latency.
        """
        for i in range(2000):
            with tracer.span(f"background-{i}"):
                tracer.set_attribute("data", "x" * 1000)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            with tracer.span("foreground"):
                pass
            latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = statistics.quantiles(latencies, n=20)[18]

        assert p95_latency < 10.0, (
            f"Foreground latency {p95_latency:.2f}ms with background pressure"
        )
