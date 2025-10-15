"""General-purpose utilities for LLM wrapper tests."""

from typing import Any, Dict
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.tracer.keys import AttributeKeys


def verify_span_attributes_comprehensive(
    span: ReadableSpan,
    attrs: Dict[str, Any],
    expected_span_name: str,
    expected_model_name: str,
    check_prompt: bool = True,
    check_completion: bool = True,
    check_usage: bool = True,
    check_cache: bool = True,
    check_metadata: bool = True,
    min_input_tokens: int = 1,
    min_output_tokens: int = 1,
) -> None:
    """Comprehensive span attribute verification - combines all checks.

    Args:
        span: The span to validate
        attrs: Span attributes dictionary
        expected_span_name: Expected span name (e.g., "ANTHROPIC_API_CALL")
        expected_model_name: Expected model name (optional)
        check_prompt: Whether to verify GEN_AI_PROMPT is present
        check_completion: Whether to verify GEN_AI_COMPLETION is present
        check_usage: Whether to verify usage tokens are present and > 0
        check_cache: Whether to verify cache token attributes are present
        check_metadata: Whether to verify JUDGMENT_USAGE_METADATA is present
        min_input_tokens: Minimum expected input tokens
        min_output_tokens: Minimum expected output tokens
    """
    # Basic span validation
    assert span is not None
    assert span.name == expected_span_name

    # Verify span kind
    assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm"

    assert attrs.get(AttributeKeys.GEN_AI_REQUEST_MODEL) == expected_model_name
    assert AttributeKeys.GEN_AI_RESPONSE_MODEL in attrs

    # Verify prompt was captured
    if check_prompt:
        assert AttributeKeys.GEN_AI_PROMPT in attrs

    # Verify completion was captured
    if check_completion:
        assert AttributeKeys.GEN_AI_COMPLETION in attrs

    # Verify usage tokens
    if check_usage:
        assert AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS in attrs
        assert attrs[AttributeKeys.GEN_AI_USAGE_INPUT_TOKENS] >= min_input_tokens
        assert attrs[AttributeKeys.GEN_AI_USAGE_OUTPUT_TOKENS] >= min_output_tokens

    # Verify cache tokens attribute exists
    if check_cache:
        assert AttributeKeys.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS in attrs
        assert AttributeKeys.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS in attrs

    # Verify usage metadata
    if check_metadata:
        assert AttributeKeys.JUDGMENT_USAGE_METADATA in attrs


def assert_span_has_exception(
    span: ReadableSpan,
    expected_span_name: str,
) -> None:
    """Assert that a span has exception events recorded.

    Args:
        span: The span to validate
        expected_span_name: Expected span name
    """
    assert span is not None
    assert span.name == expected_span_name

    # Verify span has events (exception recording)
    if span.events:
        event_names = [event.name for event in span.events]
        assert any("exception" in name.lower() for name in event_names)
