"""Utility functions for Claude Agent SDK integration tests."""

from typing import List, Optional
from opentelemetry.sdk.trace import ReadableSpan
from judgeval.tracer.keys import AttributeKeys


def find_spans_by_kind(spans: List[ReadableSpan], kind: str) -> List[ReadableSpan]:
    """Find all spans with a specific kind attribute."""
    return [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(AttributeKeys.JUDGMENT_SPAN_KIND) == kind
    ]


def verify_agent_span(
    span: ReadableSpan,
    expected_input: Optional[str] = None,
    check_output: bool = False,
) -> None:
    """Verify that a span is a valid agent span."""
    attrs = dict(span.attributes or {})

    # Check span kind
    assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "agent", (
        f"Expected agent span, got: {attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND)}"
    )

    # Check input if provided
    if expected_input:
        input_val = attrs.get(AttributeKeys.JUDGMENT_INPUT)
        assert input_val is not None, "Agent span missing input"
        assert expected_input in str(input_val), (
            f"Expected input to contain '{expected_input}', got: {input_val}"
        )

    # Check output exists if requested
    if check_output:
        output_val = attrs.get(AttributeKeys.JUDGMENT_OUTPUT)
        assert output_val is not None, "Agent span missing output"


def verify_llm_span(
    span: ReadableSpan,
    expected_model: Optional[str] = None,
    check_provider: bool = False,
    check_usage: bool = False,
) -> None:
    """Verify that a span is a valid LLM span."""
    attrs = dict(span.attributes or {})

    # Check span kind
    assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "llm", (
        f"Expected llm span, got: {attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND)}"
    )

    # Check model if provided
    if expected_model:
        model = attrs.get(AttributeKeys.JUDGMENT_LLM_MODEL_NAME)
        assert model is not None, "LLM span missing model name"
        assert expected_model in str(model), (
            f"Expected model to contain '{expected_model}', got: {model}"
        )

    # Check provider if requested
    if check_provider:
        provider = attrs.get(AttributeKeys.JUDGMENT_LLM_PROVIDER)
        assert provider == "anthropic", (
            f"Expected provider 'anthropic', got: {provider}"
        )

    # Check usage metrics if requested
    if check_usage:
        # Should have at least one of these usage metrics
        has_input_tokens = AttributeKeys.JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS in attrs
        has_output_tokens = AttributeKeys.JUDGMENT_USAGE_OUTPUT_TOKENS in attrs
        has_metadata = AttributeKeys.JUDGMENT_USAGE_METADATA in attrs

        assert has_input_tokens or has_output_tokens or has_metadata, (
            "LLM span missing usage metrics"
        )


def verify_tool_span(
    span: ReadableSpan,
    expected_tool_name: Optional[str] = None,
    check_input: bool = False,
    check_output: bool = False,
) -> None:
    """Verify that a span is a valid tool span."""
    attrs = dict(span.attributes or {})

    # Check span kind
    assert attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND) == "tool", (
        f"Expected tool span, got: {attrs.get(AttributeKeys.JUDGMENT_SPAN_KIND)}"
    )

    # Check tool name
    span_name = span.name
    if expected_tool_name:
        assert expected_tool_name in span_name, (
            f"Expected tool name to contain '{expected_tool_name}', got: {span_name}"
        )

    # Check input exists if requested
    if check_input:
        input_val = attrs.get(AttributeKeys.JUDGMENT_INPUT)
        assert input_val is not None, "Tool span missing input"

    # Check output exists if requested
    if check_output:
        output_val = attrs.get(AttributeKeys.JUDGMENT_OUTPUT)
        assert output_val is not None, "Tool span missing output"
