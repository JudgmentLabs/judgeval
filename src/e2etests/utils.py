import pytest
from typing import Dict, Any, List, Optional


def validate_trace_token_counts(trace_client) -> Dict[str, int]:
    """
    Validates token counts from trace spans and performs assertions.
    
    Args:
        trace_client: The trace client instance containing trace spans
        
    Returns:
        Dict with calculated token counts (prompt_tokens, completion_tokens, total_tokens)
        
    Raises:
        AssertionError: If token count validations fail
    """
    if not trace_client:
        pytest.fail("Failed to get trace client for token count validation")
        
    # Get spans from the trace client
    trace_spans = trace_client.trace_spans

    # Manually calculate token counts from trace spans
    manual_prompt_tokens = 0
    manual_completion_tokens = 0 
    manual_total_tokens = 0
    
    # Known LLM API call function names
    llm_span_names = {"OPENAI_API_CALL", "ANTHROPIC_API_CALL", "TOGETHER_API_CALL", "GOOGLE_API_CALL"}

    for span in trace_spans:
        if span.span_type == "llm" and span.function in llm_span_names and isinstance(span.output, dict):
            output = span.output
            usage = output.get("usage", {})
            if usage and "info" not in usage:  # Check if it's actual usage data
                # Correctly handle different key names from different providers

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                entry_total = usage.get("total_tokens", 0)

                # Accumulate separately
                manual_prompt_tokens += prompt_tokens
                manual_completion_tokens += completion_tokens
                # Accumulate the reported total_tokens from the usage dict
                manual_total_tokens += entry_total

    # Check if LLM spans were found before asserting counts
    # llm_spans_found = any(
    #     s.span_type == "llm" and isinstance(s.output, dict) and "usage" in s.output 
    #     for s in trace_spans
    # )
    
    assert manual_prompt_tokens > 0, "Prompt tokens should be counted"
    assert manual_completion_tokens > 0, "Completion tokens should be counted"
    assert manual_total_tokens > 0, "Total tokens should be counted"
    assert manual_total_tokens == (manual_prompt_tokens + manual_completion_tokens), \
        "Total tokens should equal prompt + completion"
    
    return {
        "prompt_tokens": manual_prompt_tokens,
        "completion_tokens": manual_completion_tokens,
        "total_tokens": manual_total_tokens
    }


def validate_trace_tokens(trace, fail_on_missing=True):
    """
    Helper function to validate token counts in a trace
    
    Args:
        trace: The trace client to validate
        fail_on_missing: Whether to fail the test if no trace is available
        
    Returns:
        The token counts if validation succeeded
    """
    if not trace:
        print("Warning: Could not get current trace to perform assertions.")
        if fail_on_missing:
            pytest.fail("Failed to get current trace within decorated function.")
        return None
        
    print("\nAttempting assertions on current trace state (before decorator save)...")
    
    # Use the utility function for token count validation
    token_counts = validate_trace_token_counts(trace)
    
    print(f"Calculated token counts: P={token_counts['prompt_tokens']}, C={token_counts['completion_tokens']}, T={token_counts['total_tokens']}")
        
    return token_counts 