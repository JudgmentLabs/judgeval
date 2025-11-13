from __future__ import annotations

from typing import Any
from opentelemetry.trace import Span
from judgeval.judgment_attribute_keys import AttributeKeys
from judgeval.utils.serialize import safe_serialize


def openai_tokens_converter(
    prompt_tokens: int,
    completion_tokens: int,
    cache_read: int,
    cache_creation: int,
    total_tokens: int,
) -> tuple[int, int, int, int]:
    """
    Returns:
        tuple[int, int, int, int]:
            - judgment.usage.non_cached_input
            - judgment.usage.output_tokens
            - judgment.usage.cached_input_tokens
            - judgment.usage.cache_creation_tokens
    """
    manual_tokens = prompt_tokens + completion_tokens + cache_read + cache_creation

    if manual_tokens > total_tokens:
        return prompt_tokens - cache_read, completion_tokens, cache_read, cache_creation
    else:
        return prompt_tokens, completion_tokens, cache_read, cache_creation


def set_cost_attribute(span: Span, usage_data: Any) -> None:
    """
    This is for OpenRouter case where the cost is provided in the usage data when they specify:
    extra_body={"usage": {"include": True}},
    """
    if hasattr(usage_data, "cost") and usage_data.cost:
        span.set_attribute(
            AttributeKeys.JUDGMENT_USAGE_TOTAL_COST_USD,
            safe_serialize(usage_data.cost),
        )
