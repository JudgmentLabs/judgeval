"""
Identifiers used by Judgeval to store specific types of data in the spans.
"""

from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes


class AttributeKeys:
    # General function tracing attributes (custom namespace)
    JUDGMENT_SPAN_KIND = "judgment.span_kind"
    JUDGMENT_INPUT = "judgment.input"
    JUDGMENT_OUTPUT = "judgment.output"

    # GenAI-specific attributes (semantic conventions)
    GEN_AI_PROMPT = gen_ai_attributes.GEN_AI_PROMPT
    GEN_AI_COMPLETION = gen_ai_attributes.GEN_AI_COMPLETION
    GEN_AI_REQUEST_MODEL = gen_ai_attributes.GEN_AI_REQUEST_MODEL
    GEN_AI_RESPONSE_MODEL = gen_ai_attributes.GEN_AI_RESPONSE_MODEL
    GEN_AI_SYSTEM = gen_ai_attributes.GEN_AI_SYSTEM
    GEN_AI_USAGE_INPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS
    GEN_AI_USAGE_OUTPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS
    GEN_AI_USAGE_COMPLETION_TOKENS = gen_ai_attributes.GEN_AI_USAGE_COMPLETION_TOKENS
    GEN_AI_REQUEST_TEMPERATURE = gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE
    GEN_AI_REQUEST_MAX_TOKENS = gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS
    GEN_AI_RESPONSE_FINISH_REASONS = gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS


class ResourceKeys:
    SERVICE_NAME = ResourceAttributes.SERVICE_NAME
    JUDGMENT_PROJECT_ID = "judgment.project_id"
