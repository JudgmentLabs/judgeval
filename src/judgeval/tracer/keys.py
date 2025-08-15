"""
Identifiers used by Judgeval to store specific types of data in the spans.
"""

from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes


class AttributeKeys:
    SPAN_TYPE = SpanAttributes.TYPE

    # General function tracing attributes (custom namespace)
    JUDGMENT_INPUT = "judgment.input"
    JUDGMENT_OUTPUT = "judgment.output"

    # Agent specific attributes (custom namespace)
    JUDGMENT_AGENT_ID = "judgment.agent_id"
    JUDGMENT_PARENT_AGENT_ID = "judgment.parent_agent_id"
    JUDGMENT_AGENT_CLASS_NAME = "judgment.agent_class_name"
    JUDGMENT_AGENT_INSTANCE_NAME = "judgment.agent_instance_name"
    JUDGMENT_IS_AGENT_ENTRY_POINT = "judgment.is_agent_entry_point"
    JUDGMENT_CUMULATIVE_LLM_COST = "judgment.cumulative_llm_cost"

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

    # GenAI-specific attributes (custom namespace)
    GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    GEN_AI_USAGE_TOTAL_COST = "gen_ai.usage.total_cost"


class ResourceKeys:
    SERVICE_NAME = ResourceAttributes.SERVICE_NAME
    JUDGMENT_PROJECT_ID = "judgment.project_id"
