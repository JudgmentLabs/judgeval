from __future__ import annotations

from enum import Enum


class AttributeKeys(str, Enum):
    JUDGMENT_SPAN_KIND = "judgment.span_kind"
    JUDGMENT_INPUT = "judgment.input"
    JUDGMENT_OUTPUT = "judgment.output"
    JUDGMENT_OFFLINE_MODE = "judgment.offline_mode"
    JUDGMENT_UPDATE_ID = "judgment.update_id"
    JUDGMENT_CUSTOMER_ID = "judgment.customer_id"
    JUDGMENT_AGENT_ID = "judgment.agent_id"
    JUDGMENT_PARENT_AGENT_ID = "judgment.parent_agent_id"
    JUDGMENT_AGENT_CLASS_NAME = "judgment.agent_class_name"
    JUDGMENT_AGENT_INSTANCE_NAME = "judgment.agent_instance_name"
    JUDGMENT_IS_AGENT_ENTRY_POINT = "judgment.is_agent_entry_point"
    JUDGMENT_CUMULATIVE_LLM_COST = "judgment.cumulative_llm_cost"
    JUDGMENT_STATE_BEFORE = "judgment.state_before"
    JUDGMENT_STATE_AFTER = "judgment.state_after"
    JUDGMENT_PENDING_TRACE_EVAL = "judgment.pending_trace_eval"
    JUDGMENT_USAGE_METADATA = "judgment.usage.metadata"

    JUDGMENT_LLM_PROVIDER = "judgment.llm.provider"
    JUDGMENT_LLM_MODEL_NAME = "judgment.llm.model"
    JUDGMENT_USAGE_NON_CACHED_INPUT_TOKENS = "judgment.usage.non_cached_input_tokens"
    JUDGMENT_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        "judgment.usage.cache_creation_input_tokens"
    )
    JUDGMENT_USAGE_CACHE_READ_INPUT_TOKENS = "judgment.usage.cache_read_input_tokens"
    JUDGMENT_USAGE_OUTPUT_TOKENS = "judgment.usage.output_tokens"
    JUDGMENT_USAGE_TOTAL_COST_USD = "judgment.usage.total_cost_usd"

    JUDGMENT_LLM_PROMPT = "judgment.llm.prompt"
    JUDGMENT_LLM_COMPLETION = "judgment.llm.completion"

    JUDGMENT_SESSION_ID = "judgment.session_id"
    JUDGMENT_PROJECT_ID_OVERRIDE = "judgment.project_id_override"


class InternalAttributeKeys(str, Enum):
    """
    Internal attribute keys used for temporary state management in span processors.
    These are NOT exported and are used only for internal span lifecycle management.
    """

    DISABLE_PARTIAL_EMIT = "disable_partial_emit"
    CANCELLED = "cancelled"
    IS_CUSTOMER_CONTEXT_OWNER = "is_customer_context_owner"


class ResourceKeys(str, Enum):
    SERVICE_NAME = "service.name"
    TELEMETRY_SDK_LANGUAGE = "telemetry.sdk.language"
    TELEMETRY_SDK_NAME = "telemetry.sdk.name"
    TELEMETRY_SDK_VERSION = "telemetry.sdk.version"
    JUDGMENT_PROJECT_ID = "judgment.project_id"
