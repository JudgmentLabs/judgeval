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

    JUDGMENT_SESSION_ID = "judgment.session_id"

    GEN_AI_PROMPT = "gen_ai.prompt"
    GEN_AI_COMPLETION = "gen_ai.completion"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    GEN_AI_SYSTEM = "gen_ai.system"
    GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
        "gen_ai.usage.cache_creation_input_tokens"
    )
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read_input_tokens"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"


class ResourceKeys(str, Enum):
    SERVICE_NAME = "service.name"
    TELEMETRY_SDK_LANGUAGE = "telemetry.sdk.language"
    TELEMETRY_SDK_NAME = "telemetry.sdk.name"
    TELEMETRY_SDK_VERSION = "telemetry.sdk.version"
    JUDGMENT_PROJECT_ID = "judgment.project_id"
