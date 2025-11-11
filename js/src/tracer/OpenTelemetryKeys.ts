export abstract class OpenTelemetryKeys {
  public static readonly AttributeKeys = {
    JUDGMENT_SPAN_KIND: "judgment.span_kind",
    JUDGMENT_INPUT: "judgment.input",
    JUDGMENT_OUTPUT: "judgment.output",
    JUDGMENT_OFFLINE_MODE: "judgment.offline_mode",
    JUDGMENT_UPDATE_ID: "judgment.update_id",

    JUDGMENT_CUSTOMER_ID: "judgment.customer_id",

    JUDGMENT_AGENT_ID: "judgment.agent_id",

    JUDGMENT_PARENT_AGENT_ID: "judgment.parent_agent_id",
    JUDGMENT_AGENT_CLASS_NAME: "judgment.agent_class_name",
    JUDGMENT_AGENT_INSTANCE_NAME: "judgment.agent_instance_name",

    PENDING_TRACE_EVAL: "judgment.pending_trace_eval",

    GEN_AI_PROMPT: "gen_ai.prompt",
    GEN_AI_COMPLETION: "gen_ai.completion",
    GEN_AI_REQUEST_MODEL: "gen_ai.request.model",
    GEN_AI_RESPONSE_MODEL: "gen_ai.response.model",
    GEN_AI_SYSTEM: "gen_ai.system",
    GEN_AI_USAGE_INPUT_TOKENS: "gen_ai.usage.input_tokens",
    GEN_AI_USAGE_OUTPUT_TOKENS: "gen_ai.usage.output_tokens",
    GEN_AI_USAGE_COMPLETION_TOKENS: "gen_ai.usage.completion_tokens",
    GEN_AI_REQUEST_TEMPERATURE: "gen_ai.request.temperature",
    GEN_AI_REQUEST_MAX_TOKENS: "gen_ai.request.max_tokens",
    GEN_AI_RESPONSE_FINISH_REASONS: "gen_ai.response.finish_reasons",

    GEN_AI_USAGE_TOTAL_COST: "gen_ai.usage.total_cost_usd",
  } as const;

  public static readonly InternalAttributeKeys = {
    DISABLE_PARTIAL_EMIT: "disable_partial_emit",
    CANCELLED: "cancelled",
  } as const;

  public static readonly ResourceKeys = {
    SERVICE_NAME: "service.name",
    TELEMETRY_SDK_LANGUAGE: "telemetry.sdk.language",
    TELEMETRY_SDK_NAME: "telemetry.sdk.name",
    TELEMETRY_SDK_VERSION: "telemetry.sdk.version",
    JUDGMENT_PROJECT_ID: "judgment.project_id",
  } as const;
}
