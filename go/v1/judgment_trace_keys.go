package v1

const (
	AttributeKeysJudgmentSpanKind          = "judgment.span_kind"
	AttributeKeysJudgmentInput             = "judgment.input"
	AttributeKeysJudgmentOutput            = "judgment.output"
	AttributeKeysJudgmentOfflineMode       = "judgment.offline_mode"
	AttributeKeysJudgmentUpdateID          = "judgment.update_id"
	AttributeKeysJudgmentCustomerID        = "judgment.customer_id"
	AttributeKeysJudgmentAgentID           = "judgment.agent_id"
	AttributeKeysJudgmentParentAgentID     = "judgment.parent_agent_id"
	AttributeKeysJudgmentAgentClassName    = "judgment.agent_class_name"
	AttributeKeysJudgmentAgentInstanceName = "judgment.agent_instance_name"
	AttributeKeysJudgmentIsAgentEntryPoint = "judgment.is_agent_entry_point"
	AttributeKeysJudgmentCumulativeLLMCost = "judgment.cumulative_llm_cost"
	AttributeKeysJudgmentStateBefore       = "judgment.state_before"
	AttributeKeysJudgmentStateAfter        = "judgment.state_after"
	AttributeKeysPendingTraceEval          = "judgment.pending_trace_eval"

	AttributeKeysGenAIPrompt                        = "gen_ai.prompt"
	AttributeKeysGenAICompletion                    = "gen_ai.completion"
	AttributeKeysGenAIRequestModel                  = "gen_ai.request.model"
	AttributeKeysGenAIResponseModel                 = "gen_ai.response.model"
	AttributeKeysGenAISystem                        = "gen_ai.system"
	AttributeKeysGenAIUsageInputTokens              = "gen_ai.usage.input_tokens"
	AttributeKeysGenAIUsageOutputTokens             = "gen_ai.usage.output_tokens"
	AttributeKeysGenAIUsageCacheCreationInputTokens = "gen_ai.usage.cache_creation_input_tokens"
	AttributeKeysGenAIUsageCacheReadInputTokens     = "gen_ai.usage.cache_read_input_tokens"
	AttributeKeysGenAIRequestTemperature            = "gen_ai.request.temperature"
	AttributeKeysGenAIRequestMaxTokens              = "gen_ai.request.max_tokens"
	AttributeKeysGenAIResponseFinishReasons         = "gen_ai.response.finish_reasons"
)

const (
	ResourceKeysServiceName          = "service.name"
	ResourceKeysTelemetrySDKLanguage = "telemetry.sdk.language"
	ResourceKeysTelemetrySDKName     = "telemetry.sdk.name"
	ResourceKeysTelemetrySDKVersion  = "telemetry.sdk.version"
	ResourceKeysJudgmentProjectID    = "judgment.project_id"
)
