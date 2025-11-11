package tracer

var AttributeKeys = struct {
	JudgmentSpanKind                   string
	JudgmentInput                      string
	JudgmentOutput                     string
	JudgmentOfflineMode                string
	JudgmentUpdateID                   string
	JudgmentCustomerID                 string
	JudgmentAgentID                    string
	JudgmentParentAgentID              string
	JudgmentAgentClassName             string
	JudgmentAgentInstanceName          string
	JudgmentIsAgentEntryPoint          string
	JudgmentCumulativeLLMCost          string
	JudgmentStateBefore                string
	JudgmentStateAfter                 string
	PendingTraceEval                   string
	GenAIPrompt                        string
	GenAICompletion                    string
	GenAIRequestModel                  string
	GenAIResponseModel                 string
	GenAISystem                        string
	GenAIUsageInputTokens              string
	GenAIUsageOutputTokens             string
	GenAIUsageCacheCreationInputTokens string
	GenAIUsageCacheReadInputTokens     string
	GenAIRequestTemperature            string
	GenAIRequestMaxTokens              string
	GenAIResponseFinishReasons         string
}{
	JudgmentSpanKind:                   "judgment.span_kind",
	JudgmentInput:                      "judgment.input",
	JudgmentOutput:                     "judgment.output",
	JudgmentOfflineMode:                "judgment.offline_mode",
	JudgmentUpdateID:                   "judgment.update_id",
	JudgmentCustomerID:                 "judgment.customer_id",
	JudgmentAgentID:                    "judgment.agent_id",
	JudgmentParentAgentID:              "judgment.parent_agent_id",
	JudgmentAgentClassName:             "judgment.agent_class_name",
	JudgmentAgentInstanceName:          "judgment.agent_instance_name",
	JudgmentIsAgentEntryPoint:          "judgment.is_agent_entry_point",
	JudgmentCumulativeLLMCost:          "judgment.cumulative_llm_cost",
	JudgmentStateBefore:                "judgment.state_before",
	JudgmentStateAfter:                 "judgment.state_after",
	PendingTraceEval:                   "judgment.pending_trace_eval",
	GenAIPrompt:                        "gen_ai.prompt",
	GenAICompletion:                    "gen_ai.completion",
	GenAIRequestModel:                  "gen_ai.request.model",
	GenAIResponseModel:                 "gen_ai.response.model",
	GenAISystem:                        "gen_ai.system",
	GenAIUsageInputTokens:              "gen_ai.usage.input_tokens",
	GenAIUsageOutputTokens:             "gen_ai.usage.output_tokens",
	GenAIUsageCacheCreationInputTokens: "gen_ai.usage.cache_creation_input_tokens",
	GenAIUsageCacheReadInputTokens:     "gen_ai.usage.cache_read_input_tokens",
	GenAIRequestTemperature:            "gen_ai.request.temperature",
	GenAIRequestMaxTokens:              "gen_ai.request.max_tokens",
	GenAIResponseFinishReasons:         "gen_ai.response.finish_reasons",
}

var ResourceKeys = struct {
	ServiceName          string
	TelemetrySDKLanguage string
	TelemetrySDKName     string
	TelemetrySDKVersion  string
}{
	ServiceName:          "service.name",
	TelemetrySDKLanguage: "telemetry.sdk.language",
	TelemetrySDKName:     "telemetry.sdk.name",
	TelemetrySDKVersion:  "telemetry.sdk.version",
}
