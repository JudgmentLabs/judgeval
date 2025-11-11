package com.judgmentlabs.judgeval.data;

/**
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.data.APIScorerType}
 *             instead.
 */
@Deprecated
public enum APIScorerType {
    PROMPT_SCORER("Prompt Scorer"),
    TRACE_PROMPT_SCORER("Trace Prompt Scorer"),
    FAITHFULNESS("Faithfulness"),
    ANSWER_RELEVANCY("Answer Relevancy"),
    ANSWER_CORRECTNESS("Answer Correctness"),
    INSTRUCTION_ADHERENCE("Instruction Adherence"),
    EXECUTION_ORDER("Execution Order"),
    DERAILMENT("Derailment"),
    TOOL_ORDER("Tool Order"),
    CLASSIFIER("Classifier"),
    TOOL_DEPENDENCY("Tool Dependency"),
    CUSTOM("Custom");

    private final String value;

    APIScorerType(String value) {
        this.value = value;
    }

    public String getValue() {
        return value;
    }

    @Override
    public String toString() {
        return value;
    }
}
