package com.judgmentlabs.judgeval.v1.data;

/**
 * Available types of API-based scorers.
 */
public enum APIScorerType {
    PROMPT_SCORER("Prompt Scorer"),
    TRACE_PROMPT_SCORER("Trace Prompt Scorer"),
    FAITHFULNESS("Faithfulness"),
    ANSWER_RELEVANCY("Answer Relevancy"),
    ANSWER_CORRECTNESS("Answer Correctness"),
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
