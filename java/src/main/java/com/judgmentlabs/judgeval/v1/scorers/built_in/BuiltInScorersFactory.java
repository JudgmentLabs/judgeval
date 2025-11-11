package com.judgmentlabs.judgeval.v1.scorers.built_in;

/**
 * Factory for creating built-in scorer builders.
 */
public final class BuiltInScorersFactory {
    public BuiltInScorersFactory() {
    }

    /**
     * Creates a builder for an answer correctness scorer.
     *
     * @return the scorer builder
     */
    public AnswerCorrectnessScorer.Builder answerCorrectness() {
        return AnswerCorrectnessScorer.builder();
    }

    /**
     * Creates a builder for an answer relevancy scorer.
     *
     * @return the scorer builder
     */
    public AnswerRelevancyScorer.Builder answerRelevancy() {
        return AnswerRelevancyScorer.builder();
    }

    /**
     * Creates a builder for a faithfulness scorer.
     *
     * @return the scorer builder
     */
    public FaithfulnessScorer.Builder faithfulness() {
        return FaithfulnessScorer.builder();
    }
}
