package com.judgmentlabs.judgeval.scorers.built_in;

import java.util.Arrays;

import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.scorers.APIScorer;

/**
 * Scorer that evaluates the correctness of an answer against an expected
 * output.
 */
public final class AnswerCorrectnessScorer extends APIScorer {
    private AnswerCorrectnessScorer(Builder builder) {
        super(APIScorerType.ANSWER_CORRECTNESS);
        setRequiredParams(Arrays.asList("input", "actual_output", "expected_output"));
        if (builder.threshold >= 0) {
            setThreshold(builder.threshold);
        }
        if (builder.name != null) {
            setName(builder.name);
        }
        if (builder.strictMode != null) {
            setStrictMode(builder.strictMode);
        }
        if (builder.model != null) {
            setModel(builder.model);
        }
    }

    /**
     * Creates a new builder for configuring an AnswerCorrectnessScorer.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating AnswerCorrectnessScorer instances.
     */
    public static final class Builder {
        private double  threshold = 0.5;
        private String  name;
        private Boolean strictMode;
        private String  model;

        public Builder threshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder strictMode(boolean strictMode) {
            this.strictMode = strictMode;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public AnswerCorrectnessScorer build() {
            return new AnswerCorrectnessScorer(this);
        }
    }
}
