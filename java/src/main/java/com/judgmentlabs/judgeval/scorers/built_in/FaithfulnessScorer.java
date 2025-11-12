package com.judgmentlabs.judgeval.scorers.built_in;

import java.util.Arrays;

import com.judgmentlabs.judgeval.data.APIScorerType;
import com.judgmentlabs.judgeval.scorers.APIScorer;

/**
 * Scorer that evaluates whether an answer is faithful to the provided context.
 */
public final class FaithfulnessScorer extends APIScorer {
    private FaithfulnessScorer(Builder builder) {
        super(APIScorerType.FAITHFULNESS);
        setRequiredParams(Arrays.asList("context", "actual_output"));
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
     * Creates a new builder for configuring a FaithfulnessScorer.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating FaithfulnessScorer instances.
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

        public FaithfulnessScorer build() {
            return new FaithfulnessScorer(this);
        }
    }
}
