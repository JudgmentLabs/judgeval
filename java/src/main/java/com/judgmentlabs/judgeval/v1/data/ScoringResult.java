package com.judgmentlabs.judgeval.v1.data;

import java.util.List;

/**
 * Represents a collection of scorer evaluation results.
 */
public class ScoringResult extends com.judgmentlabs.judgeval.internal.api.models.ScoringResult {

    /**
     * Creates a new builder for configuring a ScoringResult.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating ScoringResult instances.
     */
    public static final class Builder {
        private final ScoringResult result;

        private Builder() {
            this.result = new ScoringResult();
        }

        /**
         * Sets whether the overall evaluation succeeded.
         *
         * @param success
         *            true if evaluation succeeded
         * @return this builder
         */
        public Builder success(Boolean success) {
            result.setSuccess(success);
            return this;
        }

        /**
         * Sets the list of scorer results.
         *
         * @param scorersData
         *            the list of scorer data
         * @return this builder
         */
        public Builder scorersData(List<ScorerData> scorersData) {
            @SuppressWarnings("unchecked")
            List<com.judgmentlabs.judgeval.internal.api.models.ScorerData> internalList = (List<com.judgmentlabs.judgeval.internal.api.models.ScorerData>) (List<?>) scorersData;
            result.setScorersData(internalList);
            return this;
        }

        /**
         * Adds a single scorer result.
         *
         * @param scorerData
         *            the scorer data to add
         * @return this builder
         */
        public Builder scorerData(ScorerData scorerData) {
            if (result.getScorersData() == null) {
                result.setScorersData(new java.util.ArrayList<>());
            }
            result.getScorersData()
                    .add(scorerData);
            return this;
        }

        /**
         * Sets the data object for the evaluation.
         *
         * @param dataObject
         *            the example data
         * @return this builder
         */
        public Builder dataObject(Example dataObject) {
            if (dataObject != null) {
                result.setAdditionalProperty("example", dataObject);
            }
            return this;
        }

        /**
         * Builds and returns the configured ScoringResult.
         *
         * @return the configured ScoringResult
         */
        public ScoringResult build() {
            return result;
        }
    }
}
