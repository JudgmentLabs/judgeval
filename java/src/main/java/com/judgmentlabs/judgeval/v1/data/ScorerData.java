package com.judgmentlabs.judgeval.v1.data;

import java.util.Map;

/**
 * Represents the result of a single scorer evaluation.
 */
public class ScorerData extends com.judgmentlabs.judgeval.internal.api.models.ScorerData {

    /**
     * Creates a new builder for configuring ScorerData.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating ScorerData instances.
     */
    public static final class Builder {
        private final ScorerData scorerData;

        private Builder() {
            this.scorerData = new ScorerData();
        }

        /**
         * Sets the scorer name.
         *
         * @param name
         *            the scorer name
         * @return this builder
         */
        public Builder name(String name) {
            scorerData.setName(name);
            return this;
        }

        /**
         * Sets the evaluation score.
         *
         * @param score
         *            the score value
         * @return this builder
         */
        public Builder score(Double score) {
            scorerData.setScore(score);
            return this;
        }

        /**
         * Sets whether the evaluation succeeded.
         *
         * @param success
         *            true if evaluation succeeded
         * @return this builder
         */
        public Builder success(Boolean success) {
            scorerData.setSuccess(success);
            return this;
        }

        /**
         * Sets the reason for the evaluation result.
         *
         * @param reason
         *            the evaluation reason
         * @return this builder
         */
        public Builder reason(String reason) {
            scorerData.setReason(reason);
            return this;
        }

        /**
         * Sets the evaluation threshold.
         *
         * @param threshold
         *            the threshold value
         * @return this builder
         */
        public Builder threshold(Double threshold) {
            scorerData.setThreshold(threshold);
            return this;
        }

        /**
         * Sets strict mode for evaluation.
         *
         * @param strictMode
         *            true for strict mode
         * @return this builder
         */
        public Builder strictMode(Boolean strictMode) {
            scorerData.setStrictMode(strictMode);
            return this;
        }

        /**
         * Sets the model used for evaluation.
         *
         * @param evaluationModel
         *            the model name
         * @return this builder
         */
        public Builder evaluationModel(String evaluationModel) {
            scorerData.setEvaluationModel(evaluationModel);
            return this;
        }

        /**
         * Sets an error message if evaluation failed.
         *
         * @param error
         *            the error message
         * @return this builder
         */
        public Builder error(String error) {
            scorerData.setError(error);
            return this;
        }

        /**
         * Sets additional metadata for the evaluation.
         *
         * @param additionalMetadata
         *            the metadata map
         * @return this builder
         */
        public Builder additionalMetadata(Map<String, Object> additionalMetadata) {
            scorerData.setAdditionalMetadata(additionalMetadata);
            return this;
        }

        /**
         * Adds a single metadata entry.
         *
         * @param key
         *            the metadata key
         * @param value
         *            the metadata value
         * @return this builder
         */
        public Builder metadata(String key, Object value) {
            if (scorerData.getAdditionalMetadata() == null) {
                scorerData.setAdditionalMetadata(new java.util.HashMap<>());
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> metadata = (Map<String, Object>) scorerData.getAdditionalMetadata();
            metadata.put(key, value);
            return this;
        }

        /**
         * Builds and returns the configured ScorerData.
         *
         * @return the configured ScorerData
         */
        public ScorerData build() {
            return scorerData;
        }
    }
}
