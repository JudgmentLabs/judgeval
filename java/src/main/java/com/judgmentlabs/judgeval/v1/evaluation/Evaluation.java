package com.judgmentlabs.judgeval.v1.evaluation;

import java.util.Objects;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;

/**
 * Evaluation configuration for running evaluations against traces and spans.
 */
public final class Evaluation {
    @SuppressWarnings("unused") // TODO: will add run_evaluation here
    private final JudgmentSyncClient client;

    private Evaluation(Builder builder) {
        this.client = Objects.requireNonNull(builder.client, "client required");
    }

    /**
     * Creates a new builder for configuring an Evaluation.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating Evaluation instances.
     */
    public static final class Builder {
        private JudgmentSyncClient client;

        Builder client(JudgmentSyncClient client) {
            this.client = client;
            return this;
        }

        /**
         * Builds and returns a new Evaluation instance.
         *
         * @return the configured Evaluation
         */
        public Evaluation build() {
            return new Evaluation(this);
        }
    }
}
