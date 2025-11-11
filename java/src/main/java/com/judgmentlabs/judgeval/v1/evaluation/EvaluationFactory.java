package com.judgmentlabs.judgeval.v1.evaluation;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;

/**
 * Factory for creating evaluation builders.
 */
public final class EvaluationFactory {
    private final JudgmentSyncClient client;

    public EvaluationFactory(JudgmentSyncClient client) {
        this.client = client;
    }

    /**
     * Creates a new evaluation builder configured with this factory's client.
     *
     * @return a new evaluation builder
     */
    public Evaluation.Builder create() {
        return Evaluation.builder().client(client);
    }
}
