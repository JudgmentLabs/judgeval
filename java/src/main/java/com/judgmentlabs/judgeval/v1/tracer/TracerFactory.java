package com.judgmentlabs.judgeval.v1.tracer;

import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;

/**
 * Factory for creating tracer builders.
 */
public final class TracerFactory {
    private final JudgmentSyncClient client;

    public TracerFactory(JudgmentSyncClient client) {
        this.client = client;
    }

    /**
     * Creates a new tracer builder configured with this factory's client.
     *
     * @return a new tracer builder
     */
    public Tracer.Builder create() {
        return Tracer.builder()
                .client(client);
    }
}
