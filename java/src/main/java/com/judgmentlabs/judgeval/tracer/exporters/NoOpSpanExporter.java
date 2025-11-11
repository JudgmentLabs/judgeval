package com.judgmentlabs.judgeval.tracer.exporters;

import java.util.Collection;

import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SpanExporter;

/**
 * A no-op implementation of SpanExporter that discards all spans. Used as a
 * fallback when project resolution fails or when spans should not be exported.
 */
public class NoOpSpanExporter implements SpanExporter {
    /**
     * Discards the collection of spans without exporting.
     *
     * @param spans
     *            the collection of spans (ignored)
     * @return a successful CompletableResultCode
     */
    @Override
    public CompletableResultCode export(Collection<SpanData> spans) {
        return CompletableResultCode.ofSuccess();
    }

    /**
     * Performs a no-op flush operation.
     *
     * @return a successful CompletableResultCode
     */
    @Override
    public CompletableResultCode flush() {
        return CompletableResultCode.ofSuccess();
    }

    /**
     * Performs a no-op shutdown operation.
     *
     * @return a successful CompletableResultCode
     */
    @Override
    public CompletableResultCode shutdown() {
        return CompletableResultCode.ofSuccess();
    }
}
