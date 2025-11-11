package com.judgmentlabs.judgeval.tracer.exporters;

import java.util.Collection;
import java.util.Optional;

import com.judgmentlabs.judgeval.utils.Logger;

import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SpanExporter;

/**
 * SpanExporter implementation that sends spans to Judgment Labs with project
 * identification.
 * <p>
 * This exporter wraps the OTLP HTTP exporter and adds Judgment Labs specific
 * headers and project identification to all exported spans.
 */
public class JudgmentSpanExporter implements SpanExporter {
    private final SpanExporter delegate;

    /**
     * Creates a new JudgmentSpanExporter with the specified configuration.
     *
     * @param endpoint
     *            the OTLP endpoint URL
     * @param apiKey
     *            the API key for authentication
     * @param organizationId
     *            the organization ID
     * @param projectId
     *            the project ID (must not be null or empty)
     * @throws IllegalArgumentException
     *             if projectId is null or empty
     */
    protected JudgmentSpanExporter(String endpoint, String apiKey, String organizationId, String projectId) {
        if (projectId.isEmpty()) {
            throw new IllegalArgumentException("projectId is required for JudgmentSpanExporter");
        }
        this.delegate = OtlpHttpSpanExporter.builder()
                .setEndpoint(endpoint)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("X-Organization-Id", organizationId)
                .addHeader("X-Project-Id", projectId)
                .build();
    }

    /**
     * Creates a new builder for constructing JudgmentSpanExporter instances.
     *
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Exports the collection of spans to the Judgment Labs backend.
     *
     * @param spans
     *            the collection of spans to export
     * @return a CompletableResultCode representing the export operation status
     */
    @Override
    public CompletableResultCode export(Collection<SpanData> spans) {
        Logger.info("Exported " + spans.size() + " spans");
        return delegate.export(spans);
    }

    /**
     * Flushes any pending span exports.
     *
     * @return a CompletableResultCode representing the flush operation status
     */
    @Override
    public CompletableResultCode flush() {
        return delegate.flush();
    }

    /**
     * Shuts down this exporter and releases any resources.
     *
     * @return a CompletableResultCode representing the shutdown operation status
     */
    @Override
    public CompletableResultCode shutdown() {
        return delegate.shutdown();
    }

    /**
     * Builder for creating JudgmentSpanExporter instances.
     */
    public static final class Builder {
        private String endpoint;
        private String apiKey;
        private String organizationId;
        private String projectId;

        private Builder() {
        }

        /**
         * Sets the OTLP endpoint URL.
         *
         * @param endpoint
         *            the endpoint URL
         * @return this builder for method chaining
         */
        public Builder endpoint(String endpoint) {
            this.endpoint = endpoint;
            return this;
        }

        /**
         * Sets the API key for authentication.
         *
         * @param apiKey
         *            the API key
         * @return this builder for method chaining
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * Sets the organization ID.
         *
         * @param organizationId
         *            the organization ID
         * @return this builder for method chaining
         */
        public Builder organizationId(String organizationId) {
            this.organizationId = organizationId;
            return this;
        }

        /**
         * Sets the project ID.
         *
         * @param projectId
         *            the project ID
         * @return this builder for method chaining
         */
        public Builder projectId(String projectId) {
            this.projectId = projectId;
            return this;
        }

        /**
         * Builds a new JudgmentSpanExporter instance with the configured settings.
         *
         * @return a new JudgmentSpanExporter instance
         * @throws IllegalArgumentException
         *             if any required field is null or empty
         */
        public JudgmentSpanExporter build() {
            String validEndpoint = Optional.ofNullable(endpoint)
                    .map(String::trim)
                    .filter(e -> !e.isEmpty())
                    .orElseThrow(() -> new IllegalArgumentException("Endpoint is required"));
            String validApiKey = Optional.ofNullable(apiKey)
                    .map(String::trim)
                    .filter(key -> !key.isEmpty())
                    .orElseThrow(() -> new IllegalArgumentException("API key is required"));
            String validOrganizationId = Optional.ofNullable(organizationId)
                    .map(String::trim)
                    .filter(id -> !id.isEmpty())
                    .orElseThrow(() -> new IllegalArgumentException("Organization ID is required"));
            String validProjectId = Optional.ofNullable(projectId)
                    .map(String::trim)
                    .filter(id -> !id.isEmpty())
                    .orElseThrow(() -> new IllegalArgumentException("Project ID is required"));

            return new JudgmentSpanExporter(validEndpoint, validApiKey, validOrganizationId, validProjectId);
        }
    }
}
