package com.judgmentlabs.judgeval.v1.tracer;

import java.lang.reflect.Type;
import java.util.Objects;
import java.util.Optional;

import com.google.gson.Gson;
import com.judgmentlabs.judgeval.Version;
import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.utils.Logger;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;

public final class Tracer extends BaseTracer {
    private SdkTracerProvider tracerProvider;

    private Tracer(Builder builder) {
        super(
                Objects.requireNonNull(builder.projectName, "projectName required"),
                builder.enableEvaluation,
                Objects.requireNonNull(builder.client, "client required"),
                builder.serializer != null ? builder.serializer : new GsonSerializer());

        if (builder.initialize) {
            initialize();
        }
    }

    /**
     * Initializes the tracer by setting up the OpenTelemetry SDK with a span
     * exporter,
     * configuring the tracer provider with batch span processing, and registering
     * it globally.
     */
    @Override
    public void initialize() {
        SpanExporter spanExporter = getSpanExporter();

        var resource = Resource.getDefault()
                .merge(Resource.create(Attributes.builder()
                        .put("service.name", projectName)
                        .put("telemetry.sdk.name", TRACER_NAME)
                        .put("telemetry.sdk.version", Version.getVersion())
                        .build()));

        this.tracerProvider = SdkTracerProvider.builder()
                .setResource(resource)
                .addSpanProcessor(BatchSpanProcessor.builder(spanExporter)
                        .build())
                .build();

        OpenTelemetry openTelemetry = OpenTelemetrySdk.builder()
                .setTracerProvider(this.tracerProvider)
                .build();

        GlobalOpenTelemetry.set(openTelemetry);
    }

    /**
     * Forces the tracer to flush any pending spans within the specified timeout.
     *
     * @param timeoutMillis
     *            the maximum time to wait in milliseconds
     * @return true if the flush completed successfully within the timeout
     */
    @Override
    public boolean forceFlush(int timeoutMillis) {
        if (tracerProvider == null) {
            Logger.error("Cannot forceFlush: tracer not initialized");
            return false;
        }
        return tracerProvider.forceFlush()
                .join(timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS)
                .isSuccess();
    }

    /**
     * Shuts down the tracer, flushing any remaining spans and releasing resources.
     *
     * @param timeoutMillis
     *            the maximum time to wait for shutdown in milliseconds
     */
    @Override
    public void shutdown(int timeoutMillis) {
        if (tracerProvider == null) {
            Logger.error("Cannot shutdown: tracer not initialized");
            return;
        }
        tracerProvider.shutdown()
                .join(timeoutMillis, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    /**
     * Creates a new builder for configuring a Tracer.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for configuring and creating Tracer instances.
     */
    public static final class Builder {
        private JudgmentSyncClient client;
        private String             projectName;
        private boolean            enableEvaluation = true;
        private ISerializer        serializer;
        private boolean            initialize       = true;

        /**
         * Sets the Judgment API client.
         *
         * @param client
         *            the API client
         * @return this builder
         */
        public Builder client(JudgmentSyncClient client) {
            this.client = client;
            return this;
        }

        /**
         * Sets the project name for this tracer.
         *
         * @param projectName
         *            the project name
         * @return this builder
         */
        public Builder projectName(String projectName) {
            this.projectName = projectName;
            return this;
        }

        /**
         * Sets whether evaluation is enabled.
         *
         * @param enableEvaluation
         *            true to enable evaluation
         * @return this builder
         */
        public Builder enableEvaluation(boolean enableEvaluation) {
            this.enableEvaluation = enableEvaluation;
            return this;
        }

        /**
         * Sets the custom serializer for span attributes.
         *
         * @param serializer
         *            the serializer
         * @return this builder
         */
        public Builder serializer(ISerializer serializer) {
            this.serializer = serializer;
            return this;
        }

        /**
         * Sets whether to automatically initialize the tracer on build.
         *
         * @param initialize
         *            true to initialize on build
         * @return this builder
         */
        public Builder initialize(boolean initialize) {
            this.initialize = initialize;
            return this;
        }

        /**
         * Builds and returns a new Tracer instance.
         *
         * @return the configured Tracer
         */
        public Tracer build() {
            return new Tracer(this);
        }
    }

    private static class GsonSerializer implements ISerializer {
        private final Gson gson = new Gson();

        @Override
        public String serialize(Object obj) {
            return Optional.ofNullable(obj)
                    .map(o -> serialize(o, o.getClass()))
                    .orElse(null);
        }

        @Override
        public String serialize(Object obj, Type type) {
            try {
                return gson.toJson(obj, type);
            } catch (Exception e) {
                Logger.error("Failed to serialize object: " + e.getMessage());
                return Optional.ofNullable(obj)
                        .map(Object::toString)
                        .orElse(null);
            }
        }
    }
}
