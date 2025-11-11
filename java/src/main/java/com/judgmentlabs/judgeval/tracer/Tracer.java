package com.judgmentlabs.judgeval.tracer;

import java.util.Optional;

import com.judgmentlabs.judgeval.Version;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;

/**
 * Main tracer for Judgment Labs distributed tracing and evaluation.
 *
 * @deprecated Use {@link com.judgmentlabs.judgeval.v1.Judgeval} instead.
 * 
 *             <p>
 *             Migration example:
 * 
 *             <pre>{@code
 * // Old way:
 * Tracer tracer = Tracer.createDefault("my-project");
 * tracer.initialize();
 * 
 * // New way:
 * Judgeval client = Judgeval.builder().build();
 * Tracer tracer = client.tracer().create()
 *     .projectName("my-project")
 *     .build();
 * tracer.initialize();
 * }</pre>
 *
 * @see TracerConfiguration
 * @see SpanExporter
 * @see com.judgmentlabs.judgeval.scorers.BaseScorer
 * @see com.judgmentlabs.judgeval.data.Example
 */
@Deprecated
public final class Tracer extends BaseTracer {

    private Tracer(TracerConfiguration configuration, ISerializer serializer, boolean shouldInitialize) {
        super(configuration, serializer, shouldInitialize);
    }

    /**
     * Creates a new TracerBuilder for constructing a Tracer instance.
     *
     * @return a new TracerBuilder
     */
    public static TracerBuilder builder() {
        return new TracerBuilder();
    }

    /**
     * Creates a Tracer with default configuration for the given project name.
     *
     * @param projectName
     *            the name of the project
     * @return a new Tracer instance with default configuration
     */
    public static Tracer createDefault(String projectName) {
        return builder().configuration(TracerConfiguration.createDefault(projectName))
                .build();
    }

    /**
     * Creates a Tracer with the provided configuration.
     *
     * @param configuration
     *            the tracer configuration
     * @return a new Tracer instance with the given configuration
     */
    public static Tracer createWithConfiguration(TracerConfiguration configuration) {
        return builder().configuration(configuration)
                .build();
    }

    /**
     * Initializes the OpenTelemetry SDK with batch span processor and registers it
     * globally.
     */
    @Override
    public void initialize() {
        SpanExporter spanExporter = getSpanExporter();

        var resource = Resource.getDefault()
                .merge(Resource.create(Attributes.builder()
                        .put("service.name", configuration.projectName())
                        .put("telemetry.sdk.name", TRACER_NAME)
                        .put("telemetry.sdk.version", Version.getVersion())
                        .build()));

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
                .setResource(resource)
                .addSpanProcessor(BatchSpanProcessor.builder(spanExporter)
                        .build())
                .build();

        OpenTelemetry openTelemetry = OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .build();

        GlobalOpenTelemetry.set(openTelemetry);
    }

    public static final class TracerBuilder {
        private TracerConfiguration configuration;
        private ISerializer         serializer = new GsonSerializer();
        private boolean             initialize = false;

        /**
         * Sets the tracer configuration.
         *
         * @param configuration
         *            the configuration to use
         * @return this builder for method chaining
         */
        public TracerBuilder configuration(TracerConfiguration configuration) {
            this.configuration = configuration;
            return this;
        }

        /**
         * Sets the serializer to use for attribute serialization.
         *
         * @param serializer
         *            the serializer to use
         * @return this builder for method chaining
         */
        public TracerBuilder serializer(ISerializer serializer) {
            this.serializer = serializer;
            return this;
        }

        /**
         * Sets whether to automatically initialize the tracer after construction.
         *
         * @param initialize
         *            true to initialize automatically, false otherwise
         * @return this builder for method chaining
         */
        public TracerBuilder initialize(boolean initialize) {
            this.initialize = initialize;
            return this;
        }

        /**
         * Builds a new Tracer instance with the configured settings.
         *
         * @return a new Tracer instance
         * @throws IllegalArgumentException
         *             if configuration is not set
         */
        public Tracer build() {
            return Optional.ofNullable(configuration)
                    .map(config -> new Tracer(config, serializer, initialize))
                    .orElseThrow(() -> new IllegalArgumentException("Configuration is required"));
        }
    }
}
