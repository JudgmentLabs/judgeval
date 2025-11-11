package com.judgmentlabs.judgeval.tracer;

import java.util.Optional;

/**
 * Configuration for the Judgment Tracer that controls how tracing and
 * evaluation behave.
 * <p>
 * This class encapsulates all configuration parameters needed to initialize a
 * {@link Tracer}.
 * <p>
 * Example usage:
 *
 * <pre>{@code
 * TracerConfiguration config = TracerConfiguration.builder()
 *         .projectName("my-project")
 *         .apiKey("your-api-key")
 *         .organizationId("your-org-id")
 *         .enableEvaluation(true)
 *         .build();
 *
 * Tracer tracer = Tracer.createWithConfiguration(config);
 * }</pre>
 *
 * @see Tracer
 * @deprecated Replaced by
 *             com.judgmentlabs.judgeval.v1.tracer.TracerConfiguration
 */
@Deprecated
public final class TracerConfiguration {
    private final String  projectName;
    private final String  apiKey;
    private final String  organizationId;
    private final String  apiUrl;
    private final boolean enableEvaluation;

    private TracerConfiguration(Builder builder) {
        this.projectName = Optional.ofNullable(builder.projectName)
                .map(String::trim)
                .filter(name -> !name.isEmpty())
                .orElseThrow(() -> new IllegalArgumentException("Project name cannot be null or empty"));
        this.apiKey = Optional.ofNullable(builder.apiKey)
                .orElseThrow(() -> new IllegalArgumentException("API key cannot be null"));
        this.organizationId = Optional.ofNullable(builder.organizationId)
                .orElseThrow(() -> new IllegalArgumentException("Organization ID cannot be null"));
        this.apiUrl = Optional.ofNullable(builder.apiUrl)
                .orElseThrow(() -> new IllegalArgumentException("API URL cannot be null"));
        this.enableEvaluation = builder.enableEvaluation;
    }

    public String projectName() {
        return projectName;
    }

    public String apiKey() {
        return apiKey;
    }

    public String organizationId() {
        return organizationId;
    }

    public String apiUrl() {
        return apiUrl;
    }

    public boolean enableEvaluation() {
        return enableEvaluation;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a default configuration with the given project name.
     * <p>
     * This method uses default values from environment variables:
     * <ul>
     * <li>API Key: {@code Env.JUDGMENT_API_KEY}
     * <li>Organization ID: {@code Env.JUDGMENT_ORG_ID}
     * <li>API URL: {@code Env.JUDGMENT_API_URL}
     * <li>Evaluation: enabled
     * </ul>
     *
     * @param projectName
     *            the name of the project
     * @return a new TracerConfiguration with default values
     * @throws IllegalArgumentException
     *             if project name is null or empty
     */
    public static TracerConfiguration createDefault(String projectName) {
        return builder().projectName(projectName)
                .build();
    }

    /**
     * Builder for creating TracerConfiguration instances.
     * <p>
     * Example usage:
     *
     * <pre>{@code
     * TracerConfiguration config = TracerConfiguration.builder()
     *         .projectName("my-project")
     *         .apiKey("custom-api-key")
     *         .organizationId("custom-org-id")
     *         .apiUrl("https://custom-api.judgmentlabs.ai")
     *         .enableEvaluation(false)
     *         .build();
     * }</pre>
     */
    public static final class Builder {
        private String  projectName;
        private String  apiKey           = com.judgmentlabs.judgeval.Env.JUDGMENT_API_KEY;
        private String  organizationId   = com.judgmentlabs.judgeval.Env.JUDGMENT_ORG_ID;
        private String  apiUrl           = com.judgmentlabs.judgeval.Env.JUDGMENT_API_URL;
        private boolean enableEvaluation = true;

        public Builder projectName(String projectName) {
            this.projectName = projectName;
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder organizationId(String organizationId) {
            this.organizationId = organizationId;
            return this;
        }

        public Builder apiUrl(String apiUrl) {
            this.apiUrl = apiUrl;
            return this;
        }

        public Builder enableEvaluation(boolean enableEvaluation) {
            this.enableEvaluation = enableEvaluation;
            return this;
        }

        public TracerConfiguration build() {
            return new TracerConfiguration(this);
        }
    }
}
