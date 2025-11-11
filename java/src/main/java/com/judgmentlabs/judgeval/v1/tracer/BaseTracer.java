package com.judgmentlabs.judgeval.v1.tracer;

import java.lang.reflect.Type;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.judgmentlabs.judgeval.JudgmentAttributeKeys;
import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.ExampleEvaluationRun;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameRequest;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameResponse;
import com.judgmentlabs.judgeval.internal.api.models.TraceEvaluationRun;
import com.judgmentlabs.judgeval.utils.Logger;
import com.judgmentlabs.judgeval.v1.data.Example;
import com.judgmentlabs.judgeval.v1.scorers.BaseScorer;
import com.judgmentlabs.judgeval.v1.scorers.custom_scorer.CustomScorer;
import com.judgmentlabs.judgeval.v1.tracer.exporters.JudgmentSpanExporter;
import com.judgmentlabs.judgeval.v1.tracer.exporters.NoOpSpanExporter;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.trace.export.SpanExporter;

/**
 * Base tracer providing span manipulation, attribute setting, and evaluation
 * capabilities.
 */
public abstract class BaseTracer {
    public static final String         TRACER_NAME = "judgeval";

    protected final String             projectName;
    protected final boolean            enableEvaluation;
    protected final JudgmentSyncClient apiClient;
    protected final ISerializer        serializer;
    protected final ObjectMapper       jacksonMapper;
    protected final Optional<String>   projectId;

    protected BaseTracer(String projectName,
            boolean enableEvaluation, JudgmentSyncClient apiClient, ISerializer serializer) {
        this.projectName = Objects.requireNonNull(projectName, "projectName required");
        this.enableEvaluation = enableEvaluation;
        this.apiClient = Objects.requireNonNull(apiClient, "apiClient required");
        this.serializer = Objects.requireNonNull(serializer, "serializer required");
        this.jacksonMapper = new ObjectMapper();
        this.projectId = resolveProjectId(projectName);

        this.projectId.ifPresentOrElse(id -> {
        }, () -> Logger.error("Failed to resolve project " + projectName
                + ", please create it first at https://app.judgmentlabs.ai/org/" + this.apiClient.getOrganizationId()
                + "/projects. Skipping Judgment export."));
    }

    /**
     * Initializes the tracer.
     */
    public abstract void initialize();

    /**
     * Forces pending spans to flush.
     *
     * @param timeoutMillis
     *            maximum time to wait in milliseconds
     * @return true if flush succeeded within timeout
     */
    public abstract boolean forceFlush(int timeoutMillis);

    /**
     * Shuts down the tracer.
     *
     * @param timeoutMillis
     *            maximum time to wait for shutdown in milliseconds
     */
    public abstract void shutdown(int timeoutMillis);

    /**
     * Returns the span exporter for this tracer.
     *
     * @return the span exporter
     */
    public SpanExporter getSpanExporter() {
        return projectId.<SpanExporter>map(this::createJudgmentSpanExporter)
                .orElseGet(() -> {
                    Logger.error("Project not resolved; cannot create exporter, returning NoOpSpanExporter");
                    return new NoOpSpanExporter();
                });
    }

    /**
     * Sets the span kind attribute on the current span.
     *
     * @param kind
     *            the span kind
     */
    public void setSpanKind(String kind) {
        Optional.ofNullable(kind)
                .ifPresent(k -> withCurrentSpan(
                        span -> span.setAttribute(JudgmentAttributeKeys.AttributeKeys.JUDGMENT_SPAN_KIND, k)));
    }

    private static void withCurrentSpan(java.util.function.Consumer<Span> action) {
        Optional.ofNullable(Span.current())
                .ifPresent(action);
    }

    private static boolean isValidKey(String key) {
        return key != null && !key.isEmpty();
    }

    /**
     * Sets an attribute on the current span by serializing the value.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     */
    public void setAttribute(String key, Object value) {
        if (!isValidKey(key)) {
            return;
        }
        if (value != null) {
            setAttribute(key, value, value.getClass());
        }
    }

    /**
     * Sets an attribute on the current span by serializing the value with the
     * specified type.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     * @param type
     *            the type to use for serialization
     */
    public void setAttribute(String key, Object value, Type type) {
        if (!isValidKey(key)) {
            return;
        }
        if (value != null) {
            withCurrentSpan(span -> span.setAttribute(key, serializer.serialize(value, type)));
        }
    }

    /**
     * Sets a string attribute on the current span.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     */
    public void setAttribute(String key, String value) {
        if (!isValidKey(key)) {
            return;
        }
        withCurrentSpan(span -> span.setAttribute(key, value));
    }

    /**
     * Sets a long attribute on the current span.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     */
    public void setAttribute(String key, long value) {
        if (!isValidKey(key)) {
            return;
        }
        withCurrentSpan(span -> span.setAttribute(key, value));
    }

    /**
     * Sets a double attribute on the current span.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     */
    public void setAttribute(String key, double value) {
        if (!isValidKey(key)) {
            return;
        }
        withCurrentSpan(span -> span.setAttribute(key, value));
    }

    /**
     * Sets a boolean attribute on the current span.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value
     */
    public void setAttribute(String key, boolean value) {
        if (!isValidKey(key)) {
            return;
        }
        withCurrentSpan(span -> span.setAttribute(key, value));
    }

    private Optional<SpanContext> getSampledSpanContext() {
        return Optional.ofNullable(Span.current())
                .filter(span -> span.getSpanContext()
                        .isSampled())
                .map(Span::getSpanContext);
    }

    private Optional<Span> getSampledSpan() {
        return Optional.ofNullable(Span.current())
                .filter(span -> span.getSpanContext()
                        .isSampled());
    }

    private boolean isEvaluationEnabled() {
        return enableEvaluation;
    }

    private void logEvaluationInfo(String method, String traceId, String spanId, String scorerName) {
        Logger.info(method + ": project=" + projectName + ", traceId=" + traceId + ", spanId="
                + spanId + ", scorer=" + scorerName);
    }

    private void safeExecute(String operation, Runnable action) {
        try {
            action.run();
        } catch (Exception e) {
            Logger.error("Failed to " + operation + ": " + e.getMessage());
        }
    }

    /**
     * Asynchronously evaluates the current span using the specified scorer and
     * example.
     * The evaluation is queued and processed asynchronously by the Judgment
     * backend.
     *
     * @param scorer
     *            the scorer to use for evaluation
     * @param example
     *            the example data to evaluate against
     */
    public void asyncEvaluate(BaseScorer scorer, Example example) {
        safeExecute("evaluate scorer", () -> {
            if (!isEvaluationEnabled()) {
                return;
            }

            getSampledSpanContext().ifPresent(spanContext -> {
                String traceId = spanContext.getTraceId();
                String spanId = spanContext.getSpanId();

                logEvaluationInfo("asyncEvaluate", traceId, spanId, scorer.getName());

                ExampleEvaluationRun evaluationRun = createEvaluationRun(scorer, example, traceId, spanId);
                enqueueEvaluation(evaluationRun);
            });
        });
    }

    /**
     * Asynchronously evaluates the current trace using the specified scorer.
     * Attaches evaluation metadata to the current span for processing after trace
     * completion.
     *
     * @param scorer
     *            the scorer to use for trace evaluation
     */
    public void asyncTraceEvaluate(BaseScorer scorer) {
        safeExecute("evaluate trace scorer", () -> {
            if (!isEvaluationEnabled()) {
                return;
            }

            getSampledSpan().ifPresent(currentSpan -> {
                SpanContext spanContext = currentSpan.getSpanContext();
                String traceId = spanContext.getTraceId();
                String spanId = spanContext.getSpanId();

                logEvaluationInfo("asyncTraceEvaluate", traceId, spanId, scorer.getName());

                TraceEvaluationRun evaluationRun = createTraceEvaluationRun(scorer, traceId, spanId);
                try {
                    String traceEvalJson = jacksonMapper.writeValueAsString(evaluationRun);
                    currentSpan.setAttribute(JudgmentAttributeKeys.AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL,
                            traceEvalJson);
                } catch (Exception e) {
                    Logger.error("Failed to serialize trace evaluation: " + e.getMessage());
                }
            });
        });
    }

    /**
     * Sets multiple attributes on the current span.
     *
     * @param attributes
     *            map of attribute keys to values
     */
    public void setAttributes(Map<String, Object> attributes) {
        Optional.ofNullable(attributes)
                .ifPresent(attrs -> attrs.forEach(this::setAttribute));
    }

    /**
     * Marks the current span as an LLM span.
     */
    public void setLLMSpan() {
        setSpanKind("llm");
    }

    /**
     * Marks the current span as a tool span.
     */
    public void setToolSpan() {
        setSpanKind("tool");
    }

    /**
     * Marks the current span as a general span.
     */
    public void setGeneralSpan() {
        setSpanKind("span");
    }

    /**
     * Sets the input attribute on the current span.
     *
     * @param input
     *            the input value
     */
    public void setInput(Object input) {
        setInput(input, input.getClass());
    }

    /**
     * Sets the output attribute on the current span.
     *
     * @param output
     *            the output value
     */
    public void setOutput(Object output) {
        setOutput(output, output.getClass());
    }

    /**
     * Sets the input attribute on the current span using the specified type.
     *
     * @param input
     *            the input value
     * @param type
     *            the type to use for serialization
     */
    public void setInput(Object input, Type type) {
        setAttribute(JudgmentAttributeKeys.AttributeKeys.JUDGMENT_INPUT, input, type);
    }

    /**
     * Sets the output attribute on the current span using the specified type.
     *
     * @param output
     *            the output value
     * @param type
     *            the type to use for serialization
     */
    public void setOutput(Object output, Type type) {
        setAttribute(JudgmentAttributeKeys.AttributeKeys.JUDGMENT_OUTPUT, output, type);
    }

    /**
     * Executes a runnable within a new span, automatically handling span lifecycle
     * and errors.
     *
     * @param spanName
     *            the name of the span
     * @param runnable
     *            the code to execute within the span
     */
    public void span(String spanName, Runnable runnable) {
        Span span = getTracer().spanBuilder(spanName)
                .startSpan();
        try (Scope scope = span.makeCurrent()) {
            runnable.run();
        } catch (Exception e) {
            span.setStatus(StatusCode.ERROR).recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Executes a callable within a new span, automatically handling span lifecycle
     * and errors.
     *
     * @param <T>
     *            the return type
     * @param spanName
     *            the name of the span
     * @param callable
     *            the code to execute within the span
     * @return the result of the callable
     * @throws Exception
     *             if the callable throws an exception
     */
    public <T> T span(String spanName, java.util.concurrent.Callable<T> callable) throws Exception {
        Span span = getTracer().spanBuilder(spanName)
                .startSpan();
        try (Scope scope = span.makeCurrent()) {
            return callable.call();
        } catch (Exception e) {
            span.setStatus(StatusCode.ERROR).recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }

    /**
     * Returns the OpenTelemetry tracer instance.
     *
     * @return the OpenTelemetry tracer
     */
    public io.opentelemetry.api.trace.Tracer getTracer() {
        return GlobalOpenTelemetry.get()
                .getTracer(TRACER_NAME);
    }

    /**
     * Returns the project name.
     *
     * @return the project name
     */
    public String getProjectName() {
        return projectName;
    }

    /**
     * Returns whether evaluation is enabled.
     *
     * @return true if evaluation is enabled
     */
    public boolean isEnableEvaluation() {
        return enableEvaluation;
    }

    /**
     * Returns the resolved project ID if available.
     *
     * @return the project ID, or empty if not resolved
     */
    public Optional<String> getProjectId() {
        return projectId;
    }

    /**
     * Creates and returns a new span with the specified name.
     *
     * @param spanName
     *            the name of the span
     * @return the created span
     */
    public static Span span(String spanName) {
        return GlobalOpenTelemetry.get()
                .getTracer(TRACER_NAME)
                .spanBuilder(spanName)
                .startSpan();
    }

    private Optional<String> resolveProjectId(String name) {
        try {
            ResolveProjectNameRequest request = new ResolveProjectNameRequest();
            request.setProjectName(name);
            ResolveProjectNameResponse response = apiClient.projectsResolve(request);
            return Optional.ofNullable(response.getProjectId())
                    .map(Object::toString);
        } catch (Exception e) {
            return Optional.empty();
        }
    }

    private String buildEndpoint(String baseUrl) {
        return baseUrl.endsWith("/") ? baseUrl + "otel/v1/traces" : baseUrl + "/otel/v1/traces";
    }

    private JudgmentSpanExporter createJudgmentSpanExporter(String projectId) {
        return JudgmentSpanExporter.builder()
                .endpoint(buildEndpoint(apiClient.getApiUrl()))
                .apiKey(apiClient.getApiKey())
                .organizationId(apiClient.getOrganizationId())
                .projectId(projectId)
                .build();
    }

    private String generateRunId(String prefix, String spanId) {
        return prefix + Optional.ofNullable(spanId)
                .orElseGet(() -> String.valueOf(System.currentTimeMillis()));
    }

    private ExampleEvaluationRun createEvaluationRun(BaseScorer scorer, Example example, String traceId,
            String spanId) {
        String runId = generateRunId("async_evaluate_", spanId);

        ExampleEvaluationRun evaluationRun = new ExampleEvaluationRun();
        evaluationRun.setId(UUID.randomUUID().toString());
        evaluationRun.setProjectName(projectName);
        evaluationRun.setEvalName(runId);
        evaluationRun.setTraceId(traceId);
        evaluationRun.setTraceSpanId(spanId);
        evaluationRun.setExamples(List.of(example));

        if (scorer instanceof CustomScorer) {
            evaluationRun.setJudgmentScorers(List.of());
            evaluationRun.setCustomScorers(List.of((com.judgmentlabs.judgeval.internal.api.models.BaseScorer) scorer));
        } else {
            evaluationRun.setJudgmentScorers(List.of(scorer.getScorerConfig()));
            evaluationRun.setCustomScorers(List.of());
        }
        evaluationRun.setCreatedAt(Instant.now().atOffset(ZoneOffset.UTC).format(DateTimeFormatter.ISO_INSTANT));

        return evaluationRun;
    }

    private TraceEvaluationRun createTraceEvaluationRun(BaseScorer scorer, String traceId,
            String spanId) {
        String evalName = generateRunId("async_trace_evaluate_", spanId);

        TraceEvaluationRun evaluationRun = new TraceEvaluationRun();
        evaluationRun.setId(UUID.randomUUID().toString());
        evaluationRun.setProjectName(projectName);
        evaluationRun.setEvalName(evalName);
        evaluationRun.setTraceAndSpanIds(List.of(List.of(traceId, spanId)));
        evaluationRun.setJudgmentScorers(List.of(scorer.getScorerConfig()));
        evaluationRun.setCustomScorers(List.of());
        evaluationRun.setIsOffline(false);
        evaluationRun.setIsBucketRun(false);
        evaluationRun.setCreatedAt(Instant.now().atOffset(ZoneOffset.UTC).format(DateTimeFormatter.ISO_INSTANT));

        return evaluationRun;
    }

    private void enqueueEvaluation(ExampleEvaluationRun evaluationRun) {
        try {
            apiClient.addToRunEvalQueue(evaluationRun);
        } catch (Exception e) {
            Logger.error("Failed to enqueue evaluation run: " + e.getMessage());
        }
    }
}
