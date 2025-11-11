package com.judgmentlabs.judgeval.tracer;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.judgmentlabs.judgeval.Env;
import com.judgmentlabs.judgeval.data.EvaluationRunBuilder;
import com.judgmentlabs.judgeval.data.Example;
import com.judgmentlabs.judgeval.data.TraceEvaluationRunBuilder;
import com.judgmentlabs.judgeval.internal.api.JudgmentSyncClient;
import com.judgmentlabs.judgeval.internal.api.models.ExampleEvaluationRun;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameRequest;
import com.judgmentlabs.judgeval.internal.api.models.ResolveProjectNameResponse;
import com.judgmentlabs.judgeval.internal.api.models.TraceEvaluationRun;
import com.judgmentlabs.judgeval.scorers.BaseScorer;
import com.judgmentlabs.judgeval.tracer.exporters.JudgmentSpanExporter;
import com.judgmentlabs.judgeval.tracer.exporters.NoOpSpanExporter;
import com.judgmentlabs.judgeval.utils.Logger;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.trace.export.SpanExporter;

public abstract class BaseTracer {
    public static final String          TRACER_NAME = "judgeval";

    protected final TracerConfiguration configuration;
    protected final JudgmentSyncClient  apiClient;
    protected final ISerializer         serializer;
    protected final ObjectMapper        jacksonMapper;
    protected final Optional<String>    projectId;

    protected BaseTracer(TracerConfiguration configuration, ISerializer serializer, boolean initialize) {
        this.configuration = Objects.requireNonNull(configuration, "Configuration cannot be null");
        this.apiClient = new JudgmentSyncClient(configuration.apiUrl(), configuration.apiKey(),
                configuration.organizationId());
        this.serializer = Objects.requireNonNull(serializer, "Serializer cannot be null");
        this.jacksonMapper = new ObjectMapper();
        this.projectId = resolveProjectId(configuration.projectName());
        this.projectId.ifPresentOrElse(id -> {
        }, () -> Logger.error("Failed to resolve project " + configuration.projectName()
                + ", please create it first at https://app.judgmentlabs.ai/org/" + configuration.organizationId()
                + "/projects. Skipping Judgment export."));

        if (initialize) {
            initialize();
        }
    }

    /**
     * Initializes the tracer with OpenTelemetry SDK configuration and span
     * exporters. Must be implemented by subclasses.
     */
    public abstract void initialize();

    /**
     * Gets the span exporter for sending traces to the Judgment Labs backend.
     * Returns a NoOpSpanExporter if the project ID is not resolved.
     *
     * @return the configured SpanExporter instance
     */
    public SpanExporter getSpanExporter() {
        return projectId.<SpanExporter>map(this::createJudgmentSpanExporter)
                .orElseGet(() -> {
                    Logger.error("Project not resolved; cannot create exporter, returning NoOpSpanExporter");
                    return new NoOpSpanExporter();
                });
    }

    /**
     * Sets the kind of the current span (e.g., "llm", "tool", "span").
     *
     * @param kind
     *            the span kind to set, ignored if null
     */
    public void setSpanKind(String kind) {
        Optional.ofNullable(kind)
                .ifPresent(k -> withCurrentSpan(
                        span -> span.setAttribute(JudgevalTraceKeys.AttributeKeys.JUDGMENT_SPAN_KIND, k)));
    }

    private static void withCurrentSpan(java.util.function.Consumer<Span> action) {
        Optional.ofNullable(Span.current())
                .ifPresent(action);
    }

    private static boolean isValidKey(String key) {
        return key != null && !key.isEmpty();
    }

    /**
     * Sets an attribute on the current span by serializing the object value.
     * Empty strings and null are valid attribute values, but not valid keys.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value (null and empty strings are valid)
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
     * Sets an attribute on the current span by serializing the object value with
     * the specified type. Empty strings and null are valid attribute values, but
     * not valid keys.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the attribute value (null and empty strings are valid)
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
     * Sets a string attribute on the current span. Empty strings and null are valid
     * attribute values, but not valid keys.
     *
     * @param key
     *            the attribute key
     * @param value
     *            the string value (null and empty strings are valid)
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
     *            the long value
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
     *            the double value
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
     *            the boolean value
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
        return configuration.enableEvaluation();
    }

    private void logEvaluationInfo(String method, String traceId, String spanId, String scorerName) {
        Logger.info(method + ": project=" + configuration.projectName() + ", traceId=" + traceId + ", spanId="
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
     * Asynchronously evaluates a scorer against an example, associating it with the
     * current trace.
     *
     * @param scorer
     *            the scorer to evaluate
     * @param example
     *            the example to evaluate against
     * @param model
     *            the model name, or null to use the default
     */
    public void asyncEvaluate(BaseScorer scorer, Example example, String model) {
        safeExecute("evaluate scorer", () -> {
            if (!isEvaluationEnabled()) {
                return;
            }

            getSampledSpanContext().ifPresent(spanContext -> {
                String traceId = spanContext.getTraceId();
                String spanId = spanContext.getSpanId();

                logEvaluationInfo("asyncEvaluate", traceId, spanId, scorer.getName());

                ExampleEvaluationRun evaluationRun = createEvaluationRun(scorer, example, model, traceId, spanId);
                enqueueEvaluation(evaluationRun);
            });
        });
    }

    /**
     * Asynchronously evaluates a scorer against an example using the default model.
     *
     * @param scorer
     *            the scorer to evaluate
     * @param example
     *            the example to evaluate against
     */
    public void asyncEvaluate(BaseScorer scorer, Example example) {
        asyncEvaluate(scorer, example, null);
    }

    /**
     * Asynchronously evaluates a scorer for the current trace, attaching the
     * evaluation as a span attribute.
     *
     * @param scorer
     *            the scorer to evaluate
     * @param model
     *            the model name, or null to use the default
     */
    public void asyncTraceEvaluate(BaseScorer scorer, String model) {
        safeExecute("evaluate trace scorer", () -> {
            if (!isEvaluationEnabled()) {
                return;
            }

            getSampledSpan().ifPresent(currentSpan -> {
                SpanContext spanContext = currentSpan.getSpanContext();
                String traceId = spanContext.getTraceId();
                String spanId = spanContext.getSpanId();

                logEvaluationInfo("asyncTraceEvaluate", traceId, spanId, scorer.getName());

                TraceEvaluationRun evaluationRun = createTraceEvaluationRun(scorer, model, traceId, spanId);
                try {
                    String traceEvalJson = jacksonMapper.writeValueAsString(evaluationRun);
                    currentSpan.setAttribute(JudgevalTraceKeys.AttributeKeys.PENDING_TRACE_EVAL, traceEvalJson);
                } catch (Exception e) {
                    Logger.error("Failed to serialize trace evaluation: " + e.getMessage());
                }
            });
        });
    }

    /**
     * Asynchronously evaluates a scorer for the current trace using the default
     * model.
     *
     * @param scorer
     *            the scorer to evaluate
     */
    public void asyncTraceEvaluate(BaseScorer scorer) {
        asyncTraceEvaluate(scorer, null);
    }

    /**
     * Sets multiple attributes on the current span from a map.
     *
     * @param attributes
     *            the map of attribute key-value pairs, ignored if null
     */
    public void setAttributes(Map<String, Object> attributes) {
        Optional.ofNullable(attributes)
                .ifPresent(attrs -> attrs.forEach(this::setAttribute));
    }

    /**
     * Sets the current span kind to "llm".
     */
    public void setLLMSpan() {
        setSpanKind("llm");
    }

    /**
     * Sets the current span kind to "tool".
     */
    public void setToolSpan() {
        setSpanKind("tool");
    }

    /**
     * Sets the current span kind to "span".
     */
    public void setGeneralSpan() {
        setSpanKind("span");
    }

    /**
     * Sets the input attribute on the current span by serializing the object.
     *
     * @param input
     *            the input object to set, ignored if null
     */
    public void setInput(Object input) {
        setAttribute(JudgevalTraceKeys.AttributeKeys.JUDGMENT_INPUT, input);
    }

    /**
     * Sets the output attribute on the current span by serializing the object.
     *
     * @param output
     *            the output object to set, ignored if null
     */
    public void setOutput(Object output) {
        setAttribute(JudgevalTraceKeys.AttributeKeys.JUDGMENT_OUTPUT, output);
    }

    /**
     * Sets the input attribute on the current span with a specific type for
     * serialization.
     *
     * @param input
     *            the input object to set, ignored if null
     * @param type
     *            the type to use for serialization
     */
    public void setInput(Object input, Type type) {
        setAttribute(JudgevalTraceKeys.AttributeKeys.JUDGMENT_INPUT, input, type);
    }

    /**
     * Sets the output attribute on the current span with a specific type for
     * serialization.
     *
     * @param output
     *            the output object to set, ignored if null
     * @param type
     *            the type to use for serialization
     */
    public void setOutput(Object output, Type type) {
        setAttribute(JudgevalTraceKeys.AttributeKeys.JUDGMENT_OUTPUT, output, type);
    }

    /**
     * Creates a new span, executes the runnable within its context, and ends the
     * span.
     *
     * @param spanName
     *            the name of the span
     * @param runnable
     *            the code to execute within the span context
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
     * Creates a new span, executes the callable within its context, and ends the
     * span.
     *
     * @param <T>
     *            the return type of the callable
     * @param spanName
     *            the name of the span
     * @param callable
     *            the code to execute within the span context
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
     * Gets the OpenTelemetry tracer instance.
     *
     * @return the configured Tracer instance
     */
    public Tracer getTracer() {
        return GlobalOpenTelemetry.get()
                .getTracer(TRACER_NAME);
    }

    /**
     * Gets the tracer configuration.
     *
     * @return the TracerConfiguration instance
     */
    public TracerConfiguration getConfiguration() {
        return configuration;
    }

    /**
     * Gets the resolved project ID, if available.
     *
     * @return an Optional containing the project ID, or empty if not resolved
     */
    public Optional<String> getProjectId() {
        return projectId;
    }

    /**
     * Creates and returns a new span with the given name. The span must be entered
     * with a try-with-resources block or manually ended by calling
     * {@link Span#end()}.
     *
     * @param spanName
     *            the name of the span
     * @return the newly created span
     */
    public static Span span(String spanName) {
        return GlobalOpenTelemetry.get()
                .getTracer(TRACER_NAME)
                .spanBuilder(spanName)
                .startSpan();
    }

    protected Optional<String> resolveProjectId(String name) {
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
                .endpoint(buildEndpoint(configuration.apiUrl()))
                .apiKey(configuration.apiKey())
                .organizationId(configuration.organizationId())
                .projectId(projectId)
                .build();
    }

    private String generateRunId(String prefix, String spanId) {
        return prefix + Optional.ofNullable(spanId)
                .orElseGet(() -> String.valueOf(System.currentTimeMillis()));
    }

    private String getModelName(String model) {
        return Optional.ofNullable(model)
                .orElse(Env.JUDGMENT_DEFAULT_GPT_MODEL);
    }

    private ExampleEvaluationRun createEvaluationRun(BaseScorer scorer, Example example, String model, String traceId,
            String spanId) {
        String runId = generateRunId("async_evaluate_", spanId);
        return new EvaluationRunBuilder()
                .projectName(configuration.projectName())
                .evalName(runId)
                .model(model)
                .example(example)
                .trace(traceId, spanId)
                .addScorer(scorer)
                .build();
    }

    private TraceEvaluationRun createTraceEvaluationRun(BaseScorer scorer, String model, String traceId,
            String spanId) {
        String evalName = generateRunId("async_trace_evaluate_", spanId);
        return new TraceEvaluationRunBuilder()
                .projectName(configuration.projectName())
                .evalName(evalName)
                .model(model)
                .trace(traceId, spanId)
                .addScorer(scorer)
                .build();
    }

    private static List<List<Object>> convertTraceAndSpanIds(List<List<String>> traceAndSpanIds) {
        if (traceAndSpanIds == null || traceAndSpanIds.isEmpty())
            throw new IllegalArgumentException("Trace and span IDs are required for trace evaluations.");
        List<List<Object>> converted = new java.util.ArrayList<>();
        for (List<String> pair : traceAndSpanIds) {
            if (pair == null || pair.size() != 2)
                throw new IllegalArgumentException("Each trace and span ID pair must contain exactly 2 elements.");
            converted.add(List.of(pair.get(0), pair.get(1)));
        }
        return converted;
    }

    private void enqueueEvaluation(ExampleEvaluationRun evaluationRun) {
        try {
            apiClient.addToRunEvalQueue(evaluationRun);
        } catch (Exception e) {
            Logger.error("Failed to enqueue evaluation run: " + e.getMessage());
        }
    }

    protected static class GsonSerializer implements ISerializer {
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
