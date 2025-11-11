package tracer

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/JudgmentLabs/judgeval-go/pkg/data"
	"github.com/JudgmentLabs/judgeval-go/pkg/env"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
	"github.com/JudgmentLabs/judgeval-go/pkg/logger"
	"github.com/JudgmentLabs/judgeval-go/pkg/scorers"
	"github.com/JudgmentLabs/judgeval-go/pkg/tracer/exporters"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

const TracerName = "judgeval"

type TracerConfiguration struct {
	APIURL           string
	APIKey           string
	OrganizationID   string
	ProjectName      string
	EnableEvaluation bool
}

type TracerConfigurationOptions func(*TracerConfiguration)

func WithAPIURL(apiURL string) TracerConfigurationOptions {
	return func(config *TracerConfiguration) {
		config.APIURL = apiURL
	}
}

func WithAPIKey(apiKey string) TracerConfigurationOptions {
	return func(config *TracerConfiguration) {
		config.APIKey = apiKey
	}
}

func WithOrganizationID(organizationID string) TracerConfigurationOptions {
	return func(config *TracerConfiguration) {
		config.OrganizationID = organizationID
	}
}

func WithProjectName(projectName string) TracerConfigurationOptions {
	return func(config *TracerConfiguration) {
		config.ProjectName = projectName
	}
}

func WithEnableEvaluation(enableEvaluation bool) TracerConfigurationOptions {
	return func(config *TracerConfiguration) {
		config.EnableEvaluation = enableEvaluation
	}
}

func NewTracerConfiguration(options ...TracerConfigurationOptions) TracerConfiguration {
	config := TracerConfiguration{
		APIURL:         env.JudgmentAPIURL,
		APIKey:         env.JudgmentAPIKey,
		OrganizationID: env.JudgmentOrgID,
	}

	for _, option := range options {
		option(&config)
	}

	return config
}

type SerializerFunc func(obj interface{}) string

type baseTracer struct {
	configuration TracerConfiguration
	apiClient     *api.Client
	serializer    SerializerFunc
	projectID     string
	tracer        trace.Tracer
}

func (bt *baseTracer) GetSpanExporter() sdktrace.SpanExporter {
	if bt.projectID == "" {
		logger.Error("Project not resolved; cannot create exporter, returning NoOpSpanExporter")
		return exporters.NewNoOpSpanExporter()
	}
	return bt.createJudgmentSpanExporter(bt.projectID)
}

func (bt *baseTracer) SetSpanKind(span trace.Span, kind string) {
	if span.IsRecording() && kind != "" {
		span.SetAttributes(attribute.String(AttributeKeys.JudgmentSpanKind, kind))
	}
}

func (bt *baseTracer) SetAttribute(span trace.Span, key string, value interface{}) {
	if !span.IsRecording() {
		return
	}

	attr := bt.createAttribute(key, value)
	span.SetAttributes(attr)
}

func (bt *baseTracer) createAttribute(key string, value interface{}) attribute.KeyValue {
	switch v := value.(type) {
	case string:
		return attribute.String(key, v)
	case int:
		return attribute.Int(key, v)
	case int8:
		return attribute.Int64(key, int64(v))
	case int16:
		return attribute.Int64(key, int64(v))
	case int32:
		return attribute.Int64(key, int64(v))
	case int64:
		return attribute.Int64(key, v)
	case uint:
		return attribute.Int64(key, int64(v))
	case uint8:
		return attribute.Int64(key, int64(v))
	case uint16:
		return attribute.Int64(key, int64(v))
	case uint32:
		return attribute.Int64(key, int64(v))
	case uint64:
		return attribute.Int64(key, int64(v))
	case float32:
		return attribute.Float64(key, float64(v))
	case float64:
		return attribute.Float64(key, v)
	case bool:
		return attribute.Bool(key, v)
	case []string:
		return attribute.StringSlice(key, v)
	default:
		return attribute.String(key, bt.serializer(value))
	}
}

func (bt *baseTracer) AsyncEvaluate(ctx context.Context, scorer scorers.BaseScorer, example *data.Example, model string) {
	if !bt.configuration.EnableEvaluation {
		return
	}

	span := trace.SpanFromContext(ctx)
	logger.Debug("AsyncEvaluate - Span found: %v, IsRecording: %v", span != nil, span.IsRecording())

	if !span.IsRecording() {
		logger.Debug("Span not recording, returning")
		return
	}

	spanContext := span.SpanContext()
	traceID := spanContext.TraceID().String()
	spanID := spanContext.SpanID().String()

	logger.Info("asyncEvaluate: project=%s, traceId=%s, spanId=%s, scorer=%s",
		bt.configuration.ProjectName, traceID, spanID, scorer.GetName())

	evaluationRun := bt.createEvaluationRun(scorer, example, model, traceID, spanID)
	bt.enqueueEvaluation(evaluationRun)
}

func (bt *baseTracer) AsyncTraceEvaluate(ctx context.Context, scorer scorers.BaseScorer, model string) {
	if !bt.configuration.EnableEvaluation {
		return
	}

	span := trace.SpanFromContext(ctx)
	if !span.IsRecording() {
		return
	}

	spanContext := span.SpanContext()
	traceID := spanContext.TraceID().String()
	spanID := spanContext.SpanID().String()

	logger.Info("asyncTraceEvaluate: project=%s, traceId=%s, spanId=%s, scorer=%s",
		bt.configuration.ProjectName, traceID, spanID, scorer.GetName())

	evaluationRun := bt.createTraceEvaluationRun(scorer, model, traceID, spanID)

	traceEvalJSON := bt.serializer(evaluationRun)
	span.SetAttributes(attribute.String(AttributeKeys.PendingTraceEval, traceEvalJSON))
}

func (bt *baseTracer) SetAttributes(span trace.Span, attributes map[string]interface{}) {
	if attributes == nil || !span.IsRecording() {
		return
	}

	attrs := make([]attribute.KeyValue, 0, len(attributes))
	for key, value := range attributes {
		attrs = append(attrs, bt.createAttribute(key, value))
	}
	span.SetAttributes(attrs...)
}

func (bt *baseTracer) SetLLMSpan(span trace.Span) {
	bt.SetSpanKind(span, "llm")
}

func (bt *baseTracer) SetToolSpan(span trace.Span) {
	bt.SetSpanKind(span, "tool")
}

func (bt *baseTracer) SetGeneralSpan(span trace.Span) {
	bt.SetSpanKind(span, "span")
}

func (bt *baseTracer) SetInput(span trace.Span, input interface{}) {
	bt.SetAttribute(span, AttributeKeys.JudgmentInput, input)
}

func (bt *baseTracer) SetOutput(span trace.Span, output interface{}) {
	bt.SetAttribute(span, AttributeKeys.JudgmentOutput, output)
}

func (bt *baseTracer) GetTracer() trace.Tracer {
	return bt.tracer
}

func (bt *baseTracer) Span(ctx context.Context, spanName string) (trace.Span, context.Context) {
	spanCtx, span := bt.tracer.Start(ctx, spanName)

	spanCtx = trace.ContextWithSpan(spanCtx, span)
	return span, spanCtx
}

func (bt *baseTracer) createJudgmentSpanExporter(projectID string) sdktrace.SpanExporter {

	endpoint := bt.configuration.APIURL
	if endpoint[len(endpoint)-1] != '/' {
		endpoint += "/"
	}
	endpoint += "otel/v1/traces"

	exporter, err := exporters.NewJudgmentSpanExporterBuilder().
		Endpoint(endpoint).
		APIKey(bt.configuration.APIKey).
		OrganizationID(bt.configuration.OrganizationID).
		ProjectID(projectID).
		Build()
	if err != nil {
		logger.Error("Failed to create Judgment span exporter: %v", err)
		return exporters.NewNoOpSpanExporter()
	}
	return exporter
}

func (bt *baseTracer) createEvaluationRun(scorer scorers.BaseScorer, example *data.Example, model, traceID, spanID string) *data.ExampleEvaluationRun {
	runID := fmt.Sprintf("async_evaluate_%s", spanID)
	if spanID == "" {
		runID = fmt.Sprintf("async_evaluate_%d", time.Now().UnixMilli())
	}

	modelName := model
	if modelName == "" {
		modelName = env.JudgmentDefaultGPTModel
	}

	return data.NewExampleEvaluationRun(
		data.WithProjectName(bt.configuration.ProjectName),
		data.WithEvalName(runID),
		data.WithExamples([]*data.Example{example}),
		data.WithScorers([]models.ScorerConfig{scorer.GetScorerConfig()}),
		data.WithModel(modelName),
		data.WithOrganizationId(bt.configuration.OrganizationID),
		data.WithTraceId(traceID),
		data.WithTraceSpanId(spanID),
	)
}

func (bt *baseTracer) createTraceEvaluationRun(scorer scorers.BaseScorer, model, traceID, spanID string) *data.TraceEvaluationRun {
	evalName := fmt.Sprintf("async_trace_evaluate_%s", spanID)
	if spanID == "" {
		evalName = fmt.Sprintf("async_trace_evaluate_%d", time.Now().UnixMilli())
	}

	modelName := model
	if modelName == "" {
		modelName = env.JudgmentDefaultGPTModel
	}

	return data.NewTraceEvaluationRunWithOptions(data.TraceEvaluationRunOptions{
		ProjectName: bt.configuration.ProjectName,
		EvalName:    evalName,
		Scorer:      scorer.GetScorerConfig(),
		Model:       modelName,
		TraceId:     traceID,
		SpanId:      spanID,
	})
}

func (bt *baseTracer) enqueueEvaluation(evaluationRun *data.ExampleEvaluationRun) {
	_, err := bt.apiClient.AddToRunEvalQueue(evaluationRun.ExampleEvaluationRun)
	if err != nil {
		logger.Error("Failed to enqueue evaluation run: %v", err)
	}
}

func DefaultJSONSerializer(obj interface{}) string {
	if obj == nil {
		return "null"
	}

	jsonBytes, err := json.Marshal(obj)
	if err != nil {
		return fmt.Sprintf("%+v", obj)
	}

	return string(jsonBytes)
}
