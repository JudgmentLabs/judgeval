// Package tracer provides legacy tracing functionality.
//
// Deprecated: Use github.com/JudgmentLabs/judgeval-go/v1 instead.
// This package will be removed in a future version.
package tracer

import (
	"context"
	"fmt"

	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api"
	"github.com/JudgmentLabs/judgeval-go/pkg/internal/api/models"
	"github.com/JudgmentLabs/judgeval-go/pkg/logger"
	"github.com/JudgmentLabs/judgeval-go/pkg/version"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
)

const (
	TRACER_NAME = "judgeval"
)

type Tracer struct {
	*baseTracer
	tracerProvider *sdktrace.TracerProvider
}

type TracerOptions func(*TracerConfig)

type TracerConfig struct {
	Configuration TracerConfiguration
	Serializer    SerializerFunc
	Initialize    bool
}

func WithConfiguration(config TracerConfiguration) TracerOptions {
	return func(tc *TracerConfig) {
		tc.Configuration = config
	}
}

func WithSerializer(serializer SerializerFunc) TracerOptions {
	return func(tc *TracerConfig) {
		tc.Serializer = serializer
	}
}

func WithInitialize(initialize bool) TracerOptions {
	return func(tc *TracerConfig) {
		tc.Initialize = initialize
	}
}

func NewTracer(options ...TracerOptions) (*Tracer, error) {
	config := &TracerConfig{
		Configuration: NewTracerConfiguration(),
		Serializer:    DefaultJSONSerializer,
		Initialize:    true,
	}

	for _, option := range options {
		option(config)
	}

	return newTracerWithConfig(config)
}

func newTracerWithConfig(config *TracerConfig) (*Tracer, error) {
	if config.Configuration.APIURL == "" {
		return nil, fmt.Errorf("configuration 'APIURL' is required")
	}
	if config.Configuration.APIKey == "" {
		return nil, fmt.Errorf("configuration 'APIKey' is required")
	}
	if config.Configuration.OrganizationID == "" {
		return nil, fmt.Errorf("configuration 'OrganizationID' is required")
	}
	if config.Configuration.ProjectName == "" {
		return nil, fmt.Errorf("configuration 'ProjectName' is required")
	}
	if config.Serializer == nil {
		return nil, fmt.Errorf("serializer cannot be nil")
	}

	apiClient := api.NewClient(config.Configuration.APIURL, config.Configuration.APIKey, config.Configuration.OrganizationID)
	projectID := resolveProjectID(apiClient, config.Configuration.ProjectName)

	if projectID == "" {
		logger.Error("Failed to resolve project %s, please create it first at https://app.judgmentlabs.ai/org/%s/projects. Skipping Judgment export.",
			config.Configuration.ProjectName, config.Configuration.OrganizationID)
	}

	tracer := otel.Tracer(TracerName)

	baseTracer := &baseTracer{
		configuration: config.Configuration,
		apiClient:     apiClient,
		serializer:    config.Serializer,
		projectID:     projectID,
		tracer:        tracer,
	}

	tracerInstance := &Tracer{
		baseTracer: baseTracer,
	}

	if config.Initialize {
		tracerInstance.Initialize()
	}

	return tracerInstance, nil
}

func (t *Tracer) Initialize() {
	spanExporter := t.GetSpanExporter()

	res, err := resource.New(context.Background(),
		resource.WithAttributes(
			semconv.ServiceName(t.configuration.ProjectName),
			semconv.TelemetrySDKName(TRACER_NAME),
			semconv.TelemetrySDKVersion(version.Version),
		),
		resource.WithFromEnv(),
	)
	if err != nil {
		logger.Error("Failed to create resource: %v", err)
		return
	}

	t.tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithResource(res),
		sdktrace.WithSpanProcessor(sdktrace.NewBatchSpanProcessor(spanExporter)),
	)

	otel.SetTracerProvider(t.tracerProvider)
}

func (t *Tracer) Shutdown(ctx context.Context) error {
	if t.tracerProvider != nil {
		return t.tracerProvider.Shutdown(ctx)
	}
	return nil
}

func (t *Tracer) Flush(ctx context.Context) error {
	if t.tracerProvider != nil {
		return t.tracerProvider.ForceFlush(ctx)
	}
	return nil
}

func (t *Tracer) GetTracerProvider() *sdktrace.TracerProvider {
	return t.tracerProvider
}

func resolveProjectID(apiClient *api.Client, projectName string) string {
	request := &models.ResolveProjectNameRequest{
		ProjectName: projectName,
	}

	response, err := apiClient.ProjectsResolve(request)
	if err != nil {
		return ""
	}

	if response.ProjectId != "" {
		return response.ProjectId
	}
	return ""
}
