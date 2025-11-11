package v1

import (
	"context"
	"time"

	"github.com/JudgmentLabs/judgeval-go/pkg/logger"
	"github.com/JudgmentLabs/judgeval-go/v1/internal/api"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type judgmentSpanExporter struct {
	delegate sdktrace.SpanExporter
}

func newJudgmentSpanExporter(ctx context.Context, endpoint string, apiClient *api.Client, projectID string) sdktrace.SpanExporter {
	if projectID == "" {
		logger.Error("projectID is required for JudgmentSpanExporter")
		return newNoOpSpanExporter()
	}

	headers := map[string]string{
		"Authorization":     "Bearer " + apiClient.GetAPIKey(),
		"X-Organization-Id": apiClient.GetOrganizationID(),
		"X-Project-Id":      projectID,
	}

	exporter, err := otlptracehttp.New(ctx,
		otlptracehttp.WithEndpointURL(endpoint),
		otlptracehttp.WithHeaders(headers),
		otlptracehttp.WithTimeout(30*time.Second),
	)
	if err != nil {
		logger.Error("Failed to create OTLP HTTP exporter: %v", err)
		return newNoOpSpanExporter()
	}

	return &judgmentSpanExporter{
		delegate: exporter,
	}
}

func (e *judgmentSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	logger.Info("Exported %d spans", len(spans))
	return e.delegate.ExportSpans(ctx, spans)
}

func (e *judgmentSpanExporter) Shutdown(ctx context.Context) error {
	return e.delegate.Shutdown(ctx)
}

type noOpSpanExporter struct{}

func newNoOpSpanExporter() sdktrace.SpanExporter {
	return &noOpSpanExporter{}
}

func (e *noOpSpanExporter) ExportSpans(ctx context.Context, spans []sdktrace.ReadOnlySpan) error {
	logger.Warning("NoOpSpanExporter: discarding %d spans", len(spans))
	return nil
}

func (e *noOpSpanExporter) Shutdown(ctx context.Context) error {
	return nil
}
