package exporters

import (
	"context"

	"go.opentelemetry.io/otel/sdk/trace"
)

type NoOpSpanExporter struct{}

func NewNoOpSpanExporter() *NoOpSpanExporter {
	return &NoOpSpanExporter{}
}

func (n *NoOpSpanExporter) ExportSpans(ctx context.Context, spans []trace.ReadOnlySpan) error {
	return nil
}

func (n *NoOpSpanExporter) Shutdown(ctx context.Context) error {
	return nil
}

func (n *NoOpSpanExporter) ForceFlush(ctx context.Context) error {
	return nil
}
