import { type ExportResult } from "@opentelemetry/core";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { SpanExporter } from "@opentelemetry/sdk-trace-base";
import { Logger } from "../../../utils";

export class JudgmentSpanExporter implements SpanExporter {
  private delegate: SpanExporter;

  constructor(
    endpoint: string,
    apiKey: string,
    organizationId: string,
    projectId: string,
  ) {
    this.delegate = new OTLPTraceExporter({
      url: endpoint,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "X-Organization-Id": organizationId,
        "X-Project-Id": projectId,
      },
    });
  }

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    Logger.info(`JudgmentSpanExporter: exporting ${spans.length} spans`);
    this.delegate.export(spans, resultCallback);
  }

  shutdown(): Promise<void> {
    return this.delegate.shutdown();
  }

  forceFlush(): Promise<void> {
    return this.delegate.forceFlush?.() ?? Promise.resolve();
  }
}
