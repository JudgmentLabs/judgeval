import { ExportResult } from "@opentelemetry/core";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { ReadableSpan, SpanExporter } from "@opentelemetry/sdk-trace-base";
import { Logger } from "../../utils/logger";

/**
 * SpanExporter implementation that sends spans to Judgment Labs with project identification.
 *
 * This exporter wraps the OTLP HTTP exporter and adds Judgment Labs specific headers and project
 * identification to all exported spans.
 */
export class JudgmentSpanExporter implements SpanExporter {
  private readonly delegate: SpanExporter;
  private readonly projectId: string;

  /**
   * Creates a new JudgmentSpanExporter with the specified configuration.
   *
   * @param endpoint the OTLP endpoint URL
   * @param apiKey the API key for authentication
   * @param organizationId the organization ID
   * @param projectId the project ID (must not be null or empty)
   * @throws Error if projectId is null or empty
   */
  constructor(
    endpoint: string,
    apiKey: string,
    organizationId: string,
    projectId: string,
  ) {
    if (!projectId || projectId.trim() === "") {
      throw new Error("projectId is required for JudgmentSpanExporter");
    }

    this.projectId = projectId;
    this.delegate = new OTLPTraceExporter({
      url: endpoint,
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "X-Organization-Id": organizationId,
        "X-Project-Id": projectId,
      },
    });
  }

  /**
   * Creates a new builder for constructing JudgmentSpanExporter instances.
   */
  public static builder(): JudgmentSpanExporterBuilder {
    return new JudgmentSpanExporterBuilder();
  }

  /**
   * Exports a collection of spans.
   */
  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ): void {
    Logger.info(`JudgmentSpanExporter: Exported ${spans.length} spans`);
    this.delegate.export(spans, resultCallback);
  }

  /**
   * Shuts down the exporter.
   */
  shutdown(): Promise<void> {
    Logger.info(
      `JudgmentSpanExporter: Shutting down exporter for project ${this.projectId}`,
    );
    return this.delegate.shutdown();
  }

  /**
   * Forces the exporter to flush any pending spans.
   */
  forceFlush(): Promise<void> {
    Logger.info(
      `JudgmentSpanExporter: Force flushing spans for project ${this.projectId}`,
    );
    return this.delegate.forceFlush?.() ?? Promise.resolve();
  }
}

/**
 * Builder for creating JudgmentSpanExporter instances.
 */
export class JudgmentSpanExporterBuilder {
  private _endpoint?: string;
  private _apiKey?: string;
  private _organizationId?: string;
  private _projectId?: string;

  constructor() {}

  /**
   * Sets the OTLP endpoint URL.
   *
   * @param endpoint the endpoint URL
   * @return this builder for method chaining
   */
  public endpoint(endpoint: string): this {
    this._endpoint = endpoint;
    return this;
  }

  /**
   * Sets the API key for authentication.
   *
   * @param apiKey the API key
   * @return this builder for method chaining
   */
  public apiKey(apiKey: string): this {
    this._apiKey = apiKey;
    return this;
  }

  /**
   * Sets the organization ID.
   *
   * @param organizationId the organization ID
   * @return this builder for method chaining
   */
  public organizationId(organizationId: string): this {
    this._organizationId = organizationId;
    return this;
  }

  /**
   * Sets the project ID.
   *
   * @param projectId the project ID
   * @return this builder for method chaining
   */
  public projectId(projectId: string): this {
    this._projectId = projectId;
    return this;
  }

  /**
   * Builds a new JudgmentSpanExporter with the current configuration.
   *
   * @return a new JudgmentSpanExporter instance
   * @throws Error if required fields are missing
   */
  public build(): JudgmentSpanExporter {
    if (!this._endpoint || this._endpoint.trim() === "") {
      throw new Error("Endpoint is required");
    }
    if (!this._apiKey || this._apiKey.trim() === "") {
      throw new Error("API key is required");
    }
    if (!this._organizationId || this._organizationId.trim() === "") {
      throw new Error("Organization ID is required");
    }
    if (!this._projectId || this._projectId.trim() === "") {
      throw new Error("Project ID is required");
    }

    return new JudgmentSpanExporter(
      this._endpoint,
      this._apiKey,
      this._organizationId,
      this._projectId,
    );
  }
}
