import { Tracer as OpenTelemetryTracer, trace } from "@opentelemetry/api";
import { JUDGMENT_DEFAULT_GPT_MODEL } from "../env";
import { JudgmentApiClient } from "../internal/api";
import {
  ExampleEvaluationRun as ExampleEvaluationRunModel,
  Example as ExampleModel,
  TraceEvaluationRun as TraceEvaluationRunModel,
} from "../internal/api/models";
import { remoteScorerToScorerConfig } from "../scorers/adapters";
import { RemoteScorer } from "../scorers/remote-scorer";
import { parseFunctionArgs } from "../utils/annotate";
import { Logger } from "../utils/logger";
import { JudgmentSpanExporter, NoOpSpanExporter } from "./exporters";
import { OpenTelemetryKeys } from "./OpenTelemetryKeys";
import { TracerConfiguration } from "./TracerConfiguration";

export type Serializer = (obj: unknown) => string;
type SpanKind = string;

export interface TracerInitializeOptions {
  resourceAttributes?: Record<string, unknown>;
  [key: string]: unknown;
}

/**
 * @deprecated Use v1 BaseTracer and Tracer instead. This class will be removed in a future version.
 * Import from 'judgeval/v1' instead.
 */
export abstract class Tracer {
  protected static instances = new Map<string, Tracer>();

  public apiClient: JudgmentApiClient;
  public tracer: OpenTelemetryTracer;
  public serializer: Serializer = JSON.stringify;

  protected _initialized = false;

  private projectId: string | null = null;
  private spanExporter: JudgmentSpanExporter | NoOpSpanExporter | null = null;
  protected configuration: TracerConfiguration;

  public getConfiguration(): TracerConfiguration {
    return this.configuration;
  }

  public getProjectId(): string | null {
    return this.projectId;
  }

  public getSerializer(): Serializer {
    return this.serializer;
  }

  protected constructor(configuration: TracerConfiguration) {
    this.configuration = configuration;
    this.tracer = trace.getTracer(this.configuration.tracerName);

    this.apiClient = new JudgmentApiClient(
      this.configuration.apiUrl,
      this.configuration.apiKey,
      this.configuration.organizationId,
    );

    this._initialized = false;
    // Initialization is handled by the subclasses, since operation requires IO.
  }

  public abstract initialize(options: TracerInitializeOptions): Promise<Tracer>;

  private async resolveProjectId(): Promise<string> {
    try {
      Logger.info(
        `Resolving project ID for project: ${this.configuration.projectName}`,
      );

      const response = await this.apiClient.projectsResolve({
        project_name: this.configuration.projectName,
      });
      Logger.info(`Resolved project ID: ${response.project_id}`);
      const resolvedProjectId = response.project_id;

      this.projectId = resolvedProjectId;
      Logger.info(`Successfully resolved project ID: ${this.projectId}`);

      return this.projectId;
    } catch (error) {
      throw new Error(
        `Failed to resolve project ID: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }
  public async getSpanExporter(): Promise<
    JudgmentSpanExporter | NoOpSpanExporter
  > {
    if (!this.spanExporter) {
      try {
        const projectId = await this.resolveProjectId();
        this.spanExporter = this.createJudgmentSpanExporter(projectId);
      } catch {
        Logger.error(
          "Failed to resolve project " +
            this.configuration.projectName +
            ", please create it first at https://app.judgmentlabs.ai/org/" +
            (this.configuration.organizationId || "unknown") +
            "/projects. Skipping Judgment export.",
        );
        this.spanExporter = new NoOpSpanExporter();
      }
    }

    return this.spanExporter;
  }

  public setSpanKind(kind: SpanKind): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      Logger.info("No active span found, skipping setSpanKind");
      return;
    }
    currentSpan.setAttribute(
      OpenTelemetryKeys.AttributeKeys.JUDGMENT_SPAN_KIND,
      kind,
    );
  }

  public setAttribute(key: string, value: unknown): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      Logger.info("No active span found, skipping setAttribute");
      return;
    }
    currentSpan.setAttribute(key, this.serializer(value));
  }

  public setLLMSpan(): void {
    this.setSpanKind("llm");
  }

  public setToolSpan(): void {
    this.setSpanKind("tool");
  }

  public setGeneralSpan(): void {
    this.setSpanKind("span");
  }

  public setAttributes(attributes: Record<string, unknown>): void {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      Logger.info("No active span found, skipping setAttributes");
      return;
    }
    for (const [key, value] of Object.entries(attributes)) {
      currentSpan.setAttribute(key, this.serializer(value));
    }
  }

  public setInput(input: unknown): void {
    this.setAttribute(OpenTelemetryKeys.AttributeKeys.JUDGMENT_INPUT, input);
  }

  public setOutput(output: unknown): void {
    this.setAttribute(OpenTelemetryKeys.AttributeKeys.JUDGMENT_OUTPUT, output);
  }

  public asyncEvaluate(
    scorer: RemoteScorer,
    example: ExampleModel,
    model?: string,
  ): void {
    try {
      if (!this.configuration.enableEvaluation) {
        return;
      }

      const currentSpan = trace.getActiveSpan();
      if (!currentSpan) {
        Logger.info("No active span found, skipping asyncEvaluate");
        return;
      }
      if (!currentSpan.isRecording()) {
        Logger.info("Active span is not recording, skipping asyncEvaluate");
        return;
      }

      const spanContext = currentSpan.spanContext();
      const traceId = spanContext.traceId;
      const spanId = spanContext.spanId;

      Logger.info(
        `asyncEvaluate: project=${this.configuration.projectName}, traceId=${traceId}, spanId=${spanId}, scorer=${scorer.name}`,
      );

      const evaluationRun = this.createEvaluationRun(
        scorer,
        example,
        model,
        traceId,
        spanId,
      );
      void this.enqueueEvaluation(evaluationRun);
    } catch (error) {
      Logger.error(
        `Failed to asyncEvaluate: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  public asyncTraceEvaluate(scorer: RemoteScorer, model?: string): void {
    try {
      if (!this.configuration.enableEvaluation) {
        return;
      }

      const currentSpan = trace.getActiveSpan();
      if (!currentSpan) {
        Logger.info("No active span found, skipping asyncTraceEvaluate");
        return;
      }
      if (!currentSpan.isRecording()) {
        Logger.info(
          "Active span is not recording, skipping asyncTraceEvaluate",
        );
        return;
      }

      const spanContext = currentSpan.spanContext();
      const traceId = spanContext.traceId;
      const spanId = spanContext.spanId;

      Logger.info(
        `asyncTraceEvaluate: project=${this.configuration.projectName}, traceId=${traceId}, spanId=${spanId}, scorer=${scorer.name}`,
      );

      const traceEvaluationRun = this.createTraceEvaluationRun(
        scorer,
        model,
        traceId,
        spanId,
      );
      const traceEvalJson = this.serializer(traceEvaluationRun);
      currentSpan.setAttribute(
        OpenTelemetryKeys.AttributeKeys.PENDING_TRACE_EVAL,
        traceEvalJson,
      );
    } catch (error) {
      Logger.error(
        `Failed to asyncTraceEvaluate: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private createTraceEvaluationRun(
    scorer: RemoteScorer,
    model: string | undefined,
    traceId: string,
    spanId: string,
  ): TraceEvaluationRunModel {
    const evalName = `async_trace_evaluate_${spanId || Date.now()}`;
    const modelName = model ?? JUDGMENT_DEFAULT_GPT_MODEL;

    const scorerConfig = remoteScorerToScorerConfig(scorer);

    return {
      project_name: this.configuration.projectName,
      eval_name: evalName,
      judgment_scorers: [scorerConfig],
      custom_scorers: [],
      model: modelName,
      trace_and_span_ids: [[traceId, spanId]],
      created_at: new Date().toISOString(),
      is_offline: false,
      is_bucket_run: false,
    };
  }

  private createJudgmentSpanExporter(projectId: string): JudgmentSpanExporter {
    const endpoint = this.configuration.apiUrl.endsWith("/")
      ? `${this.configuration.apiUrl}otel/v1/traces`
      : `${this.configuration.apiUrl}/otel/v1/traces`;

    return JudgmentSpanExporter.builder()
      .endpoint(endpoint)
      .apiKey(this.configuration.apiKey)
      .organizationId(this.configuration.organizationId)
      .projectId(projectId)
      .build();
  }

  private createEvaluationRun(
    scorer: RemoteScorer,
    example: ExampleModel,
    model: string | undefined,
    traceId: string,
    spanId: string,
  ): ExampleEvaluationRunModel {
    const runId = `async_evaluate_${spanId || Date.now()}`;
    const modelName = model ?? JUDGMENT_DEFAULT_GPT_MODEL;

    const scorerConfig = remoteScorerToScorerConfig(scorer);

    const evaluationRun: ExampleEvaluationRunModel = {
      project_name: this.configuration.projectName,
      eval_name: runId,
      examples: [example],
      custom_scorers: [],
      judgment_scorers: [scorerConfig],
      model: modelName,
      trace_id: traceId,
      trace_span_id: spanId,
      created_at: new Date().toISOString(),
    };

    return evaluationRun;
  }

  private async enqueueEvaluation(
    evaluationRun: ExampleEvaluationRunModel,
  ): Promise<void> {
    try {
      await this.apiClient.addToRunEvalQueueExamples(evaluationRun);
      Logger.info(`Enqueuing evaluation run: ${evaluationRun.eval_name}`);
    } catch (error) {
      Logger.error(
        `Failed to enqueue evaluation run: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private _observe<TArgs extends unknown[], TResult>(
    func: (...args: TArgs) => TResult,
    spanKind: SpanKind = "span",
  ): (...args: TArgs) => TResult {
    return (...args: TArgs): TResult => {
      const spanName = func.name || "anonymous";
      return this.tracer.startActiveSpan(spanName, (span) => {
        try {
          span.setAttribute(
            OpenTelemetryKeys.AttributeKeys.JUDGMENT_SPAN_KIND,
            spanKind,
          );

          try {
            const argNames = parseFunctionArgs(func);
            if (argNames.length === args.length) {
              const inputObj: Record<string, unknown> = {};
              argNames.forEach((name, index) => {
                inputObj[name] = args[index];
              });
              span.setAttribute(
                OpenTelemetryKeys.AttributeKeys.JUDGMENT_INPUT,
                this.serializer(inputObj),
              );
            }
          } catch (error) {
            Logger.warn(
              `Failed to parse function args: ${error instanceof Error ? error.message : String(error)}`,
            );
          }

          const result = func(...args);

          if (result instanceof Promise) {
            return result
              .then((res: TResult) => {
                span.setAttribute(
                  OpenTelemetryKeys.AttributeKeys.JUDGMENT_OUTPUT,
                  this.serializer(res),
                );
                return res;
              })
              .catch((err: unknown) => {
                span.recordException(
                  err instanceof Error ? err : new Error(String(err)),
                );
                throw err;
              })
              .finally(() => {
                span.end();
              }) as TResult;
          }

          span.setAttribute(
            OpenTelemetryKeys.AttributeKeys.JUDGMENT_OUTPUT,
            this.serializer(result),
          );
          span.end();
          return result;
        } catch (err) {
          span.recordException(err as Error);
          span.end();
          throw err;
        }
      });
    };
  }

  public observe(
    spanKind?: SpanKind,
  ): (
    target: unknown,
    propertyKey: string | symbol,
    descriptor?: PropertyDescriptor,
  ) => void;
  public observe<TArgs extends unknown[], TResult>(
    func: (...args: TArgs) => TResult,
    spanKind?: SpanKind,
  ): (...args: TArgs) => TResult;
  public observe<TArgs extends unknown[], TResult>(
    funcOrSpanKind?: ((...args: TArgs) => TResult) | SpanKind,
    spanKind?: SpanKind,
  ): unknown {
    try {
      if (typeof funcOrSpanKind === "function") {
        const wrapped = this._observe(funcOrSpanKind, spanKind ?? "span");
        Object.defineProperty(wrapped, "name", { value: funcOrSpanKind.name });
        return wrapped;
      }
      return (
        _target: unknown,
        _propertyKey: string | symbol,
        descriptor?: PropertyDescriptor,
      ) => {
        try {
          if (!descriptor) return;
          const originalMethod = descriptor.value as (
            ...args: unknown[]
          ) => unknown;
          const wrapped = this._observe(
            originalMethod,
            funcOrSpanKind ?? "span",
          );
          Object.defineProperty(wrapped, "name", {
            value: originalMethod.name,
          });
          descriptor.value = wrapped;
          return descriptor;
        } catch (error) {
          Logger.error(
            `Failed to wrap method with observe: ${error instanceof Error ? error.message : String(error)}`,
          );
        }
      };
    } catch (error) {
      Logger.error(
        `Failed to observe function: ${error instanceof Error ? error.message : String(error)}`,
      );
      if (typeof funcOrSpanKind === "function") {
        return funcOrSpanKind;
      }
      return () => {
        /* empty */
      };
    }
  }

  public async shutdown(): Promise<void> {
    /* empty */
  }
}
