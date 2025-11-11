import {
  type Span,
  type SpanContext,
  SpanStatusCode,
  trace,
  type Tracer,
} from "@opentelemetry/api";
import type { SpanExporter } from "@opentelemetry/sdk-trace-base";
import { JudgmentApiClient } from "../../internal/api";
import type {
  ExampleEvaluationRun,
  ResolveProjectNameRequest,
  ResolveProjectNameResponse,
  TraceEvaluationRun,
} from "../../internal/api/models";
import { AttributeKeys } from "../../judgmentAttributeKeys";
import { Logger } from "../../utils/logger";
import { Example } from "../data/Example";
import { BaseScorer } from "../scorers/BaseScorer";
import { JudgmentSpanExporter, NoOpSpanExporter } from "./exporters";

export type Serializer = (obj: unknown) => string;

export abstract class BaseTracer {
  static readonly TRACER_NAME = "judgeval";

  protected projectName: string;
  protected enableEvaluation: boolean;
  protected apiClient: JudgmentApiClient;
  protected serializer: Serializer;
  protected jsonEncoder: (obj: unknown) => string;
  protected projectId: string | null;

  protected constructor(
    projectName: string,
    enableEvaluation: boolean,
    apiClient: JudgmentApiClient,
    serializer: Serializer,
    jsonEncoder: (obj: unknown) => string = JSON.stringify,
  ) {
    this.projectName = projectName;
    this.enableEvaluation = enableEvaluation;
    this.apiClient = apiClient;
    this.serializer = serializer;
    this.jsonEncoder = jsonEncoder;
    this.projectId = null;
  }

  protected async resolveAndSetProjectId(): Promise<void> {
    try {
      this.projectId = await this.resolveProjectId(this.projectName);
      Logger.info(`Successfully resolved project ID: ${this.projectId}`);
    } catch {
      Logger.error(
        `Failed to resolve project ${this.projectName}, ` +
          `please create it first at https://app.judgmentlabs.ai/org/${this.apiClient.getOrganizationId()}/projects. ` +
          "Skipping Judgment export.",
      );
      this.projectId = null;
    }
  }

  abstract initialize(): Promise<void>;
  abstract shutdown(): Promise<void>;

  getSpanExporter(): SpanExporter {
    if (this.projectId !== null) {
      return this.createJudgmentSpanExporter(this.projectId);
    }
    Logger.error(
      "Project not resolved; cannot create exporter, returning NoOpSpanExporter",
    );
    return new NoOpSpanExporter();
  }

  getTracer(): Tracer {
    return trace.getTracer(BaseTracer.TRACER_NAME);
  }

  setSpanKind(kind: string): void {
    if (!kind) {
      return;
    }
    const currentSpan = trace.getActiveSpan();
    if (currentSpan) {
      currentSpan.setAttribute(AttributeKeys.JUDGMENT_SPAN_KIND, kind);
    }
  }

  setAttribute(key: string, value: unknown): void {
    if (!this.isValidKey(key)) {
      return;
    }
    if (value === null || value === undefined) {
      return;
    }
    const currentSpan = trace.getActiveSpan();
    if (currentSpan) {
      const serializedValue =
        typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean"
          ? value
          : this.serializer(value);
      currentSpan.setAttribute(key, serializedValue);
    }
  }

  setAttributes(attributes: Record<string, unknown>): void {
    for (const [key, value] of Object.entries(attributes)) {
      this.setAttribute(key, value);
    }
  }

  setLLMSpan(): void {
    this.setSpanKind("llm");
  }

  setToolSpan(): void {
    this.setSpanKind("tool");
  }

  setGeneralSpan(): void {
    this.setSpanKind("span");
  }

  setInput(inputData: unknown): void {
    this.setAttribute(AttributeKeys.JUDGMENT_INPUT, inputData);
  }

  setOutput(outputData: unknown): void {
    this.setAttribute(AttributeKeys.JUDGMENT_OUTPUT, outputData);
  }

  asyncEvaluate(scorer: BaseScorer, example: Example): void {
    this.safeExecute("evaluate scorer", () => {
      if (!this.enableEvaluation) {
        return;
      }

      const spanContext = this.getSampledSpanContext();
      if (!spanContext) {
        return;
      }

      const traceId = spanContext.traceId;
      const spanId = spanContext.spanId;

      this.logEvaluationInfo(
        "asyncEvaluate",
        traceId,
        spanId,
        scorer.getName(),
      );

      const evaluationRun = this.createEvaluationRun(
        scorer,
        example,
        traceId,
        spanId,
      );

      this.enqueueEvaluation(evaluationRun).catch((e: unknown) => {
        Logger.error(`Failed to enqueue evaluation run: ${e}`);
      });
    });
  }

  asyncTraceEvaluate(scorer: BaseScorer): void {
    this.safeExecute("evaluate trace scorer", () => {
      if (!this.enableEvaluation) {
        return;
      }

      const currentSpan = this.getSampledSpan();
      if (!currentSpan) {
        return;
      }

      const spanContext = currentSpan.spanContext();
      const traceId = spanContext.traceId;
      const spanId = spanContext.spanId;

      this.logEvaluationInfo(
        "asyncTraceEvaluate",
        traceId,
        spanId,
        scorer.getName(),
      );

      const evaluationRun = this.createTraceEvaluationRun(
        scorer,
        traceId,
        spanId,
      );
      try {
        const traceEvalJson = JSON.stringify(evaluationRun);
        currentSpan.setAttribute(
          AttributeKeys.JUDGMENT_PENDING_TRACE_EVAL,
          traceEvalJson,
        );
      } catch (e) {
        Logger.error(`Failed to serialize trace evaluation: ${e}`);
      }
    });
  }

  span<T>(spanName: string, callableFunc: () => T): T {
    const tracer = this.getTracer();
    return tracer.startActiveSpan(spanName, (span) => {
      try {
        return callableFunc();
      } catch (e) {
        span.setStatus({ code: SpanStatusCode.ERROR });
        span.recordException(e as Error);
        throw e;
      } finally {
        span.end();
      }
    });
  }

  static startSpan(spanName: string): Span {
    const tracer = trace.getTracer(BaseTracer.TRACER_NAME);
    return tracer.startSpan(spanName);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  observe<TArgs extends any[], TResult>(
    func: (...args: TArgs) => TResult,
    spanType = "span",
    spanName?: string | null,
  ): (...args: TArgs) => TResult {
    const tracer = this.getTracer();
    const name = spanName ?? func.name;

    return (...args: TArgs): TResult => {
      return tracer.startActiveSpan(name, (span) => {
        if (spanType) {
          span.setAttribute(AttributeKeys.JUDGMENT_SPAN_KIND, spanType);
        }

        try {
          const inputData = this.formatInputs(
            func as (...args: unknown[]) => unknown,
            args as unknown[],
          );
          span.setAttribute(
            AttributeKeys.JUDGMENT_INPUT,
            this.serializer(inputData),
          );

          const result = func(...args);

          if (result instanceof Promise) {
            return result
              .then((res: TResult) => {
                span.setAttribute(
                  AttributeKeys.JUDGMENT_OUTPUT,
                  this.serializer(res),
                );
                return res;
              })
              .catch((err: unknown) => {
                span.recordException(err as Error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: String(err),
                });
                throw err;
              })
              .finally(() => {
                span.end();
              }) as TResult;
          }

          span.setAttribute(
            AttributeKeys.JUDGMENT_OUTPUT,
            this.serializer(result),
          );
          span.end();
          return result;
        } catch (e) {
          span.recordException(e as Error);
          span.setStatus({ code: SpanStatusCode.ERROR, message: String(e) });
          span.end();
          throw e;
        }
      });
    };
  }

  private async resolveProjectId(name: string): Promise<string> {
    try {
      Logger.info(`Resolving project ID for project: ${name}`);
      const request: ResolveProjectNameRequest = { project_name: name };
      const response: ResolveProjectNameResponse =
        await this.apiClient.projectsResolve(request);
      const projectId = response.project_id;
      if (!projectId) {
        throw new Error(`Project ID not found for project: ${name}`);
      }
      Logger.info(`Resolved project ID: ${projectId}`);
      return projectId;
    } catch (error) {
      throw new Error(
        `Failed to resolve project ID: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  private buildEndpoint(baseUrl: string): string {
    return baseUrl.endsWith("/")
      ? baseUrl + "otel/v1/traces"
      : baseUrl + "/otel/v1/traces";
  }

  private createJudgmentSpanExporter(projectId: string): SpanExporter {
    return new JudgmentSpanExporter(
      this.buildEndpoint(this.apiClient.getBaseUrl()),
      this.apiClient.getApiKey(),
      this.apiClient.getOrganizationId(),
      projectId,
    );
  }

  private generateRunId(prefix: string, spanId?: string | null): string {
    return prefix + (spanId ?? Date.now().toString());
  }

  private createEvaluationRun(
    scorer: BaseScorer,
    example: Example,
    traceId: string,
    spanId: string,
  ): ExampleEvaluationRun {
    const runId = this.generateRunId("async_evaluate_", spanId);

    return {
      project_name: this.projectName,
      eval_name: runId,
      trace_id: traceId,
      trace_span_id: spanId,
      examples: [example.toModel()],
      judgment_scorers: [scorer.getScorerConfig()],
      custom_scorers: [],
    };
  }

  private createTraceEvaluationRun(
    scorer: BaseScorer,
    traceId: string,
    spanId: string,
  ): TraceEvaluationRun {
    const evalName = this.generateRunId("async_trace_evaluate_", spanId);

    return {
      project_name: this.projectName,
      eval_name: evalName,
      trace_and_span_ids: [[traceId, spanId]],
      judgment_scorers: [scorer.getScorerConfig()],
      custom_scorers: [],
      is_offline: false,
    };
  }

  private async enqueueEvaluation(
    evaluationRun: ExampleEvaluationRun,
  ): Promise<void> {
    try {
      await this.apiClient.addToRunEvalQueueExamples(evaluationRun);
    } catch (e) {
      Logger.error(`Failed to enqueue evaluation run: ${e}`);
    }
  }

  private getSampledSpanContext(): SpanContext | null {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      return null;
    }
    const spanContext = currentSpan.spanContext();
    if (!spanContext.traceFlags || !(spanContext.traceFlags & 0x01)) {
      return null;
    }
    return spanContext;
  }

  private getSampledSpan(): Span | null {
    const currentSpan = trace.getActiveSpan();
    if (!currentSpan) {
      return null;
    }
    const spanContext = currentSpan.spanContext();
    if (!spanContext.traceFlags || !(spanContext.traceFlags & 0x01)) {
      return null;
    }
    return currentSpan;
  }

  private logEvaluationInfo(
    method: string,
    traceId: string,
    spanId: string,
    scorerName: string,
  ): void {
    Logger.info(
      `${method}: project=${this.projectName}, traceId=${traceId}, spanId=${spanId}, scorer=${scorerName}`,
    );
  }

  private safeExecute(operation: string, action: () => void): void {
    try {
      action();
    } catch (e) {
      Logger.error(`Failed to ${operation}: ${e}`);
    }
  }

  private isValidKey(key: string): boolean {
    return key.length > 0;
  }

  private formatInputs(
    f: (...args: unknown[]) => unknown,
    args: unknown[],
  ): Record<string, unknown> {
    try {
      const funcStr = f.toString();
      const match = /\(([^)]*)\)/.exec(funcStr);
      const paramNames = match
        ? match[1]
            .split(",")
            .map((param) => param.trim().split("=")[0].trim())
            .filter((param) => param.length > 0)
        : [];

      const inputs: Record<string, unknown> = {};
      paramNames.forEach((name, index) => {
        if (index < args.length) {
          inputs[name] = args[index];
        }
      });
      return inputs;
    } catch {
      return {};
    }
  }
}
