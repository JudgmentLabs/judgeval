import type { Instrumentation } from "@opentelemetry/instrumentation";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { JudgmentApiClient } from "../internal/api";
import { Logger } from "../utils/logger";
import { VERSION } from "../version";
import { BaseTracer, type Serializer } from "./BaseTracer";

export interface NodeTracerConfig {
  projectName: string;
  enableEvaluation?: boolean;
  enableMonitoring?: boolean;
  serializer?: Serializer;
  resourceAttributes?: Record<string, unknown>;
  instrumentations?: Instrumentation[];
  initialize?: boolean;
}

interface InternalNodeTracerConfig
  extends Required<
    Omit<NodeTracerConfig, "resourceAttributes" | "instrumentations">
  > {
  resourceAttributes: Record<string, unknown>;
  instrumentations: Instrumentation[];
}

export class NodeTracer extends BaseTracer {
  private nodeSDK: NodeSDK | null = null;
  private resourceAttributes: Record<string, unknown>;
  private instrumentations: Instrumentation[];

  private constructor(
    projectName: string,
    enableEvaluation: boolean,
    apiClient: JudgmentApiClient,
    serializer: Serializer,
    resourceAttributes: Record<string, unknown>,
    instrumentations: Instrumentation[],
  ) {
    super(projectName, enableEvaluation, apiClient, serializer);
    this.resourceAttributes = resourceAttributes;
    this.instrumentations = instrumentations;
  }

  static async create(
    config: InternalNodeTracerConfig,
    apiClient: JudgmentApiClient,
  ): Promise<NodeTracer> {
    const tracer = new NodeTracer(
      config.projectName,
      config.enableEvaluation,
      apiClient,
      config.serializer,
      config.resourceAttributes,
      config.instrumentations,
    );

    await tracer.resolveAndSetProjectId();

    if (config.initialize) {
      await tracer.initialize();
    }

    return tracer;
  }

  /* eslint-disable @typescript-eslint/require-await */
  async initialize(): Promise<void> {
    if (this.nodeSDK !== null) {
      Logger.warn("NodeTracer already initialized");
      return;
    }

    try {
      const attributes = {
        "service.name": this.projectName,
        "telemetry.sdk.version": VERSION,
        ...this.resourceAttributes,
      };

      const spanExporter = this.getSpanExporter();

      this.nodeSDK = new NodeSDK({
        resource: resourceFromAttributes(attributes),
        traceExporter: spanExporter,
        instrumentations: this.instrumentations,
      });

      this.nodeSDK.start();
      Logger.info("NodeTracer initialized successfully");
    } catch (error) {
      throw new Error(
        `Failed to initialize NodeTracer: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async shutdown(): Promise<void> {
    if (!this.nodeSDK) {
      Logger.warn("NodeTracer not initialized, skipping shutdown");
      return;
    }
    try {
      await this.nodeSDK.shutdown();
      this.nodeSDK = null;
      Logger.info("NodeTracer shut down successfully");
    } catch (error) {
      Logger.error(`Failed to shutdown NodeTracer: ${error}`);
    }
  }
}
