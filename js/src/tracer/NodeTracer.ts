import { resourceFromAttributes } from "@opentelemetry/resources";
import { NodeSDK, NodeSDKConfiguration } from "@opentelemetry/sdk-node";
import { Logger } from "../utils/logger";
import { VERSION } from "../version";
import { OpenTelemetryKeys } from "./OpenTelemetryKeys";
import { Tracer, TracerInitializeOptions } from "./Tracer";
import { TracerConfiguration } from "./TracerConfiguration";

export type NodeTracerInitializeOptions = TracerInitializeOptions &
  Partial<NodeSDKConfiguration>;

export class NodeTracer extends Tracer {
  private nodeSDK?: NodeSDK;

  public async initialize(
    options: NodeTracerInitializeOptions = {},
  ): Promise<NodeTracer> {
    if (this._initialized) {
      return this;
    }

    try {
      const resourceAttributes = {
        [OpenTelemetryKeys.ResourceKeys.SERVICE_NAME]:
          this.configuration.projectName,
        [OpenTelemetryKeys.ResourceKeys.TELEMETRY_SDK_VERSION]: VERSION,
        ...this.configuration.resourceAttributes,
        ...options.resourceAttributes,
      };

      const spanExporter = await this.getSpanExporter();

      this.nodeSDK = new NodeSDK({
        resource: resourceFromAttributes(resourceAttributes),
        instrumentations: options.instrumentations,
        traceExporter: spanExporter,
        ...options,
      });

      this.nodeSDK.start();

      this._initialized = true;
      return this;
    } catch (error) {
      throw new Error(
        `Failed to initialize node tracer: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  public static getInstance(configuration: TracerConfiguration): NodeTracer {
    const key = `NodeTracer:${configuration.projectName}`;
    if (!Tracer.instances.has(key)) {
      Tracer.instances.set(key, new NodeTracer(configuration));
    }
    return Tracer.instances.get(key) as NodeTracer;
  }

  public static async createDefault(projectName: string): Promise<NodeTracer> {
    const configuration = TracerConfiguration.builder()
      .projectName(projectName)
      .enableEvaluation(true)
      .build();
    const tracer = new NodeTracer(configuration);
    if (configuration.initialize) {
      await tracer.initialize();
    }
    return tracer;
  }

  public static async createWithConfiguration(
    configuration: TracerConfiguration,
  ): Promise<NodeTracer> {
    const tracer = new NodeTracer(configuration);
    if (configuration.initialize) {
      await tracer.initialize();
    }
    return tracer;
  }

  public async shutdown(): Promise<void> {
    if (!this.nodeSDK) {
      Logger.warn("Node SDK not initialized, skipping shutdown");
      return;
    }
    await this.nodeSDK.shutdown();
  }
}
