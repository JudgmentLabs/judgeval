import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  BatchSpanProcessor,
  WebTracerProvider,
} from "@opentelemetry/sdk-trace-web";
import { VERSION } from "../version";
import { OpenTelemetryKeys } from "./OpenTelemetryKeys";
import { Tracer, TracerInitializeOptions } from "./Tracer";
import { TracerConfiguration } from "./TracerConfiguration";

export type BrowserTracerInitializeOptions = TracerInitializeOptions & {};

export class BrowserTracer extends Tracer {
  private webTracerProvider?: WebTracerProvider;

  public async initialize(
    options: BrowserTracerInitializeOptions = {},
  ): Promise<BrowserTracer> {
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

      this.webTracerProvider = new WebTracerProvider({
        resource: resourceFromAttributes(resourceAttributes),
        spanProcessors: [new BatchSpanProcessor(spanExporter)],
        ...options,
      });

      this.webTracerProvider.register();

      this._initialized = true;
      return this;
    } catch (error) {
      throw new Error(
        `Failed to initialize browser tracer: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  public static getInstance(configuration: TracerConfiguration): BrowserTracer {
    const key = `BrowserTracer:${configuration.projectName}`;
    if (!Tracer.instances.has(key)) {
      Tracer.instances.set(key, new BrowserTracer(configuration));
    }
    return Tracer.instances.get(key) as BrowserTracer;
  }

  public static async createDefault(
    projectName: string,
  ): Promise<BrowserTracer> {
    const configuration = TracerConfiguration.builder()
      .projectName(projectName)
      .build();
    const tracer = new BrowserTracer(configuration);
    if (configuration.initialize) {
      await tracer.initialize();
    }
    return tracer;
  }

  public static async createWithConfiguration(
    configuration: TracerConfiguration,
  ): Promise<BrowserTracer> {
    const tracer = new BrowserTracer(configuration);
    if (configuration.initialize) {
      await tracer.initialize();
    }
    return tracer;
  }
}
