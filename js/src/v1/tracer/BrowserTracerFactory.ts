import { JudgmentApiClient } from "../../internal/api";
import { BrowserTracer, type BrowserTracerConfig } from "./BrowserTracer";

export class BrowserTracerFactory {
  private readonly client: JudgmentApiClient;

  constructor(client: JudgmentApiClient) {
    this.client = client;
  }

  async create(config: BrowserTracerConfig): Promise<BrowserTracer> {
    if (!config.projectName) {
      throw new Error("Project name is required");
    }

    return BrowserTracer.create(
      {
        projectName: config.projectName,
        enableEvaluation: config.enableEvaluation ?? true,
        enableMonitoring: config.enableMonitoring ?? false,
        serializer: config.serializer ?? JSON.stringify,
        resourceAttributes: config.resourceAttributes ?? {},
        initialize: config.initialize ?? true,
      },
      this.client,
    );
  }
}
