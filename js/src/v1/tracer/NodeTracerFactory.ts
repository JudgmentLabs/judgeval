import { JudgmentApiClient } from "../../internal/api";
import { NodeTracer, type NodeTracerConfig } from "./NodeTracer";

export class NodeTracerFactory {
  private readonly client: JudgmentApiClient;

  constructor(client: JudgmentApiClient) {
    this.client = client;
  }

  async create(config: NodeTracerConfig): Promise<NodeTracer> {
    if (!config.projectName) {
      throw new Error("Project name is required");
    }

    return NodeTracer.create(
      {
        projectName: config.projectName,
        enableEvaluation: config.enableEvaluation ?? true,
        enableMonitoring: config.enableMonitoring ?? false,
        serializer: config.serializer ?? JSON.stringify,
        resourceAttributes: config.resourceAttributes ?? {},
        instrumentations: config.instrumentations ?? [],
        initialize: config.initialize ?? true,
      },
      this.client,
    );
  }
}
