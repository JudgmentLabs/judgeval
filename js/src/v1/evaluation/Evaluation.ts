import { JudgmentApiClient } from "../../internal/api";

export interface EvaluationConfig {
  client: JudgmentApiClient;
}

export class Evaluation {
  private readonly client: JudgmentApiClient;

  constructor(config: EvaluationConfig) {
    if (!config.client) {
      throw new Error("client is required");
    }
    this.client = config.client;
  }
}
