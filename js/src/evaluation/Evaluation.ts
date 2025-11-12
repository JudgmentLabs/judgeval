import { JudgmentApiClient } from "../internal/api";

export interface EvaluationConfig {
  client: JudgmentApiClient;
}

export class Evaluation {
  private readonly client: JudgmentApiClient;

  constructor(config: EvaluationConfig) {
    this.client = config.client;
  }
}
