import { JudgmentApiClient } from "../../internal/api";
import { Evaluation, type EvaluationConfig } from "./Evaluation";

export class EvaluationFactory {
  private readonly client: JudgmentApiClient;

  constructor(client: JudgmentApiClient) {
    this.client = client;
  }

  create(config: Omit<EvaluationConfig, "client"> = {}): Evaluation {
    return new Evaluation({
      ...config,
      client: this.client,
    });
  }
}
