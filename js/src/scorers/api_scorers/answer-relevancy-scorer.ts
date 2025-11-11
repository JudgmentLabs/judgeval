import { ExampleParams } from "../../data";
import { RemoteScorer, RemoteScorerConfig } from "../remote-scorer";

export class AnswerRelevancyScorer extends RemoteScorer {
  private constructor(config: Partial<RemoteScorerConfig> = {}) {
    super({
      scoreType: "Answer Relevancy",
      ...config,
      requiredParams: [ExampleParams.INPUT, ExampleParams.ACTUAL_OUTPUT],
    });
  }

  static get(config: Partial<RemoteScorerConfig> = {}): AnswerRelevancyScorer {
    return new AnswerRelevancyScorer(config);
  }
}
