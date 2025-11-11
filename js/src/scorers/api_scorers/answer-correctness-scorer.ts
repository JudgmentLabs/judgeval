import { ExampleParams } from "../../data";
import { RemoteScorer, RemoteScorerConfig } from "../remote-scorer";

export class AnswerCorrectnessScorer extends RemoteScorer {
  private constructor(config: Partial<RemoteScorerConfig> = {}) {
    super({
      scoreType: "Answer Correctness",
      ...config,
      requiredParams: [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.EXPECTED_OUTPUT,
      ],
    });
  }

  static get(
    config: Partial<RemoteScorerConfig> = {},
  ): AnswerCorrectnessScorer {
    return new AnswerCorrectnessScorer(config);
  }
}
