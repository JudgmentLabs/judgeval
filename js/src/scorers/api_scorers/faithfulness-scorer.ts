import { ExampleParams } from "../../data";
import { RemoteScorer, RemoteScorerConfig } from "../remote-scorer";

export class FaithfulnessScorer extends RemoteScorer {
  private constructor(config: Partial<RemoteScorerConfig> = {}) {
    super({
      scoreType: "Faithfulness",
      ...config,
      requiredParams: [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.RETRIEVAL_CONTEXT,
      ],
    });
  }

  static get(config: Partial<RemoteScorerConfig> = {}): FaithfulnessScorer {
    return new FaithfulnessScorer(config);
  }
}
