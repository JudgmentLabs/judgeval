import { ExampleParams } from "../../data";
import { RemoteScorer, RemoteScorerConfig } from "../remote-scorer";

export class InstructionAdherenceScorer extends RemoteScorer {
  private constructor(config: Partial<RemoteScorerConfig> = {}) {
    super({
      scoreType: "Instruction Adherence",
      ...config,
      requiredParams: [ExampleParams.INPUT, ExampleParams.ACTUAL_OUTPUT],
    });
  }

  static get(
    config: Partial<RemoteScorerConfig> = {},
  ): InstructionAdherenceScorer {
    return new InstructionAdherenceScorer(config);
  }
}
