import { APIScorerType } from "../../data/APIScorerType";
import { APIScorer } from "../APIScorer";

export interface AnswerCorrectnessScorerConfig {
  threshold?: number;
  name?: string;
  strictMode?: boolean;
  model?: string;
}

export class AnswerCorrectnessScorer extends APIScorer {
  constructor(config: AnswerCorrectnessScorerConfig = {}) {
    super(APIScorerType.ANSWER_CORRECTNESS);
    this.setRequiredParams(["input", "actual_output", "expected_output"]);
    if (config.threshold !== undefined) {
      this.setThreshold(config.threshold);
    }
    if (config.name) {
      this.setName(config.name);
    }
    if (config.strictMode !== undefined) {
      this.setStrictMode(config.strictMode);
    }
    if (config.model) {
      this.setModel(config.model);
    }
  }
}
