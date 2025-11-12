import { APIScorerType } from "../../data/APIScorerType";
import { APIScorer } from "../APIScorer";

export interface AnswerRelevancyScorerConfig {
  threshold?: number;
  name?: string;
  strictMode?: boolean;
  model?: string;
}

export class AnswerRelevancyScorer extends APIScorer {
  constructor(config: AnswerRelevancyScorerConfig = {}) {
    super(APIScorerType.ANSWER_RELEVANCY);
    this.setRequiredParams(["input", "actual_output"]);
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
