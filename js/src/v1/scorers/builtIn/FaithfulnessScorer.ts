import { APIScorerType } from "../../data/APIScorerType";
import { APIScorer } from "../APIScorer";

export interface FaithfulnessScorerConfig {
  threshold?: number;
  name?: string;
  strictMode?: boolean;
  model?: string;
}

export class FaithfulnessScorer extends APIScorer {
  constructor(config: FaithfulnessScorerConfig = {}) {
    super(APIScorerType.FAITHFULNESS);
    this.setRequiredParams(["context", "actual_output"]);
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
