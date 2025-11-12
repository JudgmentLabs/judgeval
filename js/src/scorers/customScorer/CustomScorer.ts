import type { ScorerConfig } from "../../internal/api/models";
import { APIScorerType } from "../../data/APIScorerType";
import { APIScorer } from "../APIScorer";

export interface CustomScorerConfig {
  name: string;
  className?: string;
  serverHosted?: boolean;
}

export class CustomScorer extends APIScorer {
  private _className?: string;
  private _serverHosted: boolean;

  constructor(config: CustomScorerConfig) {
    super(APIScorerType.CUSTOM);
    this.setName(config.name);
    if (config.className) {
      this._className = config.className;
      this.setAdditionalProperty("class_name", config.className);
    }
    this._serverHosted = config.serverHosted ?? true;
    this.setAdditionalProperty("server_hosted", this._serverHosted);
  }

  getClassName(): string | undefined {
    return this._className;
  }

  isServerHosted(): boolean {
    return this._serverHosted;
  }

  getScorerConfig(): ScorerConfig {
    throw new Error("CustomScorer does not use ScorerConfig");
  }
}
