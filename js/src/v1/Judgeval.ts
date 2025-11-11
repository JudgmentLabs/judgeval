import { JUDGMENT_API_KEY, JUDGMENT_API_URL, JUDGMENT_ORG_ID } from "../env";
import { JudgmentApiClient } from "../internal/api";
import { EvaluationFactory } from "./evaluation/EvaluationFactory";
import { ScorersFactory } from "./scorers/ScorersFactory";
import { BrowserTracerFactory } from "./tracer/BrowserTracerFactory";
import { NodeTracerFactory } from "./tracer/NodeTracerFactory";

export interface JudgevalConfig {
  apiKey?: string;
  organizationId?: string;
  apiUrl?: string;
}

export class Judgeval {
  private readonly internalClient: JudgmentApiClient;

  protected constructor(config: JudgevalConfig = {}) {
    const apiKey = config.apiKey ?? JUDGMENT_API_KEY;
    const organizationId = config.organizationId ?? JUDGMENT_ORG_ID;
    const apiUrl = config.apiUrl ?? JUDGMENT_API_URL;

    if (!apiKey) {
      throw new Error("API key is required");
    }
    if (!organizationId) {
      throw new Error("Organization ID is required");
    }
    if (!apiUrl) {
      throw new Error("API URL is required");
    }

    this.internalClient = new JudgmentApiClient(apiUrl, apiKey, organizationId);
  }

  static create(config: JudgevalConfig = {}): Judgeval {
    return new Judgeval(config);
  }

  get nodeTracer(): NodeTracerFactory {
    return new NodeTracerFactory(this.internalClient);
  }

  get browserTracer(): BrowserTracerFactory {
    return new BrowserTracerFactory(this.internalClient);
  }

  get scorers(): ScorersFactory {
    return new ScorersFactory(this.internalClient);
  }

  get evaluation(): EvaluationFactory {
    return new EvaluationFactory(this.internalClient);
  }
}
