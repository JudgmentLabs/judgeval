import { JUDGMENT_DEFAULT_GPT_MODEL } from "../../../env";
import { JudgmentApiClient } from "../../../internal/api";
import type {
  PromptScorer as APIPromptScorer,
  FetchPromptScorersRequest,
  FetchPromptScorersResponse,
} from "../../../internal/api/models";
import { PromptScorer, type PromptScorerConfig } from "./PromptScorer";

export class PromptScorerFactory {
  private readonly client: JudgmentApiClient;
  private readonly isTrace: boolean;
  private static readonly cache = new Map<string, APIPromptScorer>();

  constructor(client: JudgmentApiClient, isTrace: boolean) {
    this.client = client;
    this.isTrace = isTrace;
  }

  async get(name: string): Promise<PromptScorer> {
    const cacheKey = this.getCacheKey(name);
    const cached = PromptScorerFactory.cache.get(cacheKey);

    if (cached) {
      return this.createFromModel(cached, name);
    }

    try {
      const request: FetchPromptScorersRequest = {
        names: [name],
      };

      const response: FetchPromptScorersResponse =
        await this.client.fetchScorers(request);

      if (response.scorers.length === 0) {
        throw new Error(`Failed to fetch prompt scorer '${name}': not found`);
      }

      const scorer = response.scorers[0];
      const scorerIsTrace = scorer.is_trace === true;

      if (scorerIsTrace !== this.isTrace) {
        const expectedType = this.isTrace
          ? "TracePromptScorer"
          : "PromptScorer";
        const actualType = scorerIsTrace ? "TracePromptScorer" : "PromptScorer";
        throw new Error(
          `Scorer with name ${name} is a ${actualType}, not a ${expectedType}`,
        );
      }

      PromptScorerFactory.cache.set(cacheKey, scorer);
      return this.createFromModel(scorer, name);
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Failed to fetch prompt scorer '${name}': ${error}`);
    }
  }

  create(config: PromptScorerConfig): PromptScorer {
    if (!config.name) {
      throw new Error("Name is required");
    }
    if (!config.prompt) {
      throw new Error("Prompt is required");
    }

    return new PromptScorer({
      ...config,
      isTrace: this.isTrace,
    });
  }

  private createFromModel(model: APIPromptScorer, name: string): PromptScorer {
    let options: Record<string, number> | undefined;
    if (model.options && typeof model.options === "object") {
      options = {};
      for (const [key, value] of Object.entries(model.options)) {
        if (typeof value === "number") {
          options[key] = value;
        }
      }
    }

    return new PromptScorer({
      name,
      prompt: model.prompt,
      threshold: model.threshold,
      options: options ?? {},
      model: model.model ?? JUDGMENT_DEFAULT_GPT_MODEL,
      description: model.description ?? "",
      isTrace: this.isTrace,
    });
  }

  private getCacheKey(name: string): string {
    return `${name}:${this.client.getApiKey()}:${this.client.getOrganizationId()}`;
  }
}
