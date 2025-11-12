import type { ScorerConfig } from "../internal/api/models";

export abstract class BaseScorer {
  abstract getName(): string;
  abstract getScorerConfig(): ScorerConfig;
}
