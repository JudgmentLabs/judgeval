import { Example } from "../data";

export abstract class LocalScorer {
  readonly name: string;
  readonly threshold: number;
  readonly strictMode: boolean;
  readonly model: string | null;

  score: number | null = null;
  reason: string | null = null;
  error: string | null = null;
  success = false;
  metadata: Record<string, unknown> | null = null;

  constructor(config: LocalScorerConfig) {
    this.name = config.name;
    this.strictMode = config.strictMode ?? false;
    this.threshold = this.strictMode ? 1.0 : (config.threshold ?? 0.5);
    this.model = config.model ?? null;
  }

  abstract scoreExample(example: Example): Promise<number>;

  successCheck(): boolean {
    if (this.error !== null) return false;
    if (this.score === null) return false;
    return this.score >= this.threshold;
  }
}

export interface LocalScorerConfig {
  name: string;
  threshold?: number;
  strictMode?: boolean;
  model?: string | null;
}
