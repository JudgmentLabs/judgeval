export class RemoteScorer {
  readonly scoreType: string;
  readonly name: string;
  readonly threshold: number;
  readonly strictMode: boolean;
  readonly model: string | null;
  readonly requiredParams: readonly string[];
  readonly kwargs: Record<string, unknown>;

  constructor(config: RemoteScorerConfig) {
    this.scoreType = config.scoreType;
    this.name = config.name ?? config.scoreType;
    this.strictMode = config.strictMode ?? false;
    this.threshold = this.strictMode ? 1.0 : (config.threshold ?? 0.5);
    this.model = config.model ?? null;
    this.requiredParams = Object.freeze(config.requiredParams ?? []);
    this.kwargs = config.kwargs ?? {};
  }
}

export interface RemoteScorerConfig {
  scoreType: string;
  name?: string;
  threshold?: number;
  strictMode?: boolean;
  model?: string | null;
  requiredParams?: string[];
  kwargs?: Record<string, unknown>;
}
