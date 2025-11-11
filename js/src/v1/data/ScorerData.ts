import type { ScorerData as APIScorerData } from "../../internal/api/models";

export interface ScorerDataConfig {
  name?: string;
  threshold?: number;
  success?: boolean;
  score?: number;
  reason?: string;
  strictMode?: boolean;
  evaluationModel?: string;
  error?: string;
  additionalMetadata?: Record<string, unknown>;
  id?: string;
}

export class ScorerData {
  name?: string | null;
  threshold?: number | null;
  success?: boolean | null;
  score?: number | null;
  reason?: string | null;
  strictMode?: boolean | null;
  evaluationModel?: string | null;
  error?: string | null;
  additionalMetadata?: Record<string, unknown>;
  id?: string | null;

  constructor(config: ScorerDataConfig = {}) {
    this.name = config.name ?? null;
    this.threshold = config.threshold ?? null;
    this.success = config.success ?? null;
    this.score = config.score ?? null;
    this.reason = config.reason ?? null;
    this.strictMode = config.strictMode ?? null;
    this.evaluationModel = config.evaluationModel ?? null;
    this.error = config.error ?? null;
    this.additionalMetadata = config.additionalMetadata ?? {};
    this.id = config.id ?? null;
  }

  toModel(): APIScorerData {
    const result: APIScorerData = {
      name: this.name ?? "",
      threshold: this.threshold ?? 0,
      success: this.success ?? false,
    };

    if (this.score !== undefined && this.score !== null) {
      result.score = this.score;
    }
    if (this.reason !== undefined && this.reason !== null) {
      result.reason = this.reason;
    }
    if (this.strictMode !== undefined && this.strictMode !== null) {
      result.strict_mode = this.strictMode;
    }
    if (this.evaluationModel !== undefined && this.evaluationModel !== null) {
      result.evaluation_model = this.evaluationModel;
    }
    if (this.error !== undefined && this.error !== null) {
      result.error = this.error;
    }
    if (
      this.additionalMetadata &&
      Object.keys(this.additionalMetadata).length > 0
    ) {
      result.additional_metadata = this.additionalMetadata;
    }
    if (this.id !== undefined && this.id !== null) {
      result.id = this.id;
    }

    return result;
  }
}
