import type { ScorerConfig } from "../internal/api/models";
import type { APIScorerType } from "../data/APIScorerType";
import { BaseScorer } from "./BaseScorer";

export class APIScorer extends BaseScorer {
  private readonly scoreType: string;
  private requiredParams: string[];
  private _threshold: number;
  private _name: string;
  private _strictMode: boolean;
  private _model?: string;
  private additionalProperties: Record<string, unknown>;

  constructor(scoreType: APIScorerType) {
    super();
    this.scoreType = scoreType;
    this._name = scoreType;
    this._threshold = 0.5;
    this._strictMode = false;
    this.requiredParams = [];
    this.additionalProperties = {};
  }

  getName(): string {
    return this._name;
  }

  getScoreType(): string {
    return this.scoreType;
  }

  getThreshold(): number {
    return this._threshold;
  }

  getStrictMode(): boolean {
    return this._strictMode;
  }

  getModel(): string | undefined {
    return this._model;
  }

  getRequiredParams(): string[] {
    return [...this.requiredParams];
  }

  setThreshold(threshold: number): void {
    if (threshold < 0 || threshold > 1) {
      throw new Error(`Threshold must be between 0 and 1, got: ${threshold}`);
    }
    this._threshold = threshold;
    if (this._strictMode) {
      this._threshold = 1.0;
    }
  }

  setName(name: string): void {
    this._name = name;
  }

  setStrictMode(strictMode: boolean): void {
    this._strictMode = strictMode;
    if (strictMode) {
      this._threshold = 1.0;
    }
  }

  setModel(model: string): void {
    this._model = model;
  }

  setRequiredParams(requiredParams: string[]): void {
    this.requiredParams = [...requiredParams];
  }

  setAdditionalProperty(key: string, value: unknown): void {
    this.additionalProperties[key] = value;
  }

  getAdditionalProperties(): Record<string, unknown> {
    return { ...this.additionalProperties };
  }

  getScorerConfig(): ScorerConfig {
    const kwargs: Record<string, unknown> = { ...this.additionalProperties };

    return {
      score_type: this.scoreType,
      threshold: this._threshold,
      name: this._name,
      strict_mode: this._strictMode,
      required_params: this.requiredParams,
      kwargs,
    };
  }

  static builder<T extends APIScorer>(
    this: new (scoreType: APIScorerType) => T,
    scoreType: APIScorerType,
  ): APIScorerBuilder<T> {
    return new APIScorerBuilder(this, scoreType);
  }
}

export class APIScorerBuilder<T extends APIScorer> {
  private scorer: T;

  constructor(
    scorerClass: new (scoreType: APIScorerType) => T,
    scoreType: APIScorerType,
  ) {
    this.scorer = new scorerClass(scoreType);
  }

  threshold(threshold: number): this {
    this.scorer.setThreshold(threshold);
    return this;
  }

  name(name: string): this {
    this.scorer.setName(name);
    return this;
  }

  strictMode(strictMode: boolean): this {
    this.scorer.setStrictMode(strictMode);
    return this;
  }

  requiredParams(requiredParams: string[]): this {
    this.scorer.setRequiredParams(requiredParams);
    return this;
  }

  model(model: string): this {
    this.scorer.setModel(model);
    return this;
  }

  additionalProperty(key: string, value: unknown): this {
    this.scorer.setAdditionalProperty(key, value);
    return this;
  }

  build(): T {
    return this.scorer;
  }
}
