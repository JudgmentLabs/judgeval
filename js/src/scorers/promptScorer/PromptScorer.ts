import type { ScorerConfig } from "../../internal/api/models";
import { APIScorerType } from "../../data/APIScorerType";
import { BaseScorer } from "../BaseScorer";

export interface PromptScorerConfig {
  name: string;
  prompt: string;
  threshold: number;
  options?: Record<string, number>;
  model?: string;
  description?: string;
  isTrace?: boolean;
}

export class PromptScorer extends BaseScorer {
  private readonly _name: string;
  private _prompt: string;
  private _threshold: number;
  private _options?: Record<string, number>;
  private _model?: string;
  private _description?: string;
  private readonly isTrace: boolean;

  constructor(config: PromptScorerConfig) {
    super();
    this._name = config.name;
    this._prompt = config.prompt;
    this._threshold = config.threshold;
    this._options = { ...config.options };
    this._model = config.model;
    this._description = config.description;
    this.isTrace = config.isTrace ?? false;
  }

  getName(): string {
    return this._name;
  }

  getPrompt(): string {
    return this._prompt;
  }

  getThreshold(): number {
    return this._threshold;
  }

  getOptions(): Record<string, number> | undefined {
    return this._options ? { ...this._options } : undefined;
  }

  getModel(): string | undefined {
    return this._model;
  }

  getDescription(): string | undefined {
    return this._description;
  }

  setThreshold(threshold: number): void {
    this._threshold = threshold;
  }

  setPrompt(prompt: string): void {
    this._prompt = prompt;
  }

  setModel(model: string): void {
    this._model = model;
  }

  setOptions(options: Record<string, number>): void {
    this._options = { ...options };
  }

  setDescription(description: string): void {
    this._description = description;
  }

  appendToPrompt(addition: string): void {
    this._prompt = this._prompt + addition;
  }

  getScorerConfig(): ScorerConfig {
    const scoreType = this.isTrace
      ? APIScorerType.TRACE_PROMPT_SCORER
      : APIScorerType.PROMPT_SCORER;
    const kwargs: Record<string, unknown> = {
      prompt: this._prompt,
    };

    if (this._options) {
      kwargs.options = this._options;
    }
    if (this._model) {
      kwargs.model = this._model;
    }
    if (this._description) {
      kwargs.description = this._description;
    }

    return {
      score_type: scoreType,
      threshold: this._threshold,
      name: this._name,
      kwargs,
    };
  }
}
