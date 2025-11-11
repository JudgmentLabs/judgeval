import { Example } from "../data";
import { LocalScorer, LocalScorerConfig } from "./local-scorer";

export abstract class ExampleScorer extends LocalScorer {
  abstract scoreExample(example: Example): Promise<number>;

  static get<T extends ExampleScorer>(
    this: new (config: LocalScorerConfig) => T,
    options: ExampleScorerOptions,
  ): T {
    return new this({
      name: options.name,
      threshold: options.threshold,
      strictMode: options.strictMode,
      model: options.model,
    });
  }
}

export interface ExampleScorerOptions {
  name: string;
  threshold?: number;
  strictMode?: boolean;
  model?: string | null;
}
