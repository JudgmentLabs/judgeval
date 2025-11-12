import { CustomScorer } from "./CustomScorer";

export class CustomScorerFactory {
  get(name: string, className?: string): CustomScorer {
    return new CustomScorer({
      name,
      className: className ?? name,
      serverHosted: true,
    });
  }
}
