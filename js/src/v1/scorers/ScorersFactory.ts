import { JudgmentApiClient } from "../../internal/api";
import { BuiltInScorersFactory } from "./builtIn/BuiltInScorersFactory";
import { CustomScorerFactory } from "./customScorer/CustomScorerFactory";
import { PromptScorerFactory } from "./promptScorer/PromptScorerFactory";

export class ScorersFactory {
  private readonly client: JudgmentApiClient;

  constructor(client: JudgmentApiClient) {
    this.client = client;
  }

  get promptScorer(): PromptScorerFactory {
    return new PromptScorerFactory(this.client, false);
  }

  get tracePromptScorer(): PromptScorerFactory {
    return new PromptScorerFactory(this.client, true);
  }

  get customScorer(): CustomScorerFactory {
    return new CustomScorerFactory();
  }

  get builtIn(): BuiltInScorersFactory {
    return new BuiltInScorersFactory();
  }
}
