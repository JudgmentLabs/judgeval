import {
  AnswerCorrectnessScorer,
  AnswerCorrectnessScorerConfig,
} from "./AnswerCorrectnessScorer";
import {
  AnswerRelevancyScorer,
  AnswerRelevancyScorerConfig,
} from "./AnswerRelevancyScorer";
import {
  FaithfulnessScorer,
  FaithfulnessScorerConfig,
} from "./FaithfulnessScorer";
export class BuiltInScorersFactory {
  answerCorrectness(
    config: AnswerCorrectnessScorerConfig = {},
  ): AnswerCorrectnessScorer {
    return new AnswerCorrectnessScorer(config);
  }

  answerRelevancy(
    config: AnswerRelevancyScorerConfig = {},
  ): AnswerRelevancyScorer {
    return new AnswerRelevancyScorer(config);
  }

  faithfulness(config: FaithfulnessScorerConfig = {}): FaithfulnessScorer {
    return new FaithfulnessScorer(config);
  }
}
