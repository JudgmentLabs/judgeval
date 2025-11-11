import { BaseScorer as BaseScorerModel } from "../internal/api/models/BaseScorer";
import { ScorerConfig } from "../internal/api/models/ScorerConfig";
import { LocalScorer } from "./local-scorer";
import { RemoteScorer } from "./remote-scorer";

export function localScorerToBaseScorer(scorer: LocalScorer): BaseScorerModel {
  return {
    score_type: "Custom",
    name: scorer.name,
    class_name: scorer.constructor.name,
    threshold: scorer.threshold,
    strict_mode: scorer.strictMode,
    model: scorer.model,
    score: scorer.score,
    score_breakdown: null,
    reason: scorer.reason,
    using_native_model: null,
    success: scorer.success,
    model_client: null,
    error: scorer.error,
    additional_metadata: scorer.metadata,
    user: null,
    server_hosted: false,
  };
}

export function remoteScorerToScorerConfig(scorer: RemoteScorer): ScorerConfig {
  return {
    score_type: scorer.scoreType,
    name: scorer.name,
    threshold: scorer.threshold,
    model: scorer.model,
    strict_mode: scorer.strictMode,
    required_params: Array.from(scorer.requiredParams),
    kwargs: scorer.kwargs,
  };
}
