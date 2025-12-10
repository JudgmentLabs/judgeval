"""
Implementation of the ScorerData class.

ScorerData holds the information related to a single, completed Scorer evaluation run.
"""

from __future__ import annotations

from judgeval.data.judgment_types import ScorerData
from judgeval.scorers import BaseScorer
from typing import List


def create_scorer_data(scorer: BaseScorer) -> List[ScorerData]:
    """
    After a `scorer` is run, it contains information about the example that was evaluated
    using the scorer. For example, after computing Faithfulness, the `scorer` object will contain
    whether the example passed its threshold, the score, the reason for score, etc.

    This function takes an executed `scorer` object and produces a ScorerData object that
    contains the output of the scorer run that can be exported to be logged as a part of
    the ScorerResult.
    """
    scorers_result = list()

    scorers_result.append(
        ScorerData(
            name=scorer.name,
            threshold=scorer.threshold,
            score=scorer.score,
            minimum_score_range=scorer.minimum_score_range,
            maximum_score_range=scorer.maximum_score_range,
            reason=scorer.reason,
            success=scorer.success,
            strict_mode=scorer.strict_mode,
            evaluation_model=scorer.model,
            error=scorer.error,
            additional_metadata=scorer.additional_metadata,
        )
    )
    if hasattr(scorer, "internal_scorer") and scorer.internal_scorer is not None:
        scorers_result.append(
            ScorerData(
                name=scorer.internal_scorer.name,
                score=scorer.internal_scorer.score,
                threshold=scorer.internal_scorer.threshold,
                minimum_score_range=scorer.internal_scorer.minimum_score_range,
                maximum_score_range=scorer.internal_scorer.maximum_score_range,
                reason=scorer.internal_scorer.reason,
                success=scorer.internal_scorer.success,
                strict_mode=scorer.internal_scorer.strict_mode,
                evaluation_model=scorer.internal_scorer.model,
                error=scorer.internal_scorer.error,
                additional_metadata=scorer.internal_scorer.additional_metadata,
            )
        )
    return scorers_result
