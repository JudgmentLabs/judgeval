from __future__ import annotations

from typing import TypedDict, List

from .example_evaluation_run import ExampleEvaluationRun
from .scoring_result import ScoringResult


class LogEvalResultsRequest(TypedDict):
    results: List[ScoringResult]
    run: ExampleEvaluationRun
