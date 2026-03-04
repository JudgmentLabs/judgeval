from __future__ import annotations

from typing import TypedDict, List

from .example_evaluation_run import ExampleEvaluationRun
from .local_scorer_result import LocalScorerResult


class LogEvalResultsExamplesRequest(TypedDict):
    results: List[LocalScorerResult]
    run: ExampleEvaluationRun
