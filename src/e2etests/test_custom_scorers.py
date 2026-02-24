from judgeval.v1.scorers import ExampleScorer
from judgeval.v1.scorers.base_custom_scorer.custom_scorer_result import (
    CustomScorerResult,
)
from judgeval.v1 import Judgeval
from judgeval.v1.data import Example
from typing import List


def test_basic_custom_scorer(client: Judgeval, random_name: str):
    class HappinessScorer(ExampleScorer):
        def score(self, data: Example) -> CustomScorerResult:
            actual_output = data._properties.get("actual_output") or ""
            if "happy" in actual_output:
                return CustomScorerResult(score=1.0, reason="happy detected")
            elif "sad" in actual_output:
                return CustomScorerResult(score=0.0, reason="sad detected")
            else:
                return CustomScorerResult(score=0.5, reason="neutral")

    examples: List[Example] = [
        Example.create(actual_output="I'm happy"),
        Example.create(actual_output="I'm sad"),
        Example.create(actual_output="I dont know"),
    ]

    scorer = HappinessScorer()
    results = []
    for example in examples:
        results.append(scorer.score(example))

    assert results[0].score == 1.0
    assert results[1].score == 0.0
    assert results[2].score == 0.5
