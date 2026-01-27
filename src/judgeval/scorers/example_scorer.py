from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import Example
from typing import List
from pydantic import Field


class ExampleScorer(BaseScorer):
    score_type: str = "Custom"
    required_params: List[str] = Field(default_factory=list)

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single example
        """
        raise NotImplementedError(
            "You must implement the `a_score_example` method in your custom scorer"
        )
