from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import Example
from typing import List
from pydantic import Field
from judgeval.common.logger import judgeval_logger


class ExampleScorer(BaseScorer):
    score_type: str = "Custom"  # default to custom score type
    required_params: List[str] = Field(default_factory=list)

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single example
        """
        judgeval_logger.error("a_score_example method not implemented")
        raise NotImplementedError(
            "You must implement the `a_score_example` method in your custom scorer"
        )
