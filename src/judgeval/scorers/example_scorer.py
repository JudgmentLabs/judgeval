from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import Example
from typing import Any, Dict, List, Tuple, Union
from pydantic import Field

# Type alias for the return type of a_score_example
# Can return just a float score, or a tuple of (score, additional_metadata)
ScorerResult = Union[float, Tuple[float, Dict[str, Any]]]


class ExampleScorer(BaseScorer):
    score_type: str = "Custom"
    required_params: List[str] = Field(default_factory=list)

    async def a_score_example(self, example: Example, *args, **kwargs) -> ScorerResult:
        """
        Asynchronously measures the score on a single example.

        Returns:
            Union[float, Tuple[float, Dict[str, Any]]]: Either:
                - A float score value
                - A tuple of (score, additional_metadata) where additional_metadata
                  is a dict that will be stored in the scorer's additional_metadata field
        """
        raise NotImplementedError(
            "You must implement the `a_score_example` method in your custom scorer"
        )
