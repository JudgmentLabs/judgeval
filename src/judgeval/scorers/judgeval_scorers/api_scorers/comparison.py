"""
`judgeval` comparison scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from typing import Optional, Dict
from judgeval.data import ExampleParams


class ComparisonScorer(APIJudgmentScorer):
    kwargs: Optional[Dict] = None

    def __init__(self, threshold: float, criteria: str, description: str):
        super().__init__(
            threshold=threshold,
            score_type=APIScorer.COMPARISON,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
                ExampleParams.EXPECTED_OUTPUT,
            ],
        )
        self.kwargs = {"criteria": criteria, "description": description}

    @property
    def __name__(self):
        return f"Comparison-{self.kwargs['criteria']}"

    def to_dict(self) -> dict:
        """
        Converts the scorer configuration to a dictionary format.

        Returns:
            dict: A dictionary containing the scorer's configuration
        """
        return {
            "score_type": self.score_type,
            "threshold": self.threshold,
            "kwargs": self.kwargs,
        }
