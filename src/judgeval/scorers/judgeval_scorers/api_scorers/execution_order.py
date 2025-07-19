"""
`judgeval` tool correctness scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from typing import Optional, Dict
from judgeval.data import ExampleParams


class ExecutionOrderScorer(APIScorerConfig):
    kwargs: Optional[Dict] = None

    def __init__(
        self,
        threshold: float,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            score_type=APIScorerType.EXECUTION_ORDER,
            required_params=[
                ExampleParams.ACTUAL_OUTPUT,
                ExampleParams.EXPECTED_OUTPUT,
            ],
        )
        self.kwargs = {
            "should_exact_match": should_exact_match,
            "should_consider_ordering": should_consider_ordering,
        }

    @property
    def __name__(self):
        return "Execution Order"

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
