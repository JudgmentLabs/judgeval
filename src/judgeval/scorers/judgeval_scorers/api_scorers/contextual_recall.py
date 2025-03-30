"""
`judgeval` contextual recall scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from judgeval.data import ExampleParams


class ContextualRecallScorer(APIJudgmentScorer):
    def __init__(self, threshold: float):
        super().__init__(
            threshold=threshold, 
            score_type=APIScorer.CONTEXTUAL_RECALL,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
                ExampleParams.EXPECTED_OUTPUT,
                ExampleParams.RETRIEVAL_CONTEXT,
            ]
        )
    @property
    def __name__(self):
        return "Contextual Recall"
