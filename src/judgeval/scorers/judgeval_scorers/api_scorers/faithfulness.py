"""
`judgeval` faithfulness scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from judgeval.data import ExampleParams
from typing import List


class FaithfulnessScorer(APIScorerConfig):
    score_type: APIScorerType = APIScorerType.FAITHFULNESS
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.RETRIEVAL_CONTEXT,
    ]
