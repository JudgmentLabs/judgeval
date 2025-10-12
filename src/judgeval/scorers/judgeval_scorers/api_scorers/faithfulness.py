from judgeval.scorers.api_scorer import ExampleAPIScorerConfig
from judgeval.constants import APIScorerType
from judgeval.data import ExampleParams
from typing import List


class FaithfulnessScorer(ExampleAPIScorerConfig):
    score_type: APIScorerType = APIScorerType.FAITHFULNESS
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.RETRIEVAL_CONTEXT,
    ]
