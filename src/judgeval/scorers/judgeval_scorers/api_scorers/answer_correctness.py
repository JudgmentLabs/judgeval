from judgeval.scorers.api_scorer import ExampleAPIScorerConfig
from judgeval.constants import APIScorerType
from judgeval.data import ExampleParams
from typing import List


class AnswerCorrectnessScorer(ExampleAPIScorerConfig):
    score_type: APIScorerType = APIScorerType.ANSWER_CORRECTNESS
    required_params: List[ExampleParams] = [
        ExampleParams.INPUT,
        ExampleParams.ACTUAL_OUTPUT,
        ExampleParams.EXPECTED_OUTPUT,
    ]
