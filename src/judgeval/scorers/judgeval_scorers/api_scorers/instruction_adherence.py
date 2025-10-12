from judgeval.scorers.api_scorer import ExampleAPIScorerConfig
from judgeval.constants import APIScorerType
from judgeval.data import ExampleParams


class InstructionAdherenceScorer(ExampleAPIScorerConfig):
    def __init__(self, threshold: float):
        super().__init__(
            threshold=threshold,
            score_type=APIScorerType.INSTRUCTION_ADHERENCE,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
            ],
        )
