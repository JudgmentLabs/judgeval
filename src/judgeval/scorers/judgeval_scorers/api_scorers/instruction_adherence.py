"""
`judgeval` instruction adherence scorer

TODO add link to docs page for this scorer

"""

# Internal imports
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType
from judgeval.data import ExampleParams


class InstructionAdherenceScorer(APIScorerConfig):
    def __init__(self, threshold: float):
        super().__init__(
            threshold=threshold,
            score_type=APIScorerType.INSTRUCTION_ADHERENCE,
            required_params=[
                ExampleParams.INPUT,
                ExampleParams.ACTUAL_OUTPUT,
            ],
        )

    @property
    def __name__(self):
        return "Instruction Adherence"
