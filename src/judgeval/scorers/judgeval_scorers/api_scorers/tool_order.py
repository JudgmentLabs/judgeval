"""
`judgeval` tool order scorer
"""

# Internal imports
from judgeval.scorers.api_scorer import APIScorerConfig
from judgeval.constants import APIScorerType


class ToolOrderScorer(APIScorerConfig):
    score_type: APIScorerType = APIScorerType.TOOL_ORDER
    threshold: float = 1.0
    exact_match: bool = False

    def to_dict(self) -> dict:
        return {
            "score_type": APIScorerType.TOOL_ORDER,
            "threshold": self.threshold,
            "kwargs": {
                "exact_match": self.exact_match,
            },
        }
