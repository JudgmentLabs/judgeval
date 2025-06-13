"""
`judgeval` tool order scorer
"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from typing import Optional, Dict


class ToolOrderScorer(APIJudgmentScorer):
    kwargs: Optional[Dict] = None

    def __init__(self, threshold: float = 1.0, exact_match: bool = False):
        super().__init__(
            threshold=threshold,
            score_type=APIScorer.TOOL_ORDER,
        )
        self.kwargs = {"exact_match": exact_match}

    @property
    def __name__(self):
        return "Tool Order"
