"""
Util functions for Scorer objects
"""

from typing import List

from judgeval.scorers import BaseScorer


def clone_scorers(scorers: List[BaseScorer]) -> List[BaseScorer]:
    """
    Creates duplicates of the scorers passed as argument.
    """
    return [s.model_copy(deep=True) for s in scorers]
