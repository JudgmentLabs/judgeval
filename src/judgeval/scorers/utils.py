"""
Util functions for Scorer objects
"""

from typing import List

from judgeval.scorers import BaseScorer


def clone_scorers(scorers: List[BaseScorer]) -> List[BaseScorer]:
    """
    Creates duplicates of the scorers passed as argument.
    """
    cloned_scorers = []
    for s in scorers:
        cloned_scorers.append(s.model_copy(deep=True))
    return cloned_scorers
