"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel, field_validator
from typing import List
from judgeval.data import ExampleParams
from judgeval.constants import APIScorerType, UNBOUNDED_SCORERS
from judgeval.common.logger import judgeval_logger


class APIScorerConfig(BaseModel):
    """
    Scorer config that is used to send to our Judgment server.

    Args:
        score_type (APIScorer): The Judgment metric to use for scoring `Example`s
        name (str): The name of the scorer, usually this is the same as the score_type
        threshold (float): A value between 0 and 1 that determines the scoring threshold
        strict_mode (bool): Whether to use strict mode for the scorer
        required_params (List[ExampleParams]): List of the required parameters on examples for the scorer
        kwargs (dict): Additional keyword arguments to pass to the scorer
    """

    score_type: APIScorerType
    name: str = ""
    threshold: float = 0.5
    strict_mode: bool = False
    required_params: List[
        ExampleParams
    ] = []  # This is used to check if the example has the required parameters before running the scorer
    kwargs: dict = {}

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v, info):
        """
        Validates that the threshold is between 0 and 1 inclusive.
        """
        score_type = info.data.get("score_type")
        if score_type in UNBOUNDED_SCORERS:
            if v < 0:
                judgeval_logger.error(
                    f"Threshold for {score_type} must be greater than 0, got: {v}"
                )
                raise ValueError(
                    f"Threshold for {score_type} must be greater than 0, got: {v}"
                )
        else:
            if not 0 <= v <= 1:
                judgeval_logger.error(
                    f"Threshold for {score_type} must be between 0 and 1, got: {v}"
                )
                raise ValueError(
                    f"Threshold for {score_type} must be between 0 and 1, got: {v}"
                )
        return v

    @field_validator("name", mode="after")
    @classmethod
    def set_name_to_score_type_if_none(cls, v, info):
        """Set name to score_type if not provided"""
        if v is None:
            return info.data.get("score_type")
        return v

    def __str__(self):
        return f"JudgmentScorer(score_type={self.score_type.value}, threshold={self.threshold})"
