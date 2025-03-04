"""
Judgment Scorer class.

Scores `Example`s using ready-made Judgment evaluators.
"""

from pydantic import BaseModel, field_validator
from judgeval.common.logger import debug, info, warning, error

from judgeval.constants import APIScorer


class APIJudgmentScorer(BaseModel):
    """
    Class for ready-made, "out-of-the-box" scorer that uses Judgment evaluators to score `Example`s.

    Args:
        score_type (APIScorer): The Judgment metric to use for scoring `Example`s
        threshold (float): A value between 0 and 1 that determines the scoring threshold
    """
    threshold: float
    score_type: APIScorer

    @field_validator('threshold')
    def validate_threshold(cls, v):
        """
        Validates that the threshold is between 0 and 1 inclusive.
        """
        if not 0 <= v <= 1:
            error(f"Threshold must be between 0 and 1, got: {v}")
            raise ValueError(f"Threshold must be between 0 and 1, got: {v}")
        return v

    @field_validator('score_type')
    def convert_to_enum_value(cls, v):
        """
        Validates that the `score_type` is a valid `APIScorer` enum value.
        Converts string values to `APIScorer` enum values.
        """
        debug(f"Attempting to convert score_type value: {v}")
        if isinstance(v, APIScorer):
            info(f"Using existing APIScorer: {v}")
            return v
        elif isinstance(v, str):
            debug(f"Converting string value to APIScorer enum: {v}")
            return APIScorer[v.upper()]
        error(f"Invalid score_type value: {v}")
        raise ValueError(f"Invalid value for score_type: {v}")
    
    def __str__(self):
        return f"JudgmentScorer(score_type={self.score_type}, threshold={self.threshold})"
    
    def to_dict(self) -> dict:
        """
        Converts the scorer configuration to a dictionary format.
        
        Returns:
            dict: A dictionary containing the scorer's configuration
        """
        return {
            "score_type": self.score_type.value,  # Convert enum to string for serialization
            "threshold": self.threshold
        }
    