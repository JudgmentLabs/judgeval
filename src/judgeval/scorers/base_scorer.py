"""
Base class for all scorers.
"""

from __future__ import annotations
from typing import Dict, Optional

from pydantic import BaseModel


from judgeval.judges.utils import create_judge
from typing import Any
from pydantic import model_validator, Field


class BaseScorer(BaseModel):
    """
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class. This is best used for special use cases
    where none of Judgment's scorers are suitable.
    """

    # type of your scorer (Faithfulness, PromptScorer)
    score_type: str

    # The threshold to pass a test while using this scorer as a scorer
    threshold: float = 0.5

    # name of your scorer (Faithfulness, PromptScorer-randomslug)
    name: str = ""

    # The name of the class of the scorer
    class_name: Optional[str] = None

    # The float score of the scorer run on the test case
    score: Optional[float] = None

    maximum_score_range: float = 1
    minimum_score_range: float = 0

    reason: Optional[str] = ""

    # Whether the model is a native model
    using_native_model: Optional[bool] = None

    # Whether the test case passed or failed
    success: bool = False

    # The name of the model used to evaluate the test case
    model: Optional[str] = None

    # The model used to evaluate the test case
    model_client: Optional[Any] = Field(default=None, exclude=True)

    # Whether to run the scorer in strict mode
    strict_mode: bool = False

    # The error message if the scorer failed
    error: Optional[str] = None

    # Additional metadata for the scorer
    additional_metadata: Optional[Dict] = None

    # The user ID of the scorer
    user: Optional[str] = None

    # Whether the scorer is hosted on the server
    server_hosted: bool = False

    @model_validator(mode="after")
    def enforce_strict_threshold(self):
        if self.strict_mode:
            self.threshold = 1.0
        return self

    @model_validator(mode="after")
    def default_name(self):
        self.class_name = self.__class__.__name__
        if not self.name:
            self.name = self.class_name
        return self

    @model_validator(mode="after")
    def validate_scorer_range(self):
        """
        Validates that minimum_score_range <= maximum_score_range.
        Raises ValueError if the ranges are invalid.
        """
        if self.minimum_score_range > self.maximum_score_range:
            raise ValueError(
                f"minimum_score_range ({self.minimum_score_range}) must be less than or equal to "
                f"maximum_score_range ({self.maximum_score_range})"
            )
        return self

    def _add_model(self, model: str):
        """
        Adds the evaluation model to the BaseScorer instance

        This method is used at eval time
        """
        self.model_client, self.using_native_model = create_judge(model)

    def success_check(self) -> bool:
        """
        For unit testing, determines whether the test case passes or fails
        """
        if self.error:
            return False
        if self.score is None:
            return False
        return self.score >= self.threshold
