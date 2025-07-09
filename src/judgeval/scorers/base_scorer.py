"""
Base class for all scorers.
"""

from typing import Dict, Optional

from pydantic import BaseModel


from judgeval.judges.utils import create_judge
from typing import Any
from pydantic import model_validator, Field
from judgeval.common.logger import judgeval_logger


class BaseScorer(BaseModel):
    """
    If you want to create a scorer that does not fall under any of the ready-made Judgment scorers,
    you can create a custom scorer by extending this class. This is best used for special use cases
    where none of Judgment's scorers are suitable.
    """

    score_type: str  # type of your scorer (Faithfulness, PromptScorer)
    threshold: float = (
        0.5  # The threshold to pass a test while using this scorer as a scorer
    )
    name: Optional[str] = (
        None  # name of your scorer (Faithfulness, PromptScorer-randomslug)
    )
    score: Optional[float] = None  # The float score of the scorer run on the test case
    score_breakdown: Optional[Dict] = None
    reason: Optional[str] = ""
    using_native_model: Optional[bool] = None  # Whether the model is a native model
    success: Optional[bool] = None  # Whether the test case passed or failed
    model: Optional[Any] = Field(
        default=None, exclude=True
    )  # The model used to evaluate the test case
    evaluation_model: Optional[str] = None  # The model used to evaluate the test case
    strict_mode: bool = False  # Whether to run the scorer in strict mode
    error: Optional[str] = None  # The error message if the scorer failed
    additional_metadata: Optional[Dict] = None  # Additional metadata for the scorer
    user: Optional[str] = None  # The user ID of the scorer

    @model_validator(mode="before")
    @classmethod
    def enforce_strict_threshold(cls, data: dict):
        if data.get("strict_mode"):
            data["threshold"] = 1.0
        return data

    @model_validator(mode="after")
    @classmethod
    def default_name(cls, m: "BaseScorer") -> "BaseScorer":
        if not m.name:
            # Try to use the class name if it exists and is not empty
            class_name = getattr(m, "__class__", None)
            if class_name and getattr(m.__class__, "__name__", None):
                m.name = m.__class__.__name__
            else:
                m.name = m.score_type
        return m

    def _add_model(self, model: str):
        """
        Adds the evaluation model to the BaseScorer instance

        This method is used at eval time
        """
        self.model, self.using_native_model = create_judge(model)
        self.evaluation_model = self.model.get_model_name()

    def success_check(self) -> bool:
        """
        For unit testing, determines whether the test case passes or fails
        """
        if self.error:
            return False
        if self.score is None:
            return False
        return self.score >= self.threshold

    def __str__(self):
        if self.error:
            judgeval_logger.warning(f"BaseScorer contains error: {self.error}")
        attributes = {
            "score_type": self.score_type,
            "threshold": self.threshold,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "reason": self.reason,
            "success": self.success,
            "model": self.model,
            "evaluation_model": self.evaluation_model,
            "strict_mode": self.strict_mode,
            "error": self.error,
            "additional_metadata": self.additional_metadata,
        }
        return f"BaseScorer({attributes})"
