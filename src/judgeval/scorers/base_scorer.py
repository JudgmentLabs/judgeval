"""
Base class for all scorers.
"""

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

    score_type: str  # type of your scorer (Faithfulness, PromptScorer)
    threshold: float = (
        0.5  # The threshold to pass a test while using this scorer as a scorer
    )
    name: Optional[str] = (
        None  # name of your scorer (Faithfulness, PromptScorer-randomslug)
    )
    class_name: Optional[str] = None  # The name of the class of the scorer
    score: Optional[float] = None  # The float score of the scorer run on the test case
    score_breakdown: Optional[Dict] = None
    reason: Optional[str] = ""
    using_native_model: Optional[bool] = None  # Whether the model is a native model
    success: Optional[bool] = None  # Whether the test case passed or failed
    model: Optional[str] = None  # The name of the model used to evaluate the test case
    model_client: Optional[Any] = Field(
        default=None, exclude=True
    )  # The model used to evaluate the test case
    strict_mode: bool = False  # Whether to run the scorer in strict mode
    error: Optional[str] = None  # The error message if the scorer failed
    additional_metadata: Optional[Dict] = None  # Additional metadata for the scorer
    user: Optional[str] = None  # The user ID of the scorer
    server_hosted: bool = False  # Whether the scorer is enabled for e2b

    @model_validator(mode="after")
    @classmethod
    def enforce_strict_threshold(cls, data: "BaseScorer"):
        if data.strict_mode:
            data.threshold = 1.0
        return data

    @model_validator(mode="after")
    @classmethod
    def default_name(cls, m: "BaseScorer") -> "BaseScorer":
        # Always set class_name to the string name of the class
        m.class_name = m.__class__.__name__
        if not m.name:
            m.name = m.class_name
        return m

    def _add_model(self, model: str):
        """
        Adds the evaluation model to the BaseScorer instance

        This method is used at eval time
        """
        self.model_client, self.using_native_model = create_judge(model)
        self.model = self.model_client.get_model_name() or model

    def success_check(self) -> bool:
        """
        For unit testing, determines whether the test case passes or fails
        """
        if self.error:
            return False
        if self.score is None:
            return False
        return self.score >= self.threshold
