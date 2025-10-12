from typing import List, Optional, Union, Tuple, Sequence
from pydantic import field_validator, model_validator, Field, BaseModel
from datetime import datetime, timezone
import uuid

from judgeval.data import Example
from judgeval.scorers import APIScorerConfig
from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.constants import ACCEPTABLE_MODELS
from judgeval.data.judgment_types import (
    ExampleEvaluationRun as ExampleEvaluationRunJudgmentType,
    TraceEvaluationRun as TraceEvaluationRunJudgmentType,
)


class EvaluationRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    custom_scorers: List[ExampleScorer] = Field(default_factory=list)
    judgment_scorers: Sequence[APIScorerConfig] = Field(default_factory=list)
    scorers: Sequence[Union[ExampleScorer, APIScorerConfig]] = Field(
        default_factory=list
    )
    model: Optional[str] = None

    def __init__(
        self,
        scorers: Optional[List[Union[ExampleScorer, APIScorerConfig]]] = None,
        **kwargs,
    ):
        """
        Initialize EvaluationRun with automatic scorer classification.

        Args:
            scorers: List of scorers that will be automatically sorted into custom_scorers or judgment_scorers
            **kwargs: Other initialization arguments
        """
        if scorers is not None:
            # Automatically sort scorers into appropriate fields
            custom_scorers = [s for s in scorers if isinstance(s, ExampleScorer)]
            judgment_scorers = [s for s in scorers if isinstance(s, APIScorerConfig)]

            # Always set both fields as lists (even if empty) to satisfy validation
            kwargs["custom_scorers"] = custom_scorers
            kwargs["judgment_scorers"] = judgment_scorers

        super().__init__(**kwargs)

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["custom_scorers"] = [s.model_dump() for s in self.custom_scorers]
        data["judgment_scorers"] = [s.model_dump() for s in self.judgment_scorers]

        return data

    @model_validator(mode="after")
    @classmethod
    def validate_scorer_lists(cls, values):
        custom_scorers = values.custom_scorers
        judgment_scorers = values.judgment_scorers

        # Check that both lists are not empty
        if not custom_scorers and not judgment_scorers:
            raise ValueError(
                "At least one of custom_scorers or judgment_scorers must be provided."
            )

        # Check that only one list is filled
        if custom_scorers and judgment_scorers:
            raise ValueError(
                "Only one of custom_scorers or judgment_scorers can be provided, not both."
            )

        return values

    @field_validator("model")
    def validate_model(cls, v, values):
        # Check if model is string or list of strings
        if v is not None and isinstance(v, str):
            if v not in ACCEPTABLE_MODELS:
                raise ValueError(
                    f"Model name {v} not recognized. Please select a valid model name.)"
                )
            return v


class ExampleEvaluationRun(EvaluationRun, ExampleEvaluationRunJudgmentType):  # type: ignore
    """
    Stores example and evaluation scorers together for running an eval task

    Args:
        project_name (str): The name of the project the evaluation results belong to
        eval_name (str): A name for this evaluation run
        examples (List[Example]): The examples to evaluate
        scorers (List[Union[BaseScorer, APIScorerConfig]]): A list of scorers to use for evaluation
        model (str): The model used as a judge when using LLM as a Judge
    """

    examples: List[Example]  # type: ignore

    @field_validator("examples")
    def validate_examples(cls, v):
        if not v:
            raise ValueError("Examples cannot be empty.")
        for item in v:
            if not isinstance(item, Example):
                raise ValueError(f"Item of type {type(item)} is not a Example")
        return v

    def model_dump(self, **kwargs):
        data = super().model_dump(**kwargs)
        data["examples"] = [example.model_dump() for example in self.examples]
        return data


class TraceEvaluationRun(EvaluationRun, TraceEvaluationRunJudgmentType):  # type: ignore
    trace_and_span_ids: List[Tuple[str, str]]  # type: ignore

    @field_validator("trace_and_span_ids")
    def validate_trace_and_span_ids(cls, v):
        if not v:
            raise ValueError("Trace and span IDs are required for trace evaluations.")
        return v
