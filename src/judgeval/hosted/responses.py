from typing import ClassVar, List, Literal, Optional, Union
from abc import ABC
from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator
from typing_extensions import final

ReturnType = Literal["binary", "categorical", "numeric"]


class Citation(BaseModel):
    """Links a score back to a specific span in a trace.

    Use citations in your `Judge.score()` return value to highlight which
    part of the trace contributed to the score.

    Attributes:
        span_id: The span ID being referenced.
        span_attribute: The attribute name within the span.
    """

    span_id: str
    span_attribute: str


class Category(BaseModel):
    """Defines one allowed category for a `CategoricalResponse` scorer.

    Attributes:
        value: Category label (must be unique within the scorer).
        description: What this category means.
    """

    value: str
    description: str = ""


class BaseResponse(BaseModel, ABC):
    """Base class for all scorer response types.

    You don't use this directly -- use `BinaryResponse`, `NumericResponse`,
    or a `CategoricalResponse` subclass instead.

    Attributes:
        value: The score (bool, str, or float depending on type).
        reason: Explanation of why this score was given.
        citations: Optional references to specific trace spans.
    """

    value: Union[bool, str, float]
    reason: str
    citations: Optional[List[Citation]] = None
    _return_type: ClassVar[Literal["binary", "categorical", "numeric"]]


@final
class BinaryResponse(BaseResponse):
    """Pass/fail response for binary scorers.

    Attributes:
        value: `True` if the evaluation passed, `False` otherwise.

    Examples:
        ```python
        return BinaryResponse(
            value=True,
            reason="Output correctly answers the question.",
        )
        ```
    """

    value: bool
    _return_type: ClassVar[Literal["binary"]] = "binary"


class CategoricalResponse(BaseResponse, ABC):
    """Response for classification-style scorers.

    Subclass this and define a `categories` class variable listing every
    allowed category. The `value` is validated against this list.

    Attributes:
        value: The selected category label.
        categories: Class-level list of allowed `Category` objects.

    Examples:
        ```python
        class SentimentResponse(CategoricalResponse):
            categories = [
                Category(value="positive", description="Positive sentiment"),
                Category(value="neutral", description="Neutral sentiment"),
                Category(value="negative", description="Negative sentiment"),
            ]

        class SentimentJudge(Judge[SentimentResponse]):
            async def score(self, data: Example) -> SentimentResponse:
                return SentimentResponse(
                    value="positive",
                    reason="The output has an enthusiastic tone.",
                )
        ```
    """

    value: str
    _return_type: ClassVar[Literal["categorical"]] = "categorical"
    categories: ClassVar[List[Category]]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            TypeAdapter(List[Category]).validate_python(cls.categories)
        except (AttributeError, ValidationError) as e:
            raise TypeError(
                f"{cls.__name__} must define a 'categories' class variable "
                f"as a list of Category models"
            ) from e

    @model_validator(mode="after")
    def validate_value_in_categories(self):
        valid_names = [c.value for c in self.categories]
        if self.value not in valid_names:
            raise ValueError(
                f"value '{self.value}' is not a valid category. "
                f"Must be one of: {valid_names}"
            )
        return self


@final
class NumericResponse(BaseResponse):
    """Response for numeric scorers (e.g. 0.0 to 1.0).

    Attributes:
        value: The numeric score.

    Examples:
        ```python
        return NumericResponse(
            value=0.85,
            reason="Output covers most of the expected content.",
        )
        ```
    """

    value: float
    _return_type: ClassVar[Literal["numeric"]] = "numeric"


ScorerResponse = Union[BinaryResponse, CategoricalResponse, NumericResponse]
