from typing import ClassVar, List, Literal, Optional, Union
from abc import ABC
from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator
from typing_extensions import final

ReturnType = Literal["binary", "categorical", "numeric"]


class Citation(BaseModel):
    span_id: str
    span_attribute: str


class Category(BaseModel):
    value: str
    description: str = ""


class BaseResponse(BaseModel, ABC):
    value: Union[bool, str, float]
    reason: str
    citations: Optional[List[Citation]] = None
    _return_type: ClassVar[Literal["binary", "categorical", "numeric"]]


@final
class BinaryResponse(BaseResponse):
    value: bool
    _return_type: ClassVar[Literal["binary"]] = "binary"


class CategoricalResponse(BaseResponse, ABC):
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
    value: float
    _return_type: ClassVar[Literal["numeric"]] = "numeric"


ScorerResponse = Union[BinaryResponse, CategoricalResponse, NumericResponse]
