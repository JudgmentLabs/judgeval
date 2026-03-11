from typing import ClassVar, List, Literal, Optional, Union
from abc import ABC
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import final

ReturnType = Literal["binary", "categorical", "numeric"]


class Citation(BaseModel):
    span_id: str
    span_attribute: str


class Category(BaseModel):
    name: str
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


@final
class NumericResponse(BaseResponse):
    value: float
    _return_type: ClassVar[Literal["numeric"]] = "numeric"


ScorerResponse = Union[BinaryResponse, CategoricalResponse, NumericResponse]
