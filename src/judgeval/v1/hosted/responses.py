from typing import ClassVar, List, Literal, Optional, Union

from pydantic import BaseModel


ReturnType = Literal["binary", "categorical", "numeric"]


class Citation(BaseModel):
    content: str
    source: Optional[str] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None


class BinaryResponse(BaseModel):
    value: bool
    reason: str
    citations: Optional[List[Citation]] = None
    _return_type: ClassVar[Literal["binary"]] = "binary"


class CategoricalResponse(BaseModel):
    value: str
    reason: str
    citations: Optional[List[Citation]] = None
    _return_type: ClassVar[Literal["categorical"]] = "categorical"


class NumericResponse(BaseModel):
    value: float
    reason: str
    citations: Optional[List[Citation]] = None
    _return_type: ClassVar[Literal["numeric"]] = "numeric"


ScorerResponse = Union[BinaryResponse, CategoricalResponse, NumericResponse]
