from pydantic import BaseModel
from typing import Optional


class CustomScorerResult(BaseModel):
    score: float
    reason: str
    choice: Optional[str] = None
