from pydantic import BaseModel


class CustomScorerResult(BaseModel):
    score: float
    reason: str
