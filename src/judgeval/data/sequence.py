from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Any
from judgeval.data.example import Example
from judgeval.scorers import JudgevalScorer, APIJudgmentScorer
from uuid import uuid4
from datetime import datetime, timezone

class Sequence(BaseModel):
    """
    A sequence is a list of either Examples or nested Sequence objects.
    """
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = "Sequence"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    items: List[Union["Sequence", Example]]
    scorers: Optional[Any] = None
    parent_sequence_id: Optional[str] = None
    sequence_order: Optional[int] = 0
    root_sequence_id: Optional[str] = None
    inputs: Optional[str] = None
    output: Optional[str] = None

    @field_validator("scorers")
    def validate_scorer(cls, v):
        for scorer in v or []:
            if not isinstance(scorer, APIJudgmentScorer) and not isinstance(scorer, JudgevalScorer):
                raise ValueError(f"Invalid scorer type: {type(scorer)}")
        return v
    
    @model_validator(mode="after")
    def populate_sequence_metadata(self) -> "Sequence":
        """Recursively set parent_sequence_id, root_sequence_id, and sequence_order."""
        # If root_sequence_id isn't already set, assign it to self
        if self.root_sequence_id is None:
            self.root_sequence_id = self.sequence_id

        for idx, item in enumerate(self.items):
            item.sequence_order = idx
            if isinstance(item, Sequence):
                item.parent_sequence_id = self.sequence_id
                item.root_sequence_id = self.root_sequence_id
                item.populate_sequence_metadata()
        return self

    class Config:
        arbitrary_types_allowed = True

# Update forward references so that "Sequence" inside items is resolved.
Sequence.model_rebuild()
