from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Union, Any
from judgeval.data.example import Example
from judgeval.scorers import ScorerWrapper, JudgevalScorer
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

    @field_validator("scorers")
    def validate_scorer(cls, v):
        loaded_scorers = []
        for scorer in v or []:
            try:
                if isinstance(scorer, ScorerWrapper):
                    loaded_scorers.append(scorer.load_implementation())
                else:
                    loaded_scorers.append(scorer)
            except Exception as e:
                raise ValueError(f"Failed to load implementation for scorer {scorer}: {str(e)}")
        return loaded_scorers
    
    @model_validator(mode='after')
    def set_parent_sequence_ids(self) -> "Sequence":
        """Recursively set the parent_sequence_id for all nested Sequences."""
        for item in self.items:
            if isinstance(item, Sequence):
                item.parent_sequence_id = self.sequence_id
                # Recurse into deeper nested sequences
                item.set_parent_sequence_ids()  
        return self

    @model_validator(mode='after')
    def set_parent_and_order(self) -> "Sequence":
        """Set parent_sequence_id and sequence_order for all items."""
        for idx, item in enumerate(self.items):
            # Set sequence_order for both Example and Sequence objects
            item.sequence_order = idx
            
            if isinstance(item, Sequence):
                item.parent_sequence_id = self.sequence_id
                item.set_parent_and_order()  # Recurse for nested sequences
        return self
    
    class Config:
        arbitrary_types_allowed = True

# Update forward references so that "Sequence" inside items is resolved.
Sequence.model_rebuild()
