from pydantic import BaseModel, Field
from typing import List, Optional
from judgeval.data.example import Example
from uuid import uuid4
from datetime import datetime

class Sequence(BaseModel):
    """
    A sequence is a list of either examples of nested sequence objects.
    """
    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    items: List[Example]

