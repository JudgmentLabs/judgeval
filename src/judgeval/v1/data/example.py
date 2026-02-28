from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from judgeval.v1.internal.api.api_types import Example as APIExample

if TYPE_CHECKING:
    from judgeval.v1.data.trace import Trace


@dataclass(slots=True)
class Example:
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    name: Optional[str] = None
    _properties: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Trace] = None

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]

    def __contains__(self, key: object) -> bool:
        return key in self._properties

    @classmethod
    def create(cls, **kwargs: Any) -> Example:
        example = cls()
        for key, value in kwargs.items():
            example._properties[key] = value
        return example

    def to_dict(self) -> APIExample:
        result: APIExample = {
            "example_id": self.example_id,
            "created_at": self.created_at,
            "name": self.name,
        }
        for key, value in self._properties.items():
            result[key] = value  # type: ignore[literal-required]
        return result

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties.copy()
