from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from judgeval.v1.internal.api.api_types import Example as APIExample


@dataclass(slots=True)
class Example:
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    name: Optional[str] = None
    _properties: Dict[str, Any] = field(default_factory=dict)

    def set_property(self, key: str, value: Any) -> Example:
        self._properties[key] = value
        return self

    def get_property(self, key: str) -> Any:
        return self._properties.get(key)

    @classmethod
    def create(cls, **kwargs: Any) -> Example:
        example = cls()
        for key, value in kwargs.items():
            example.set_property(key, value)
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
