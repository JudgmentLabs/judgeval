from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from judgeval.internal.api.models import Example as APIExample

if TYPE_CHECKING:
    from judgeval.data.trace import Trace


@dataclass(slots=True)
class Example:
    """A single evaluation example with flexible key-value properties.

    Use `Example.create()` to build examples with any fields your scorers
    need. Common fields include `input`, `actual_output`, `expected_output`,
    and `retrieval_context`, but you can add any custom fields.

    Access properties with bracket notation: `example["input"]`.

    Examples:
        Create an example for evaluating a Q&A system:

        ```python
        example = Example.create(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            expected_output="Paris",
        )
        ```

        Include retrieval context for RAG evaluation:

        ```python
        example = Example.create(
            input="What is Python?",
            actual_output="A programming language.",
            retrieval_context=["Python is a high-level programming language..."],
        )
        ```

        Access properties:

        ```python
        print(example["input"])      # "What is Python?"
        print("input" in example)    # True
        ```
    """

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
        """Create an example with the given properties.

        Args:
            **kwargs: Any key-value pairs. Common keys:
                `input`, `actual_output`, `expected_output`,
                `retrieval_context`, `context`.

        Returns:
            A new `Example` instance.
        """
        example = cls()
        for key, value in kwargs.items():
            example._properties[key] = value
        return example

    def to_dict(self) -> APIExample:
        """Serialize the example to a dictionary suitable for the API."""
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
        """Return a copy of the example's custom properties."""
        return self._properties.copy()
