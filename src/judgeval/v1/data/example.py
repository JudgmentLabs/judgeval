from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from judgeval.v1.internal.api.api_types import Example as APIExample

if TYPE_CHECKING:
    from judgeval.v1.data.trace import Trace


# Example.property => example.input, example.output (type = Any), Example.trace (type Trace) => Example.propery, Example.golden (type = Trace)
# Example.golden = TraceItem(trace_id = "90")


# Agent Test Judge !+ Behavior

# Evaluate differencves between {golden} and {actual} (golden trace, actual trace)


# run_agent_testing(agent_fn, mapper, scorer,...)
# call agent_fn() => post_process() (upload new Example(**OldExample, new_trace))


# context manager
#    with create_experiment(dataset, experiment_name) as Experiment:
#      for example in Experiment:
#         trace_id = agent(example)
#         Exoeriment.log_example(newTrace(trace_id), oldTrace, xyz)
#


# Input Dataset (schema), Output Dataset (schema)


@dataclass(slots=True)
class Example:
    example_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    name: Optional[str] = None
    _properties: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[Trace] = None

    def __getitem__(self, key: str) -> Any:
        return self._properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._properties[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self._properties

    def set_property(self, key: str, value: Any) -> Example:
        # priviledge: trace, golden_trace
        # run_agent_testing, Upload (Example)
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
