from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Template
from typing import Dict, List, Optional


@dataclass
class Prompt:
    name: str
    prompt: str
    created_at: str
    tags: List[str]
    commit_id: str
    parent_commit_id: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    _template: Template = field(init=False, repr=False)

    def __post_init__(self):
        template_str = re.sub(r"\{\{([^}]+)\}\}", r"$\1", self.prompt)
        self._template = Template(template_str)

    def compile(self, **kwargs) -> str:
        try:
            return self._template.substitute(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
