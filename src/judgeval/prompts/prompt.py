from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Template
from typing import Dict, List, Optional


@dataclass
class Prompt:
    """A versioned prompt template with variable substitution.

    Prompts are stored and versioned on the Judgment platform. Use
    `{{variable}}` placeholders in the template, then call `.compile()`
    to substitute values at runtime. Each version gets a unique `commit_id`
    and can be tagged (e.g. `"production"`, `"staging"`).

    Attributes:
        name: Prompt name.
        prompt: The template string with `{{variable}}` placeholders.
        created_at: ISO-8601 creation timestamp.
        tags: Tags on this version (e.g. `["production"]`).
        commit_id: Unique identifier for this version.
        parent_commit_id: The previous version's commit ID.
        metadata: Creator information.

    Examples:
        ```python
        prompt = client.prompts.get(name="system-prompt", tag="production")
        text = prompt.compile(product="Acme Search", tone="friendly")
        print(text)
        ```
    """

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
        """Fill in template variables and return the final prompt string.

        Args:
            **kwargs: Values for each `{{variable}}` in the template.

        Returns:
            The prompt with all variables substituted.

        Raises:
            ValueError: If a required variable is not provided.

        Examples:
            ```python
            prompt = client.prompts.get(name="system-prompt", tag="production")
            # Template: "You are a {{tone}} assistant for {{product}}."
            text = prompt.compile(tone="helpful", product="Acme Search")
            # "You are a helpful assistant for Acme Search."
            ```
        """
        try:
            return self._template.substitute(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'")
            raise ValueError(f"Missing required variable: {missing_var}")
