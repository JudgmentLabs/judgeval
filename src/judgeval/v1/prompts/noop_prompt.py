from __future__ import annotations

from judgeval.v1.prompts.prompt import Prompt


class NoopPrompt(Prompt):
    """A no-op Prompt that returns empty/placeholder values.

    Used when project_id is not available, allowing code to continue
    without raising exceptions.
    """

    def __init__(self, name: str = ""):
        super().__init__(
            name=name,
            prompt="",
            created_at="",
            tags=[],
            commit_id="",
        )

    def compile(self, **kwargs) -> str:
        return ""
