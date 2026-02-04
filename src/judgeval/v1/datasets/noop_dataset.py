from __future__ import annotations

from typing import Iterable, Literal, Optional

from judgeval.v1.data.example import Example
from judgeval.v1.datasets.dataset import Dataset


class NoopDataset(Dataset):
    """A no-op Dataset that silently skips all operations.

    Used when project_id is not available, allowing code to continue
    without raising exceptions.
    """

    def __init__(self, name: str = "", project_name: str = ""):
        super().__init__(
            name=name,
            project_id="",
            project_name=project_name,
            examples=[],
            client=None,
        )

    def add_from_json(self, file_path: str, batch_size: int = 100) -> None:
        pass

    def add_from_yaml(self, file_path: str, batch_size: int = 100) -> None:
        pass

    def add_examples(self, examples: Iterable[Example], batch_size: int = 100) -> None:
        pass

    def save_as(
        self,
        file_type: Literal["json", "yaml"],
        dir_path: str,
        save_name: Optional[str] = None,
    ) -> None:
        pass

    def __str__(self):
        return f"NoopDataset(name={self.name})"
