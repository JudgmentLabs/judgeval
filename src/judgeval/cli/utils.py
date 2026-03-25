from pathlib import Path
from pathspec import PathSpec
import os
import tarfile

DEFAULT_EXCLUDE_SPEC = PathSpec.from_lines(
    "gitignore",
    [
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "**/*.pyw",
        "*.pyz",
        ".venv/",
        "venv/",
        ".env",
        ".env.*",
    ],
)


def _find_gitignore_path(start_path: str) -> str | None:
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent
    while current != current.parent:
        if (current / ".gitignore").is_file():
            return str(current / ".gitignore")
        current = current.parent
    return None


class TarFilter:
    def __init__(self, common: str):
        self.common = common
        self.seen_files: set[str] = set()

        self.gitignore_path = _find_gitignore_path(common)
        self.gitignore_spec: PathSpec | None = None
        if self.gitignore_path:
            with open(self.gitignore_path, "r") as f:
                self.gitignore_spec = PathSpec.from_lines("gitignore", f)

    def get_file_count(self) -> int:
        return len(self.seen_files)

    def is_excluded_by_default(self, path: str) -> bool:
        return DEFAULT_EXCLUDE_SPEC.match_file(path)

    def is_excluded_by_gitignore(self, path: str) -> bool:
        if self.gitignore_spec and self.gitignore_path:
            abs_path = os.path.join(self.common, path)
            rel_to_gitignore = os.path.relpath(
                abs_path, os.path.dirname(self.gitignore_path)
            )
            if path.endswith("/"):
                rel_to_gitignore += "/"
            return self.gitignore_spec.match_file(rel_to_gitignore)
        return False

    def __call__(self, tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        normalized = os.path.normpath(tarinfo.name)
        tarinfo.name = normalized
        path_to_check_exclusion = normalized + "/" if tarinfo.isdir() else normalized
        is_excluded = self.is_excluded_by_default(
            path_to_check_exclusion
        ) or self.is_excluded_by_gitignore(path_to_check_exclusion)
        if normalized in self.seen_files or is_excluded:
            return None
        self.seen_files.add(normalized)

        return tarinfo
