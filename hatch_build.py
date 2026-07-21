from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Regenerate deterministic JQL artifacts before building a distribution."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        subprocess.run(
            [sys.executable, str(Path(self.root) / "scripts" / "generate_jql.py")],
            cwd=self.root,
            check=True,
        )
