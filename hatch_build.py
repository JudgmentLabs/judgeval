from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Regenerate deterministic JQL artifacts before building a distribution."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        """Regenerate JQL artifacts when building from a repository checkout.

        Skips generation when building from an sdist, which has neither a .git
        directory nor the private build-time inputs required by generate_jql.py.

        Args:
            version: The version string supplied by Hatch for the current build.
            build_data: Mutable mapping that Hatch uses to collect extra build
                metadata (e.g. additional artifacts and forced inclusion paths).
        """
        # Generated files are checked in. Regenerate only in a repository checkout;
        # an sdist installation has neither .git nor the private build-time inputs.
        if not (Path(self.root) / ".git").exists():
            return
        subprocess.run(
            [sys.executable, str(Path(self.root) / "scripts" / "generate_jql.py")],
            cwd=self.root,
            check=True,
        )
