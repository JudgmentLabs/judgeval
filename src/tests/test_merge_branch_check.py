import os
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize(
    ("base", "source", "expected_code", "expected_message"),
    [
        (
            "staging",
            "feature/query",
            0,
            "Skipping branch validation - not targeting main branch",
        ),
        ("main", "staging", 0, "Branch validation passed. Source branch: staging"),
        (
            "main",
            "hotfix/query",
            0,
            "Branch validation passed. Source branch: hotfix/query",
        ),
        (
            "main",
            "codex/public-virtual-sql",
            0,
            "Branch validation passed. Source branch: codex/public-virtual-sql",
        ),
        (
            "main",
            "codex/other",
            1,
            "::error::Pull requests to main require 'staging', 'hotfix/*', or the approved 'codex/public-virtual-sql' release branch. Current branch: codex/other",
        ),
        (
            "main",
            "codex/public-virtual-sql-other",
            1,
            "::error::Pull requests to main require 'staging', 'hotfix/*', or the approved 'codex/public-virtual-sql' release branch. Current branch: codex/public-virtual-sql-other",
        ),
    ],
)
def test_merge_branch_policy(base, source, expected_code, expected_message):
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/merge-branch-check.yaml"
    )
    script = dedent(workflow.read_text().split("        run: |\n", 1)[1])
    result = subprocess.run(
        ["bash", "-e", "-c", script],
        env={**os.environ, "BASE_BRANCH": base, "SOURCE_BRANCH": source},
        capture_output=True,
        text=True,
    )
    assert (result.returncode, result.stdout, result.stderr) == (
        expected_code,
        f"BASE_BRANCH: {base}\nSOURCE_BRANCH: {source}\n{expected_message}\n",
        "",
    )
