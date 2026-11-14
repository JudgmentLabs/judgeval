from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judgeval.cli import app
from judgeval.api import JudgmentApiClient

AUTH = {"JUDGMENT_API_KEY": "api-key", "JUDGMENT_ORG_ID": "org-1"}


def test_a_scaffolded_judge_passes_its_test_in_monty(tmp_path: Path) -> None:
    runner = CliRunner()
    init = runner.invoke(
        app, ["scorer", "init", "-n", "ToolUse", "-p", str(tmp_path), "-y"]
    )
    assert init.exit_code == 0, init.output

    result = runner.invoke(
        app,
        [
            "scorer",
            "test",
            str(tmp_path / "tool_use.py"),
            "--trace",
            str(tmp_path / "tool_use_trace.json"),
            "-c",
            "Pass",
            "-c",
            "Fail",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "value": "Pass",
        "reason": "The agent called search_flights",
        "citations": [{"span_id": "search", "span_attribute": "span_name"}],
        "metrics": {"tool_calls": 1},
    }


def test_the_local_test_rejects_code_monty_cannot_run(tmp_path: Path) -> None:
    judge = tmp_path / "judge.py"
    judge.write_text(
        "class Base:\n    pass\n\nclass Child(Base):\n    pass\n\n"
        "def judge(example):\n    return True\n"
    )
    trace = tmp_path / "trace.json"
    trace.write_text("[]")

    result = CliRunner().invoke(
        app, ["scorer", "test", str(judge), "--trace", str(trace), "-o", "binary"]
    )

    assert result.exit_code == 1
    assert result.output.splitlines()[0] == (
        "Judgment would reject this judge at upload:"
    )


def test_a_new_function_judge_uploads_as_a_categorical_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = 'def judge(example):\n    return "Pass"\n'
    judge = tmp_path / "always_pass.py"
    judge.write_text(source)
    calls = []

    def fake_request(self, method, url, payload, params=None):  # type: ignore[no-untyped-def]
        calls.append((method, url, payload))
        return {"judges": []} if method == "GET" else {}

    monkeypatch.setattr("judgeval.cli.resolve_project_id", lambda *_: "project-1")
    monkeypatch.setattr(JudgmentApiClient, "_request", fake_request)

    result = CliRunner().invoke(
        app,
        [
            "scorer",
            "upload",
            str(judge),
            "-p",
            "demo",
            "-c",
            "Pass",
            "-c",
            "Fail",
            "-y",
        ],
        env=AUTH,
    )

    assert result.exit_code == 0, result.output
    judges_url = calls[0][1]
    assert judges_url.endswith("/v1/projects/project-1/judges")
    assert calls == [
        ("GET", judges_url, {}),
        (
            "POST",
            judges_url,
            {
                "judge": {
                    "name": "always_pass",
                    "judgeType": "custom",
                    "initialVersion": {
                        "scoreType": "categorical",
                        "outputs": [
                            {"name": "Pass", "description": ""},
                            {"name": "Fail", "description": ""},
                        ],
                    },
                    "config": {"functionSource": source},
                }
            },
        ),
    ]


def test_a_class_judge_upload_warns_that_class_judges_are_deprecated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    judge = tmp_path / "old_judge.py"
    judge.write_text(
        "from judgeval.judges import Judge\n"
        "from judgeval.hosted.responses import BinaryResponse\n\n"
        "class OldJudge(Judge[BinaryResponse]):\n"
        "    async def score(self, data):\n"
        "        return BinaryResponse(value=True, reason='ok')\n"
    )
    monkeypatch.setattr("judgeval.cli.resolve_project_id", lambda *_: "project-1")
    monkeypatch.setattr(
        JudgmentApiClient,
        "post_projects_scorers_custom_bundle",
        lambda self, project_id, payload: {"status": "success"},
    )
    monkeypatch.chdir(tmp_path)

    with caplog.at_level(logging.WARNING, logger="judgeval"):
        result = CliRunner().invoke(
            app, ["scorer", "upload", str(judge), "-p", "demo", "-y"], env=AUTH
        )

    assert result.exit_code == 0, result.output
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == [
        "Class judges are deprecated. Rewrite this one as a function judge, "
        "`def judge(example)` in one file; run `judgeval scorer init` for an example."
    ]
