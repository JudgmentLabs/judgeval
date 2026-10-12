from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from judgeval.cli import app
from judgeval.exceptions import JudgmentAPIError

runner = CliRunner()
ENV = {"JUDGMENT_API_KEY": "key", "JUDGMENT_ORG_ID": "org"}
SQL_RESULT = {
    "catalog_version": "v1",
    "columns": [{"name": "n", "type": "UInt64", "nullable": False}],
    "rows": [{"n": 3}],
    "row_count": 1,
    "elapsed_ms": 1.5,
}


def test_sql_prints_json():
    with (
        patch("judgeval.judgeval.resolve_project_id", return_value="proj-1"),
        patch(
            "judgeval.internal.api.JudgmentSyncClient._request",
            return_value=SQL_RESULT,
        ) as request,
    ):
        result = runner.invoke(
            app, ["sql", "-p", "my-project", "SELECT count() AS n"], env=ENV
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == SQL_RESULT
    method, url, payload = request.call_args.args
    assert (method, payload) == ("POST", {"sql": "SELECT count() AS n"})
    assert url.endswith("/v1/projects/proj-1/sql")


def test_sql_api_error_exits_nonzero_with_server_message():
    error = JudgmentAPIError(
        403, "Public SQL is not enabled", None, hint="Ask an admin to opt in."
    )
    with (
        patch("judgeval.judgeval.resolve_project_id", return_value="proj-1"),
        patch("judgeval.internal.api.JudgmentSyncClient._request", side_effect=error),
    ):
        result = runner.invoke(app, ["sql", "-p", "my-project", "SELECT 1"], env=ENV)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        "Error: 403: Public SQL is not enabled\nHint: Ask an admin to opt in.\n"
    )


def test_sql_unknown_project_exits_nonzero():
    with patch("judgeval.judgeval.resolve_project_id", return_value=None):
        result = runner.invoke(app, ["sql", "-p", "missing", "SELECT 1"], env=ENV)
    assert result.exit_code == 1
    assert "Project 'missing' was not found" in result.stderr


def test_schema_prints_markdown():
    with patch(
        "judgeval.internal.api.JudgmentSyncClient._request",
        return_value={"schema": "# Schema"},
    ) as request:
        result = runner.invoke(app, ["schema"], env=ENV)
    assert result.exit_code == 0, result.output
    assert result.stdout == "# Schema\n"
    method, url, _ = request.call_args.args
    assert method == "GET" and url.endswith("/v1/sql/schema")
