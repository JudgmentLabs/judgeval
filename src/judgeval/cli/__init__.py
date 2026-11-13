#!/usr/bin/env python3

import os
import subprocess
import sys
import typer
import re
from pathlib import Path
import json
from typing import Optional
import shlex
from dotenv import load_dotenv
from judgeval.utils import resolve_project_id
from judgeval.internal.api import JudgmentSyncClient
from judgeval.api import JudgmentApiClient
from judgeval.env import JUDGMENT_API_URL
from judgeval.logger import judgeval_logger
from judgeval.exceptions import JudgmentAPIError
from judgeval.version import get_version
from judgeval.utils.url import url_for
from judgeval.cli.function_judge import OutputType

load_dotenv()

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
    rich_help_panel=None,
    rich_markup_mode=None,
)

scorer_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
    rich_help_panel=None,
    rich_markup_mode=None,
)

app.add_typer(scorer_app, name="scorer", help="Commands to manage Code judges")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def load_otel_env(
    ctx: typer.Context,
    project_name: str = typer.Argument(help="Project name to send telemetry to"),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
):
    """Run command with OpenTelemetry environment variables configured for Judgment."""
    if not api_key or not organization_id:
        raise typer.BadParameter("JUDGMENT_API_KEY and JUDGMENT_ORG_ID required")

    client = JudgmentSyncClient(JUDGMENT_API_URL, api_key, organization_id)
    project_id = resolve_project_id(client, project_name)
    if not project_id:
        raise typer.BadParameter(f"Project '{project_name}' not found")

    if not ctx.args:
        raise typer.BadParameter(
            "No command provided. Usage: judgeval load_otel_env PROJECT_NAME -- COMMAND"
        )

    env = os.environ.copy()
    env["OTEL_TRACES_EXPORTER"] = "otlp"
    env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
    env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = url_for("/otel/v1/traces")
    env["OTEL_EXPORTER_OTLP_HEADERS"] = (
        f"Authorization=Bearer {api_key},X-Organization-Id={organization_id},X-Project-Id={project_id}"
    )

    result = subprocess.run(ctx.args, env=env)
    sys.exit(result.returncode)


@scorer_app.command()
def upload(
    entrypoint_path: str = typer.Argument(
        help="Path to the judge file: a function judge, or a class judge's entrypoint"
    ),
    project_name: str = typer.Option(..., "--project", "-p", help="Project name"),
    categories: list[str] = typer.Option(
        [],
        "--category",
        "-c",
        help="A label a new function judge can return; repeat for each (at least two)",
    ),
    requirements_file_path: str = typer.Option(
        None, "--requirements", "-r", help="Class judges only: requirements.txt"
    ),
    included_files_paths: list[str] = typer.Option(
        [],
        "--included-files",
        "-i",
        help="Class judges only: extra files or directories to bundle.",
    ),
    unique_name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Judge name (defaults to the file name for a function judge, the class name for a class judge)",
    ),
    bump_major: bool = typer.Option(
        False, "--bump-major", "-m", help="Bump major version"
    ),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Upload a Code judge to Judgment."""
    from judgeval.cli.function_judge import is_function_judge, upload_function_judge
    from judgeval.cli.upload_judge import upload_judge

    scorer_path = Path(entrypoint_path)
    if not scorer_path.exists():
        raise typer.BadParameter(f"Code judge file not found: {entrypoint_path}")

    if not api_key or not organization_id:
        raise typer.BadParameter("JUDGMENT_API_KEY and JUDGMENT_ORG_ID required")

    source = scorer_path.read_text()
    try:
        function_judge = is_function_judge(source)
    except SyntaxError as e:
        raise typer.BadParameter(f"{entrypoint_path}:{e.lineno}: {e.msg}")
    if function_judge and (requirements_file_path or included_files_paths):
        raise typer.BadParameter(
            "A function judge is one file; --requirements and --included-files are for class judges"
        )

    client = JudgmentApiClient(JUDGMENT_API_URL, api_key, organization_id)
    project_id = resolve_project_id(client, project_name)
    if not project_id:
        raise typer.BadParameter(f"Project '{project_name}' not found")

    try:
        if function_judge:
            name = unique_name or scorer_path.stem
            if not yes:
                typer.confirm(
                    f"Upload function judge '{name}' to project '{project_name}'? "
                    "If it exists, a new production version is created.",
                    abort=True,
                )
            typer.echo(
                upload_function_judge(
                    client, project_id, name, source, categories, bump_major
                )
            )
            return
        judgeval_logger.warning(
            "Class judges are deprecated. Rewrite this one as a function judge, "
            "`def judge(example)` in one file; run `judgeval scorer init` for an example."
        )
        result = upload_judge(
            client=client,
            project_id=project_id,
            entrypoint_path=entrypoint_path,
            included_files_paths=included_files_paths,
            requirements_file_path=requirements_file_path,
            unique_name=unique_name,
            bump_major=bump_major,
            project_name=project_name,
            yes=yes,
        )
        if not result:
            raise typer.Abort()
        typer.echo(f"Code judge uploaded successfully to project '{project_name}'!")
    except JudgmentAPIError as e:
        if e.status_code in (400, 409, 422):
            judgeval_logger.error(e.detail)
            raise typer.Exit(1)
        raise
    except ValueError as e:
        judgeval_logger.error(str(e))
        raise typer.Exit(1)


_SAMPLE_TRACE = [
    {
        "span_id": "root",
        "parent_span_id": None,
        "span_name": "agent",
        "span_kind": "agent",
        "span_attributes": {},
        "duration": 1200000000,
    },
    {
        "span_id": "search",
        "parent_span_id": "root",
        "span_name": "search_flights",
        "span_kind": "tool",
        "span_attributes": {"judgment.output": "3 flights found"},
        "duration": 300000000,
    },
]


def _function_judge_template(passed: str, failed: str, test_command: str) -> str:
    return f'''def judge(example):
    """Passes when the agent called at least one tool.

    `example` holds the example's fields; a trace's spans are in
    example["trace"]["spans"]. Return a label, or a dict with "value" and
    optional "reason", "citations" and "metrics".

    Runs in Monty: plain Python, no third-party packages, no class
    inheritance, no `del`. Test it in Monty before uploading:
        {test_command}
    """
    tools = [s for s in example["trace"]["spans"] if s["span_kind"] == "tool"]
    if not tools:
        return {{"value": {json.dumps(failed)}, "reason": "The agent called no tools"}}
    return {{
        "value": {json.dumps(passed)},
        "reason": "The agent called " + tools[0]["span_name"],
        "citations": [{{"span_id": tools[0]["span_id"], "span_attribute": "span_name"}}],
        "metrics": {{"tool_calls": len(tools)}},
    }}
'''


@scorer_app.command()
def init(
    name: str = typer.Option(
        ..., "--name", "-n", help="Judge name, used for the file name"
    ),
    categories: list[str] = typer.Option(
        ["Pass", "Fail"],
        "--category",
        "-c",
        help="A label the judge returns; repeat for each. The first is the pass label.",
    ),
    init_path: str = typer.Option(
        ".", "--init-path", "-p", help="Directory to write the judge into"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Write a function judge and a sample trace to test it on."""
    if len(categories) < 2:
        raise typer.BadParameter("Give at least two categories")
    stem = re.sub(r"\W+", "_", re.sub(r"(?<!^)(?=[A-Z])", "_", name)).strip("_").lower()
    judge_path = Path(init_path, f"{stem}.py")
    trace_path = Path(init_path, f"{stem}_trace.json")
    for path in (judge_path, trace_path):
        if path.exists():
            raise typer.BadParameter(f"File already exists: {path}")

    category_flags = " ".join(f"-c {shlex.quote(c)}" for c in categories)
    test_command = f"judgeval scorer test {judge_path.name} --trace {trace_path.name} {category_flags}"
    if not yes:
        typer.confirm(
            f"Write a function judge to {judge_path.resolve()} and a sample trace to {trace_path.resolve()}?",
            abort=True,
        )
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path.write_text(
        _function_judge_template(categories[0], categories[-1], test_command)
    )
    trace_path.write_text(json.dumps(_SAMPLE_TRACE, indent=2) + "\n")
    typer.echo(
        f"Wrote {judge_path} and {trace_path}. Test it in Monty:\n  {test_command}"
    )


@scorer_app.command()
def test(
    judge_path: Path = typer.Argument(
        exists=True, dir_okay=False, help="Function judge file defining judge(example)"
    ),
    trace_path: Path = typer.Option(
        None,
        "--trace",
        "-t",
        exists=True,
        dir_okay=False,
        help="Saved trace: a JSON list of spans, {'spans': [...]}, or NDJSON",
    ),
    example_path: Path = typer.Option(
        None,
        "--example",
        "-e",
        exists=True,
        dir_okay=False,
        help="JSON object of example fields",
    ),
    output: Optional[OutputType] = typer.Option(
        None,
        "--output",
        "-o",
        help="Defaults to categorical when --category is given",
    ),
    categories: list[str] = typer.Option(
        [], "--category", "-c", help="A label the judge may return; repeat for each"
    ),
):
    """Run a function judge in Monty, as Judgment does, and check its result."""
    from judgeval.cli.function_judge import (
        JudgeFailed,
        check_result,
        load_spans,
        run_judge,
        source_problems,
    )

    output = output or ("categorical" if categories else None)
    if output is None:
        raise typer.BadParameter("Give --output, or --category for a categorical judge")
    if output == "categorical" and len(categories) < 2:
        raise typer.BadParameter("A categorical judge needs at least two --category")
    if trace_path is None and example_path is None:
        raise typer.BadParameter("Give --trace, --example, or both")

    source = judge_path.read_text()
    try:
        example = json.loads(example_path.read_text()) if example_path else {}
        if trace_path:
            example["trace"] = {"spans": load_spans(trace_path)}
        problems = source_problems(source)
        if problems:
            typer.echo(
                f"Judgment would reject this judge at upload:\n{problems}", err=True
            )
            raise typer.Exit(1)
        result = check_result(run_judge(source, example), output, categories)
    except (JudgeFailed, RuntimeError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def version():
    """Show Judgeval CLI version."""
    typer.echo(f"Judgeval CLI v{get_version()}")


if __name__ == "__main__":
    app()
