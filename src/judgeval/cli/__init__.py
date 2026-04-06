#!/usr/bin/env python3

import os
import subprocess
import sys
import typer
import re
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from judgeval.utils import resolve_project_id
from judgeval.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_URL
from judgeval.logger import judgeval_logger
from judgeval.exceptions import JudgmentAPIError
from judgeval.version import get_version
from judgeval.utils.url import url_for
from judgeval.hosted.templates import (
    get_binary_scorer_template,
    get_categorical_scorer_template,
    get_numeric_scorer_template,
)

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

app.add_typer(scorer_app, name="scorer", help="Commands to manage custom scorers")


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
    entrypoint_path: str = typer.Argument(help="Path to scorer entrypoint Python file"),
    project_name: str = typer.Option(..., "--project", "-p", help="Project name"),
    requirements_file_path: str = typer.Option(
        None, "--requirements", "-r", help="Path to requirements.txt file"
    ),
    included_files_paths: list[str] = typer.Option(
        [],
        "--included-files",
        "-i",
        help="Path to included files or directories. If a directory is provided, all non-ignored files in the directory will be included.",
    ),
    unique_name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Custom scorer name (auto-detected if not provided)",
    ),
    bump_major: bool = typer.Option(
        False, "--bump-major", "-m", help="Bump major version"
    ),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Upload custom scorer to Judgment."""
    from judgeval.cli.upload_judge import upload_judge

    scorer_path = Path(entrypoint_path)
    if not scorer_path.exists():
        raise typer.BadParameter(f"Scorer file not found: {entrypoint_path}")

    if not api_key or not organization_id:
        raise typer.BadParameter("JUDGMENT_API_KEY and JUDGMENT_ORG_ID required")

    client = JudgmentSyncClient(JUDGMENT_API_URL, api_key, organization_id)
    project_id = resolve_project_id(client, project_name)
    if not project_id:
        raise typer.BadParameter(f"Project '{project_name}' not found")

    try:
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
        typer.echo(f"Custom scorer uploaded successfully to project '{project_name}'!")
    except JudgmentAPIError as e:
        if e.status_code == 409:
            judgeval_logger.error(e.detail)
            raise typer.Exit(1)
        raise
    except ValueError as e:
        judgeval_logger.error(str(e))
        raise typer.Exit(1)


@scorer_app.command()
def init(
    response_type: Literal["binary", "categorical", "numeric"] = typer.Option(
        ..., "--response-type", "-t", help="Response type"
    ),
    include_requirements: bool = typer.Option(
        False, "--include-requirements", "-r", help="Include requirements.txt file"
    ),
    scorer_name: str = typer.Option(..., "--name", "-n", help="Scorer class name"),
    init_path: str = typer.Option(
        ".", "--init-path", "-p", help="Path to initialize the scorer"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Initialize skeleton code for a new custom scorer."""
    if not scorer_name.isidentifier():
        raise typer.BadParameter("Scorer name must be a valid Python identifier")
    scorer_path = Path(
        init_path, f"{re.sub(r'(?<!^)(?=[A-Z])', '_', scorer_name).lower()}.py"
    )
    if scorer_path.exists():
        raise typer.BadParameter(f"Scorer file already exists: {scorer_name}")

    scorer_path.parent.mkdir(parents=True, exist_ok=True)

    if response_type == "binary":
        template = get_binary_scorer_template(scorer_name)
    elif response_type == "categorical":
        template = get_categorical_scorer_template(scorer_name)
    elif response_type == "numeric":
        template = get_numeric_scorer_template(scorer_name)
    else:
        raise typer.BadParameter(f"Unsupported response type: {response_type}")

    if include_requirements:
        requirements_path = Path(init_path, "requirements.txt")
        if requirements_path.exists():
            raise typer.BadParameter(
                f"Requirements file already exists: {requirements_path}"
            )
        if not yes:
            typer.confirm(
                f"Are you sure you want to initialize an empty requirements file at:\n{os.path.abspath(requirements_path)}?",
                abort=True,
            )
        with open(requirements_path, "w") as f:
            f.write("")
        typer.echo(
            f"Requirements file initialized successfully:\n{os.path.abspath(requirements_path)}"
        )

    if not yes:
        typer.confirm(
            f"Are you sure you want to initialize a {response_type} judge file at:\n{os.path.abspath(scorer_path)}?",
            abort=True,
        )
    with open(scorer_path, "w") as f:
        f.write(template)
    typer.echo(f"Scorer initialized successfully:\n{os.path.abspath(scorer_path)}")


@app.command()
def version():
    """Show Judgeval CLI version."""
    typer.echo(f"Judgeval CLI v{get_version()}")


if __name__ == "__main__":
    app()
