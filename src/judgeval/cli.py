#!/usr/bin/env python3

import os
import subprocess
import sys
import typer
from pathlib import Path
from dotenv import load_dotenv
from judgeval.v1.utils import resolve_project_id
from judgeval.v1.internal.api import JudgmentSyncClient
from judgeval.env import JUDGMENT_API_URL
from judgeval.logger import judgeval_logger
from judgeval.exceptions import JudgmentAPIError
from judgeval import Judgeval
from judgeval.version import get_version
from judgeval.utils.url import url_for

load_dotenv()

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
    rich_help_panel=None,
    rich_markup_mode=None,
)


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


@app.command()
def upload_scorer(
    scorer_file_path: str = typer.Argument(help="Path to scorer Python file"),
    requirements_file_path: str = typer.Argument(help="Path to requirements.txt file"),
    project_name: str = typer.Option(..., "--project", "-p", help="Project name"),
    unique_name: str = typer.Option(
        None, help="Custom scorer name (auto-detected if not provided)"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-o", help="Overwrite if exists"
    ),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
):
    """Upload custom scorer to Judgment."""
    scorer_path = Path(scorer_file_path)
    requirements_path = Path(requirements_file_path)

    if not scorer_path.exists():
        raise typer.BadParameter(f"Scorer file not found: {scorer_file_path}")
    if not requirements_path.exists():
        raise typer.BadParameter(
            f"Requirements file not found: {requirements_file_path}"
        )

    client = Judgeval(project_name, api_key=api_key, organization_id=organization_id)

    try:
        result = client.scorers.custom_scorer.upload(
            scorer_file_path=scorer_file_path,
            requirements_file_path=requirements_file_path,
            unique_name=unique_name,
            overwrite=overwrite,
        )
        if not result:
            raise typer.Abort()
        judgeval_logger.info("Custom scorer uploaded successfully!")
    except JudgmentAPIError as e:
        if e.status_code == 409:
            judgeval_logger.error("Scorer exists. Use --overwrite to replace")
            raise typer.Exit(1)
        raise


@app.command()
def version():
    """Show Judgeval CLI version."""
    typer.echo(f"Judgeval CLI v{get_version()}")


if __name__ == "__main__":
    app()
