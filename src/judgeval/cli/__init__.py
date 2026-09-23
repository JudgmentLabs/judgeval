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

tests_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
    rich_help_panel=None,
    rich_markup_mode=None,
)

app.add_typer(
    tests_app,
    name="tests",
    help="Run your agent for offline test runs started from the platform",
)


def _load_agent_function(spec: str):
    """Resolve ``path/to/file.py:function`` or ``package.module:function``."""
    import importlib
    import importlib.util

    if ":" not in spec:
        raise typer.BadParameter(
            "Expected --agent as 'path/to/agent.py:function' or 'module.path:function'"
        )
    module_spec, function_name = spec.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    module_path = Path(module_spec)
    if module_path.suffix == ".py" and module_path.exists():
        sys.path.insert(0, str(module_path.resolve().parent))
        loaded = importlib.util.spec_from_file_location(
            module_path.stem, module_path.resolve()
        )
        if loaded is None or loaded.loader is None:
            raise typer.BadParameter(f"Could not load {module_spec}")
        module = importlib.util.module_from_spec(loaded)
        sys.modules[module_path.stem] = module
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)
    agent_function = getattr(module, function_name, None)
    if agent_function is None or not callable(agent_function):
        raise typer.BadParameter(
            f"{function_name!r} is not a callable in {module_spec}"
        )
    return agent_function


def _parse_field_mapping(pairs: list[str]) -> dict[str, str] | None:
    if not pairs:
        return None
    mapping: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(
                f"--map expects agent_param=dataset_field, got {pair!r}"
            )
        param, field = pair.split("=", 1)
        mapping[param.strip()] = field.strip()
    return mapping


def _offline_tests(project_name: str, api_key: str, organization_id: str):
    from judgeval import Judgeval

    if not api_key or not organization_id:
        raise typer.BadParameter("JUDGMENT_API_KEY and JUDGMENT_ORG_ID required")
    return Judgeval(
        project_name=project_name, api_key=api_key, organization_id=organization_id
    ).offline_tests


@tests_app.command()
def attach(
    test_run_id: str = typer.Argument(help="Test run id shown in the platform"),
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent entrypoint: path/to/agent.py:function"
    ),
    project_name: str = typer.Option(
        ..., "--project", "-p", envvar="JUDGMENT_PROJECT", help="Project name"
    ),
    field_mapping: list[str] = typer.Option(
        [], "--map", help="agent_param=dataset_field (repeatable)"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Return once traces are attached, before judging"
    ),
    timeout_seconds: int = typer.Option(600, "--timeout", help="Seconds to wait"),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
):
    """Run your agent for a test run that is waiting for traces."""
    agent_function = _load_agent_function(agent)
    offline_tests = _offline_tests(project_name, api_key, organization_id)
    result = offline_tests.attach(
        test_run_id,
        agent_function,
        field_mapping=_parse_field_mapping(field_mapping),
        timeout_seconds=timeout_seconds,
        wait=not no_wait,
    )
    if result is None:
        raise typer.Exit(code=1)
    if result.ui_results_url:
        typer.echo(result.ui_results_url)


@tests_app.command()
def serve(
    agent: str = typer.Option(
        ..., "--agent", "-a", help="Agent entrypoint: path/to/agent.py:function"
    ),
    project_name: str = typer.Option(
        ..., "--project", "-p", envvar="JUDGMENT_PROJECT", help="Project name"
    ),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8787, "--port"),
    path: str = typer.Option("/judgment/run", "--path"),
    secret: str = typer.Option(
        None, "--secret", envvar="JUDGMENT_AGENT_TARGET_SECRET",
        help="Shared secret the platform sends as a bearer token",
    ),
    field_mapping: list[str] = typer.Option(
        [], "--map", help="agent_param=dataset_field (repeatable)"
    ),
    api_key: str = typer.Option(None, envvar="JUDGMENT_API_KEY"),
    organization_id: str = typer.Option(None, envvar="JUDGMENT_ORG_ID"),
):
    """Serve your agent as an endpoint the platform can dispatch runs to."""
    agent_function = _load_agent_function(agent)
    offline_tests = _offline_tests(project_name, api_key, organization_id)
    typer.echo(
        f"Serving agent at http://{'localhost' if host == '0.0.0.0' else host}:{port}{path}"
    )
    offline_tests.serve(
        agent_function,
        host=host,
        port=port,
        path=path,
        secret=secret,
        field_mapping=_parse_field_mapping(field_mapping),
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
