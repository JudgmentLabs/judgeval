#!/usr/bin/env python3

import typer
from pathlib import Path
from dotenv import load_dotenv
from judgeval.common.logger import judgeval_logger
from judgeval.judgment_client import JudgmentClient

load_dotenv()

app = typer.Typer(no_args_is_help=True)


@app.command("upload_scorer")
def upload_scorer(
    unique_name: str,
    scorer_file_path: str,
    requirements_file_path: str,
):
    # Validate file paths
    if not Path(scorer_file_path).exists():
        judgeval_logger.error(f"Scorer file not found: {scorer_file_path}")
        return

    if not Path(requirements_file_path).exists():
        judgeval_logger.error(f"Requirements file not found: {requirements_file_path}")
        return

    try:
        client = JudgmentClient()

        result = client.save_custom_scorer(
            unique_name=unique_name,
            scorer_file_path=scorer_file_path,
            requirements_file_path=requirements_file_path,
        )

        # Check status and return
        status = (
            getattr(result, "status", None) or result.get("status")
            if isinstance(result, dict)
            else None
        )
        return status == "success"

    except Exception:
        raise


@app.command()
def version():
    """Show version info"""
    judgeval_logger.info("JudgEval CLI v0.0.0")


if __name__ == "__main__":
    app()

# judgeval upload_scorer profile_score12 /Users/alanzhang/repo/JudgmentLabs/judgeval/src/demo/c1.py /Users/alanzhang/repo/JudgmentLabs/judgeval/src/demo/requirements.txt
