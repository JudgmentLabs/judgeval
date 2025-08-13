from contextlib import contextmanager
from typing import Optional
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

# Shared console instance for the trainer module to avoid conflicts
shared_console = Console()


@contextmanager
def _spinner_progress(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Context manager for spinner-based progress display."""
    if step is not None and total_steps is not None:
        full_message = f"[Step {step}/{total_steps}] {message}"
    else:
        full_message = f"[Training] {message}"

    spinner = Spinner("dots", text=Text(full_message, style="cyan"))

    with Live(spinner, console=shared_console, refresh_per_second=10):
        yield


@contextmanager
def _model_spinner_progress(message: str):
    """Context manager for model operation spinner-based progress display."""
    spinner = Spinner("dots", text=Text(f"[Model] {message}", style="blue"))

    with Live(spinner, console=shared_console, refresh_per_second=10):
        yield


def _print_progress(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Print progress message with consistent formatting."""
    if step is not None and total_steps is not None:
        shared_console.print(f"[Step {step}/{total_steps}] {message}", style="green")
    else:
        shared_console.print(f"[Training] {message}", style="green")


def _print_progress_update(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Print progress update message (for status changes during long operations)."""
    if step is not None and total_steps is not None:
        shared_console.print(f"  └─ {message}", style="yellow")
    else:
        shared_console.print(f"  └─ {message}", style="yellow")


def _print_model_progress(message: str):
    """Print model progress message with consistent formatting."""
    shared_console.print(f"[Model] {message}", style="blue")
