from contextlib import contextmanager
from typing import Optional
import sys
import os


# Detect if we're running in a Jupyter environment
def _is_jupyter_environment():
    """Check if we're running in a Jupyter notebook or similar environment."""
    try:
        # Check for IPython kernel
        if "ipykernel" in sys.modules or "IPython" in sys.modules:
            return True
        # Check for Jupyter environment variables
        if "JPY_PARENT_PID" in os.environ:
            return True
        # Check if we're in Google Colab
        if "google.colab" in sys.modules:
            return True
        return False
    except Exception:
        return False


# Check environment once at import time
IS_JUPYTER = _is_jupyter_environment()

if not IS_JUPYTER:
    # Safe to use Rich in non-Jupyter environments
    try:
        from rich.console import Console
        from rich.spinner import Spinner
        from rich.live import Live
        from rich.text import Text

        # Shared console instance for the trainer module to avoid conflicts
        shared_console = Console()
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
else:
    # In Jupyter, avoid Rich to prevent recursion issues
    RICH_AVAILABLE = False


# Fallback implementations for when Rich is not available or safe
class SimpleSpinner:
    def __init__(self, name, text):
        self.text = text


class SimpleLive:
    def __init__(self, spinner, console=None, refresh_per_second=None):
        self.spinner = spinner

    def __enter__(self):
        print(f"🔄 {self.spinner.text}")
        return self

    def __exit__(self, *args):
        pass

    def update(self, spinner):
        print(f"🔄 {spinner.text}")


def safe_print(message, style=None):
    """Safe print function that works in all environments."""
    if RICH_AVAILABLE and not IS_JUPYTER:
        shared_console.print(message, style=style)
    else:
        # Use simple print with emoji indicators for different styles
        if style == "green":
            print(f"✅ {message}")
        elif style == "yellow":
            print(f"⚠️ {message}")
        elif style == "blue":
            print(f"🔵 {message}")
        elif style == "cyan":
            print(f"🔷 {message}")
        else:
            print(message)


@contextmanager
def _spinner_progress(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Context manager for spinner-based progress display."""
    if step is not None and total_steps is not None:
        full_message = f"[Step {step}/{total_steps}] {message}"
    else:
        full_message = f"[Training] {message}"

    if RICH_AVAILABLE and not IS_JUPYTER:
        spinner = Spinner("dots", text=Text(full_message, style="cyan"))
        with Live(spinner, console=shared_console, refresh_per_second=10):
            yield
    else:
        # Fallback for Jupyter or when Rich is not available
        print(f"🔄 {full_message}")
        try:
            yield
        finally:
            print(f"✅ {full_message} - Complete")


@contextmanager
def _model_spinner_progress(message: str):
    """Context manager for model operation spinner-based progress display."""
    if RICH_AVAILABLE and not IS_JUPYTER:
        spinner = Spinner("dots", text=Text(f"[Model] {message}", style="blue"))
        with Live(spinner, console=shared_console, refresh_per_second=10) as live:

            def update_progress(progress_message: str):
                """Update the spinner with a new progress message."""
                new_text = f"[Model] {message}\n  └─ {progress_message}"
                spinner.text = Text(new_text, style="blue")
                live.update(spinner)

            yield update_progress
    else:
        # Fallback for Jupyter or when Rich is not available
        print(f"🔵 [Model] {message}")

        def update_progress(progress_message: str):
            print(f"  └─ {progress_message}")

        try:
            yield update_progress
        finally:
            print(f"✅ [Model] {message} - Complete")


def _print_progress(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Print progress message with consistent formatting."""
    if step is not None and total_steps is not None:
        safe_print(f"[Step {step}/{total_steps}] {message}", style="green")
    else:
        safe_print(f"[Training] {message}", style="green")


def _print_progress_update(
    message: str, step: Optional[int] = None, total_steps: Optional[int] = None
):
    """Print progress update message (for status changes during long operations)."""
    if step is not None and total_steps is not None:
        safe_print(f"  └─ {message}", style="yellow")
    else:
        safe_print(f"  └─ {message}", style="yellow")


def _print_model_progress(message: str):
    """Print model progress message with consistent formatting."""
    safe_print(f"[Model] {message}", style="blue")
