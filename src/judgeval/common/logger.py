# logger.py
import logging
from rich.console import Console
from rich.logging import RichHandler


def _setup_judgeval_logger():
    logger = logging.getLogger("judgeval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    handler = RichHandler(
        console=Console(force_terminal=True, markup=True),
        markup=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )

    logger.addHandler(handler)
    return logger


judgeval_logger = _setup_judgeval_logger()
