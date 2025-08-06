import logging
import os
import sys

from judgeval.utils.cache import use_once

RESET = "\033[0m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
GRAY = "\033[90m"


class ColorFormatter(logging.Formatter):
    """
    Wrap the final formatted log record in ANSI color codes based on level.
    """

    COLORS = {
        logging.DEBUG: GRAY,
        logging.INFO: GRAY,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def __init__(self, fmt=None, datefmt=None, use_color=True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color and sys.stdout.isatty()

    def format(self, record):
        message = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelno, "")
            if color:
                message = f"{color}{message}{RESET}"
        return message


@use_once
def _setup_judgeval_logger():
    use_color = sys.stdout.isatty() and os.getenv("JUDGMMENT_NO_COLOR") is None
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=use_color,
        )
    )

    logger = logging.getLogger("judgeval")
    logger.addHandler(handler)
    return logger


judgeval_logger = _setup_judgeval_logger()


__all__ = ("judgeval_logger",)
