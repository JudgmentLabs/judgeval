import logging
import sys

from judgeval.env import JUDGMENT_NO_COLOR, JUDGMENT_LOG_LEVEL
from judgeval.utils.decorators.use_once import use_once

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


def _parse_log_level(level_str: str) -> int:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    return level_map.get(level_str.lower(), logging.WARNING)


@use_once
def _setup_judgeval_logger():
    use_color = sys.stdout.isatty() and JUDGMENT_NO_COLOR is None
    log_level = _parse_log_level(JUDGMENT_LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(
        ColorFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_color=use_color,
        )
    )

    logger = logging.getLogger("judgeval")
    logger.setLevel(log_level)
    logger.addHandler(handler)
    return logger


judgeval_logger = _setup_judgeval_logger()


__all__ = ("judgeval_logger",)
