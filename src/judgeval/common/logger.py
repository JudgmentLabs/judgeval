import logging
from logging.handlers import RotatingFileHandler
import sys
from pathlib import Path
from contextlib import contextmanager

# Global variables
logger = None


class LoggingState:
    enabled: bool = False
    path: str | None = None


LOGGING_STATE = LoggingState()

# Add these as module-level variables
current_example_id = None
current_timestamp = None


@contextmanager
def enable_logging(
    name: str = "judgeval",
    path: str = "./logs",
    max_bytes: int = 1024 * 1024,
    backup_count: int = 5,
):
    """
    Context manager to temporarily enable logging for a specific block of code.
    """
    global logger
    LOGGING_STATE.enabled = True
    LOGGING_STATE.path = path
    # Initialize logger if not already initialized
    if logger is None:
        logger = _initialize_logger(
            name=name, path=path, max_bytes=max_bytes, backup_count=backup_count
        )
    try:
        logger.info("Logging enabled")
        yield
    finally:
        logger.info("Logging disabled")
        LOGGING_STATE.enabled = False
        LOGGING_STATE.path = None


def _initialize_logger(
    name: str = "judgeval",
    max_bytes: int = 1024 * 1024,  # 1MB
    backup_count: int = 5,
    path: str = "./logs",  # Added path parameter with default
) -> logging.Logger:
    """
    Initialize the global logger instance if it doesn't exist.
    Returns the global logger instance.
    """
    global logger

    log_dir = Path(path)
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"{name}.log"
    if log_file.exists():
        log_file.unlink()  # Delete existing log file

    if logger is not None:
        return logger

    # Create logs directory if it doesn't exist
    log_dir = Path(path)
    log_dir.mkdir(exist_ok=True)

    # Create a custom formatter that includes example info when available
    class ExampleFormatter(logging.Formatter):
        def format(self, record):
            if current_example_id is not None and current_timestamp is not None:
                record.example_id = current_example_id
                record.timestamp = current_timestamp
                return logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - [Example_%(example_id)s][%(timestamp)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                ).format(record)
            return logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ).format(record)

    # Use the custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ExampleFormatter())
    console_handler.setLevel(logging.DEBUG)

    log_filename = f"{name}.log"
    file_handler = RotatingFileHandler(
        log_dir / log_filename, maxBytes=max_bytes, backupCount=backup_count, mode="a"
    )
    file_handler.setFormatter(ExampleFormatter())
    file_handler.setLevel(logging.DEBUG)

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# Initialize the global logger when module is imported
# logger = _initialize_logger()


def log_if_enabled(func):
    """Decorator to check if logging is enabled before executing logging statements"""

    def wrapper(*args, **kwargs):
        if LOGGING_STATE.enabled:
            return func(*args, **kwargs)

    return wrapper


@log_if_enabled
def debug(msg: str, example_idx: int | None = None):
    """Log debug message if logging is enabled"""
    if logger:
        logger.debug(msg)


@log_if_enabled
def info(msg: str, example_idx: int | None = None):
    """Log info message if logging is enabled"""
    if logger:
        logger.info(msg)


@log_if_enabled
def warning(msg: str, example_idx: int | None = None):
    """Log warning message if logging is enabled"""
    if logger:
        logger.warning(msg)


@log_if_enabled
def error(msg: str, example_idx: int | None = None):
    """Log error message if logging is enabled"""
    if logger:
        logger.error(msg)


def create_example_handler(
    timestamp: str,
    example_idx: int,
    path: str = "./logs",  # Added path parameter with default
) -> RotatingFileHandler:
    """Creates a file handler for a specific example"""
    debug(
        f"Creating example handler for timestamp={timestamp}, example_idx={example_idx}"
    )
    log_dir = Path(path) / "examples"
    log_dir.mkdir(exist_ok=True, parents=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - [Example_%(example_id)s][%(timestamp)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a unique file for each example
    file_handler = RotatingFileHandler(
        log_dir / f"{timestamp}_example_{example_idx}.log",
        maxBytes=1024 * 1024,  # 1MB
        backupCount=5,
        mode="a",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    info(f"Created example handler for example {example_idx}")
    return file_handler


@contextmanager
def example_logging_context(timestamp: str, example_idx: int):
    """Context manager for example-specific logging"""
    if not LOGGING_STATE.enabled:
        yield
        return

    global current_example_id, current_timestamp

    debug(f"Entering example logging context for example {example_idx}")
    current_example_id = example_idx
    current_timestamp = timestamp

    if LOGGING_STATE.path:
        handler = create_example_handler(
            timestamp, example_idx, path=LOGGING_STATE.path
        )
    if handler and logger:
        logger.addHandler(handler)
    try:
        yield
    finally:
        current_example_id = None
        current_timestamp = None
        if handler and logger:
            logger.removeHandler(handler)
            handler.close()
            debug(f"Closed example handler for example {example_idx}")
