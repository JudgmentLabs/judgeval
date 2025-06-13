import pytest
from pathlib import Path
import logging
from judgeval.common.logger import (
    enable_logging,
    _initialize_logger,
    debug,
    info,
    warning,
    error,
    example_logging_context,
    LOGGING_STATE,
)

TEST_LOG_PATH = "./tests/common/logs"


@pytest.fixture
def clean_logs():
    """Remove log files before and after tests"""
    # Clean before test
    log_dir = Path(TEST_LOG_PATH)
    if log_dir.exists():
        for file in log_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in reversed(list(log_dir.glob("**/*"))):
            if dir.is_dir():
                dir.rmdir()
        log_dir.rmdir()

    yield

    # Clean after test
    if log_dir.exists():
        for file in log_dir.glob("**/*"):
            if file.is_file():
                file.unlink()
        for dir in reversed(list(log_dir.glob("**/*"))):
            if dir.is_dir():
                dir.rmdir()
        log_dir.rmdir()


def test_enable_logging_context(clean_logs):
    """Test that logging is properly enabled and disabled with context manager"""
    assert not LOGGING_STATE.enabled

    with enable_logging(path=TEST_LOG_PATH):
        assert LOGGING_STATE.enabled

    assert not LOGGING_STATE.enabled


def test_logger_initialization(clean_logs):
    """Test logger initialization and file creation"""
    test_logger = _initialize_logger("test_logger", path=TEST_LOG_PATH)

    assert isinstance(test_logger, logging.Logger)
    assert len(test_logger.handlers) == 2  # Console and file handler

    # Check that log directory exists instead of file
    log_dir = Path(TEST_LOG_PATH)
    assert log_dir.exists()
    assert log_dir.is_dir()


# def test_logging_levels(clean_logs):
#     """Test all logging levels work when enabled"""
#     log_dir = Path(TEST_LOG_PATH)
#     log_file = log_dir / "judgeval.log"

#     with enable_logging(path=TEST_LOG_PATH):
#         debug("Debug message")
#         info("Info message")
#         warning("Warning message")
#         error("Error message")

#     assert log_file.exists()
#     content = log_file.read_text()

#     assert "DEBUG" in content
#     assert "INFO" in content
#     assert "WARNING" in content
#     assert "ERROR" in content
#     assert "Debug message" in content
#     assert "Info message" in content
#     assert "Warning message" in content
#     assert "Error message" in content


def test_logging_disabled(clean_logs):
    """Test that logging doesn't occur when disabled"""
    log_file = Path(TEST_LOG_PATH) / "judgeval.log"

    # These should not create any logs
    debug("Debug message", path=TEST_LOG_PATH)
    info("Info message", path=TEST_LOG_PATH)
    warning("Warning message", path=TEST_LOG_PATH)
    error("Error message", path=TEST_LOG_PATH)

    assert not log_file.exists() or log_file.stat().st_size == 0


def test_example_logging_context(clean_logs):
    """Test example-specific logging context"""
    example_log_dir = Path(TEST_LOG_PATH) / "examples"
    timestamp = "2024-03-21_103045"
    example_idx = 123

    with enable_logging(path=TEST_LOG_PATH):
        with example_logging_context(timestamp, example_idx):
            info("Test example message")

    # Check example-specific log file
    example_log_file = example_log_dir / f"{timestamp}_example_{example_idx}.log"
    assert example_log_file.exists()

    content = example_log_file.read_text()
    assert f"[Example_{example_idx}]" in content
    assert timestamp in content
    assert "Test example message" in content


def test_nested_example_contexts(clean_logs):
    """Test that nested example contexts work correctly"""
    with enable_logging(path=TEST_LOG_PATH):
        with example_logging_context("time1", 1):
            info("Message 1")
            with example_logging_context("time2", 2):
                info("Message 2")
            info("Back to 1")

    # Check that example log files exist
    assert (Path(TEST_LOG_PATH) / "examples" / "time1_example_1.log").exists()
    assert (Path(TEST_LOG_PATH) / "examples" / "time2_example_2.log").exists()


# def test_logger_rotation(clean_logs):
#     """Test that log files rotate when they exceed max size"""
#     # Create a logger with very small max_bytes to test rotation
#     with enable_logging(name="rotation_test", path=TEST_LOG_PATH, max_bytes=50, backup_count=2):
#         # Write enough data to trigger multiple rotations
#         for i in range(10):
#             info(f"This is a long message that will cause rotation {i}")

#     log_dir = Path(TEST_LOG_PATH)
#     assert (log_dir / "rotation_test.log").exists()
#     assert (log_dir / "rotation_test.log.1").exists()
#     assert (log_dir / "rotation_test.log.2").exists()
#     # Should not exist due to backup_count=2
#     assert not (log_dir / "rotation_test.log.3").exists()

# def test_formatter_without_example_context(clean_logs):
#     """Test that logs are properly formatted without example context"""
#     with enable_logging(name="judgeval", path=TEST_LOG_PATH):
#         info("Regular message")

#     content = (Path(TEST_LOG_PATH) / "judgeval.log").read_text()
#     assert "[Example_" not in content
#     assert " - judgeval - INFO - Regular message" in content
