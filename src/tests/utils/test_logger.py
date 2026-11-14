import os
import subprocess
import sys

WARN = "from judgeval.logger import judgeval_logger; judgeval_logger.warning('hello')"


def test_logs_go_to_stderr_leaving_stdout_to_the_program():
    result = subprocess.run(
        [sys.executable, "-c", WARN],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "JUDGMENT_LOG_LEVEL": "WARNING"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert "hello" in result.stderr
