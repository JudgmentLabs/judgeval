from __future__ import annotations

from judgeval.offline_tests.offline_tests_factory import OfflineTestsFactory
from judgeval.offline_tests.offline_test_runner import (
    AgentFunction,
    JudgeVersionPin,
    OfflineTestRunner,
    PassConditionFn,
)
from judgeval.offline_tests.types import OfflineTestResult, TestConfig

OFFLINE_TESTS_DEPRECATION = (
    "Offline tests and OfflineTracer are deprecated. Use client.tests and "
    "client.test_runs, and tag your normal Tracer's traces with Tracer.test_case."
)

__all__ = [
    "AgentFunction",
    "JudgeVersionPin",
    "OFFLINE_TESTS_DEPRECATION",
    "OfflineTestsFactory",
    "OfflineTestRunner",
    "OfflineTestResult",
    "PassConditionFn",
    "TestConfig",
]
