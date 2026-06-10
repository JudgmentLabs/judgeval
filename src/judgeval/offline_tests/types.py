from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from judgeval.data.scoring_result import ScoringResult


@dataclass
class TestConfigJudge:
    """A judge attached to a test config.

    Attributes:
        judge_id: The platform judge ID.
        name: Judge name (when expanded by the server).
        judge_type: Judge type (e.g. `"prompt"`, `"agent"`).
        score_type: The judge's score type.
    """

    __test__ = False

    judge_id: str
    name: Optional[str] = None
    judge_type: Optional[str] = None
    score_type: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestConfigJudge:
        judge = data.get("judge") or {}
        return cls(
            judge_id=data.get("judge_id", "") or judge.get("id", ""),
            name=judge.get("name"),
            judge_type=judge.get("judge_type"),
            score_type=judge.get("score_type"),
        )


@dataclass
class TestConfig:
    """A reusable offline-test configuration (dataset + judges).

    Created via `client.offline_tests.create_config()`. A test config
    pins a dataset and a set of platform judges; each `run()` against it
    creates a new test run.

    Attributes:
        id: Unique test config ID.
        name: Config name.
        dataset_id: The dataset evaluated by this config.
        description: Optional human-readable description.
        created_at: ISO-8601 creation timestamp.
        judges: The judges attached to this config.
    """

    __test__ = False

    id: str
    name: str
    dataset_id: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    judges: List[TestConfigJudge] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestConfig:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            dataset_id=data.get("dataset_id", ""),
            description=data.get("description"),
            created_at=data.get("created_at"),
            judges=[
                TestConfigJudge.from_dict(j)
                for j in (data.get("judges") or [])
                if isinstance(j, dict)
            ],
        )


@dataclass
class TestRunInfo:
    """Metadata for a single offline test run.

    Attributes:
        id: Unique test run ID.
        test_config_id: The config this run was created from.
        dataset_id: Dataset evaluated by the run.
        dataset_version_id: The pinned dataset version.
        status: One of `pending`, `running`, `completed`, `error`,
            `cancelled`.
        source: Where the run was created from (`sdk`, `cli`, `mcp`,
            `platform`).
        error_message: Error detail when `status == "error"`.
        created_at: ISO-8601 creation timestamp.
    """

    __test__ = False

    id: str
    test_config_id: str
    dataset_id: str
    dataset_version_id: str
    status: str
    source: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestRunInfo:
        return cls(
            id=data.get("id", ""),
            test_config_id=data.get("test_config_id", ""),
            dataset_id=data.get("dataset_id", ""),
            dataset_version_id=data.get("dataset_version_id", ""),
            status=data.get("status", ""),
            source=data.get("source"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at"),
        )


@dataclass
class OfflineTestResult:
    """The outcome of an offline test run.

    Returned by `client.offline_tests.run()`. Contains the per-example
    scoring results plus run-level metadata.

    Attributes:
        test_run_id: The test run ID.
        status: Final run status.
        ui_results_url: Link to the results page in the dashboard.
        results: One `ScoringResult` per dataset example, with per-judge
            `ScorerData` entries. When a `pass_condition_fn` was supplied,
            each `ScorerData.success` carries the per-row outcome.
        agent_offline_trace_ids: Mapping of example ID to the offline
            trace produced by the agent entrypoint (agent testing only).
    """

    test_run_id: str
    status: str
    ui_results_url: Optional[str] = None
    results: List[ScoringResult] = field(default_factory=list)
    agent_offline_trace_ids: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> Optional[bool]:
        """Whether every row passed its pass condition.

        Returns `None` when no pass condition was evaluated.
        """
        successes = [
            scorer.success
            for result in self.results
            for scorer in result.scorers_data
            if scorer.success is not None
        ]
        if not successes:
            return None
        return all(successes)
