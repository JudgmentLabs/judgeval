from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from judgeval.data.scoring_result import ScoringResult


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
        judges: The judge membership rows as returned by the server.
    """

    __test__ = False

    id: str
    name: str
    dataset_id: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    judges: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestConfig:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            dataset_id=data.get("dataset_id", ""),
            description=data.get("description"),
            created_at=data.get("created_at"),
            judges=[j for j in (data.get("judges") or []) if isinstance(j, dict)],
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
