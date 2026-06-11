from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from judgeval.exceptions import JudgmentAPIError, map_judgment_api_error
from judgeval.internal.api import JudgmentSyncClient
from judgeval.logger import judgeval_logger
from judgeval.offline_tests.offline_test_runner import (
    AgentFunction,
    OfflineTestRunner,
    PassConditionFn,
)
from judgeval.offline_tests.types import OfflineTestResult, TestConfig, TestRunInfo
from judgeval.utils.guards import expect_project_id

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_PATTERN.match(value))


class OfflineTestsFactory:
    """Create test configs and execute offline test runs.

    Access this via `client.offline_tests` -- you don't instantiate it
    directly. A *test config* pairs a dataset with a set of platform
    judges; a *test run* evaluates one dataset version with pinned judge
    versions and stores per-example results.

    Examples:
        Create a config and run it:

        ```python
        config = client.offline_tests.create_config(
            name="nightly-regression",
            dataset="golden-set",
            judges=["helpfulness", "faithfulness"],
        )

        result = client.offline_tests.run(test_config="nightly-regression")
        print(result.ui_results_url)
        ```

        Agent testing with a pass condition:

        ```python
        def my_agent(input: str) -> str:
            return agent.invoke(input)

        result = client.offline_tests.run(
            test_config="nightly-regression",
            agent_function=my_agent,
            pass_condition_fn=lambda fields, scorers: all(
                s.value != "No" for s in scorers
            ),
            assert_test=True,
        )
        ```
    """

    __slots__ = ("_client", "_project_id", "_project_name")

    def __init__(
        self,
        client: JudgmentSyncClient,
        project_id: Optional[str],
        project_name: str,
    ):
        self._client = client
        self._project_id = project_id
        self._project_name = project_name

    # ------------------------------------------------------------------ #
    #  Test configs                                                      #
    # ------------------------------------------------------------------ #

    def create_config(
        self,
        name: str,
        dataset: str,
        judges: List[Union[str, Dict[str, str]]],
        description: Optional[str] = None,
    ) -> Optional[TestConfig]:
        """Create a test config binding a dataset to a set of judges.

        The server enforces judge/dataset compatibility: every judge
        placeholder variable must be a required property of the dataset
        schema (`{{trace}}` requires a required `trace` property).

        Args:
            name: Name for the test config.
            dataset: Dataset name or dataset ID.
            judges: Judges to attach -- judge names (strings) or dicts
                with a `judge_id` or `name` key.
            description: Optional human-readable description.

        Returns:
            The created `TestConfig`, or `None` if the project is not
            resolved.

        Raises:
            JudgmentValidationError: If a judge does not exist or is
                incompatible with the dataset schema.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        judge_entries: List[Dict[str, str]] = []
        for judge in judges:
            if isinstance(judge, str):
                judge_entries.append(
                    {"judge_id": judge} if _is_uuid(judge) else {"name": judge}
                )
            elif isinstance(judge, dict) and (
                judge.get("judge_id") or judge.get("name")
            ):
                judge_entries.append(judge)
            else:
                raise ValueError(
                    "judges entries must be judge names or dicts with a "
                    "'judge_id' or 'name' key"
                )

        payload: Dict[str, Any] = {"name": name, "judges": judge_entries}
        if description is not None:
            payload["description"] = description
        if _is_uuid(dataset):
            payload["dataset_id"] = dataset
        else:
            payload["dataset_name"] = dataset

        try:
            response = self._client.post_projects_test_configs(
                project_id=project_id,
                payload=payload,  # type: ignore[arg-type]
            )
        except JudgmentAPIError as e:
            raise map_judgment_api_error(
                e, f"Failed to create test config '{name}': {e.detail}"
            ) from e

        judgeval_logger.info(f"Created test config {name}")
        return TestConfig.from_dict(response.get("test_config") or {})

    def get_config(self, test_config: str) -> Optional[TestConfig]:
        """Fetch a test config by ID or name.

        Args:
            test_config: Test config ID (UUID) or name.

        Returns:
            The `TestConfig`, or `None` if the project is not resolved or
            no config matches.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        if _is_uuid(test_config):
            response = self._client.get_projects_test_configs_by_test_config_id(
                project_id=project_id,
                test_config_id=test_config,
            )
            return TestConfig.from_dict(response.get("test_config") or {})

        configs = self.list_configs() or []
        for config in configs:
            if config.name == test_config:
                return config
        judgeval_logger.error(f"Test config '{test_config}' not found")
        return None

    def list_configs(
        self, dataset_id: Optional[str] = None
    ) -> Optional[List[TestConfig]]:
        """List test configs in the project.

        Args:
            dataset_id: Optionally filter to configs for one dataset.

        Returns:
            A list of `TestConfig` objects, or `None` if the project is
            not resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        response = self._client.get_projects_test_configs(
            project_id=project_id,
            dataset_id=dataset_id,
        )
        return [TestConfig.from_dict(c) for c in response.get("test_configs", []) or []]

    def delete_config(self, test_config_id: str) -> bool:
        """Delete a test config by ID.

        Returns:
            True if the config was deleted, False if the project is not
            resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return False

        self._client.delete_projects_test_configs_by_test_config_id(
            project_id=project_id,
            test_config_id=test_config_id,
        )
        judgeval_logger.info(f"Deleted test config {test_config_id}")
        return True

    # ------------------------------------------------------------------ #
    #  Test runs                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        test_config: Union[str, TestConfig],
        agent_function: Optional[AgentFunction] = None,
        judge_versions: Optional[List[Dict[str, Any]]] = None,
        dataset_version: Optional[int | str] = None,
        versioned_results: Optional[bool] = None,
        pass_condition_fn: Optional[PassConditionFn] = None,
        assert_test: bool = False,
        timeout_seconds: int = 600,
    ) -> Optional[OfflineTestResult]:
        """Run an offline test for a test config.

        Fetches the dataset version's examples, optionally calls your
        agent entrypoint once per example (each call producing an offline
        trace), then creates the test run with the agent traces attached
        so server-side judges evaluate with the agent's trace in context.
        Waits for results and reports per-row pass/fail outcomes back to
        the platform.

        Args:
            test_config: Test config name, ID, or `TestConfig` object.
            agent_function: Optional agent entrypoint. Called once per
                dataset example with the example's data fields as
                same-named keyword arguments; a signature mismatch raises
                `TypeError`. Each call is wrapped in an `OfflineTracer`
                and its offline trace is attributed to the result row.
                The agent runs *before* the test run is created; the
                collected traces are attached at creation so judges see
                them in context.
            judge_versions: Optional version pins, e.g.
                `[{"name": "helpfulness", "tag": "prod"}]` or
                `[{"name": "helpfulness", "version": "1.2"}]`. Judges not
                listed default to their `prod` tag (else latest). Pinning
                two versions of the same judge runs both (results are
                attributed per version).
            dataset_version: Dataset version to evaluate -- a version
                number (int) or version ID (str). Defaults to the latest
                version.
            versioned_results: Advanced override; defaults to True
                automatically when `judge_versions` pins the same judge
                twice.
            pass_condition_fn: Optional callable
                `(data_fields, scorer_data_list) -> bool` evaluated per
                example row; the outcome is stored as the row's `success`.
            assert_test: Raise `JudgmentTestError` unless every row passes
                its pass condition. Requires `pass_condition_fn`.
            timeout_seconds: Maximum seconds to wait for judge results.

        Returns:
            An `OfflineTestResult`, or `None` if the project is not
            resolved or the config cannot be found.

        Raises:
            ValueError: If `assert_test` is set without `pass_condition_fn`,
                or if `dataset_version` does not match any version of the
                config's dataset.
            TypeError: If the agent entrypoint cannot accept an example's
                fields.
            JudgmentValidationError: If the server rejects the run (e.g.
                unknown judge version, empty dataset).
            JudgmentTestError: If `assert_test` is set and any row fails.
            TimeoutError: If results are not ready within `timeout_seconds`.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        if isinstance(test_config, TestConfig):
            config: Optional[TestConfig] = test_config
        else:
            config = self.get_config(test_config)
        if config is None or not config.id:
            return None

        runner = OfflineTestRunner(
            client=self._client,
            project_id=project_id,
            project_name=self._project_name,
        )
        return runner.run(
            config,
            agent_function=agent_function,
            judge_versions=judge_versions,
            dataset_version=dataset_version,
            versioned_results=versioned_results,
            pass_condition_fn=pass_condition_fn,
            assert_test=assert_test,
            timeout_seconds=timeout_seconds,
        )

    def list_runs(
        self,
        test_config_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[List[TestRunInfo]]:
        """List test runs in the project, optionally filtered.

        Args:
            test_config_id: Filter to runs of one test config.
            dataset_id: Filter to runs over one dataset.
            status: Filter by run status.

        Returns:
            A list of `TestRunInfo` objects, or `None` if the project is
            not resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        response = self._client.get_projects_test_runs(
            project_id=project_id,
            test_config_id=test_config_id,
            dataset_id=dataset_id,
            status=status,
        )
        return [TestRunInfo.from_dict(r) for r in response.get("test_runs", []) or []]

    def get_run(self, test_run_id: str) -> Optional[TestRunInfo]:
        """Fetch test run metadata by ID.

        Returns:
            The `TestRunInfo`, or `None` if the project is not resolved.
        """
        project_id = expect_project_id(self._project_id)
        if not project_id:
            return None

        response = self._client.get_projects_test_runs_by_test_run_id(
            project_id=project_id,
            test_run_id=test_run_id,
        )
        return TestRunInfo.from_dict(response.get("test_run") or {})
