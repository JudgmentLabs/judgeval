from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from judgeval.exceptions import JudgmentAPIError, JudgmentValidationError
from judgeval.offline_tests.offline_test_runner import OfflineTestRunner
from judgeval.offline_tests.offline_tests_factory import OfflineTestsFactory
from judgeval.offline_tests.types import TestConfig, TestRunInfo

CONFIG_RESPONSE = {
    "test_config": {
        "id": "cfg-1",
        "name": "nightly",
        "dataset_id": "d1",
        "description": "desc",
        "user_id": "u1",
        "judges": [
            {
                "judge_id": "j1",
                "user_id": "u1",
                "judge": {
                    "id": "j1",
                    "name": "helpfulness",
                    "judge_type": "prompt",
                    "score_type": "binary",
                },
            }
        ],
    }
}

UUID = "123e4567-e89b-42d3-a456-426614174000"


def _make_factory(project_id="proj-1"):
    client = MagicMock()
    return OfflineTestsFactory(
        client=client, project_id=project_id, project_name="test-project"
    ), client


class TestCreateConfig:
    def test_create_with_judge_names(self):
        factory, client = _make_factory()
        client.post_projects_test_configs.return_value = CONFIG_RESPONSE
        config = factory.create_config(
            name="nightly", dataset="golden-set", judges=["helpfulness"]
        )
        assert isinstance(config, TestConfig)
        assert config.id == "cfg-1"
        assert config.judges[0].name == "helpfulness"
        payload = client.post_projects_test_configs.call_args.kwargs["payload"]
        assert payload["dataset_name"] == "golden-set"
        assert payload["judges"] == [{"name": "helpfulness"}]

    def test_create_with_dataset_and_judge_ids(self):
        factory, client = _make_factory()
        client.post_projects_test_configs.return_value = CONFIG_RESPONSE
        factory.create_config(name="nightly", dataset=UUID, judges=[UUID])
        payload = client.post_projects_test_configs.call_args.kwargs["payload"]
        assert payload["dataset_id"] == UUID
        assert payload["judges"] == [{"judge_id": UUID}]

    def test_create_invalid_judge_entry_raises(self):
        factory, _ = _make_factory()
        with pytest.raises(ValueError):
            factory.create_config(name="n", dataset="d", judges=[{"foo": "bar"}])

    def test_create_maps_validation_error(self):
        factory, client = _make_factory()
        client.post_projects_test_configs.side_effect = JudgmentAPIError(
            422, "Judges not found: nope", None
        )
        with pytest.raises(JudgmentValidationError):
            factory.create_config(name="n", dataset="d", judges=["nope"])

    def test_create_missing_project_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        assert factory.create_config(name="n", dataset="d", judges=["j"]) is None


class TestGetConfig:
    def test_get_by_id(self):
        factory, client = _make_factory()
        client.get_projects_test_configs_by_test_config_id.return_value = (
            CONFIG_RESPONSE
        )
        config = factory.get_config(UUID)
        assert config.id == "cfg-1"
        client.get_projects_test_configs_by_test_config_id.assert_called_once()

    def test_get_by_name_scans_list(self):
        factory, client = _make_factory()
        client.get_projects_test_configs.return_value = {
            "test_configs": [CONFIG_RESPONSE["test_config"]]
        }
        config = factory.get_config("nightly")
        assert config.id == "cfg-1"

    def test_get_by_name_not_found_returns_none(self):
        factory, client = _make_factory()
        client.get_projects_test_configs.return_value = {"test_configs": []}
        assert factory.get_config("missing") is None


class TestListAndDelete:
    def test_list_configs(self):
        factory, client = _make_factory()
        client.get_projects_test_configs.return_value = {
            "test_configs": [CONFIG_RESPONSE["test_config"]]
        }
        configs = factory.list_configs()
        assert len(configs) == 1
        assert configs[0].name == "nightly"

    def test_delete_config(self):
        factory, client = _make_factory()
        assert factory.delete_config("cfg-1") is True
        client.delete_projects_test_configs_by_test_config_id.assert_called_once()


class TestRuns:
    def test_run_resolves_config_and_delegates(self):
        factory, client = _make_factory()
        client.get_projects_test_configs.return_value = {
            "test_configs": [CONFIG_RESPONSE["test_config"]]
        }
        sentinel = object()
        with patch.object(OfflineTestRunner, "run", return_value=sentinel) as run_mock:
            result = factory.run(test_config="nightly")
        assert result is sentinel
        config_arg = run_mock.call_args.args[0]
        assert isinstance(config_arg, TestConfig)
        assert config_arg.id == "cfg-1"

    def test_run_accepts_test_config_object(self):
        factory, client = _make_factory()
        config = TestConfig(id="cfg-1", name="nightly", dataset_id="d1")
        sentinel = object()
        with patch.object(OfflineTestRunner, "run", return_value=sentinel):
            result = factory.run(test_config=config)
        assert result is sentinel
        client.get_projects_test_configs.assert_not_called()

    def test_run_unknown_config_returns_none(self):
        factory, client = _make_factory()
        client.get_projects_test_configs.return_value = {"test_configs": []}
        assert factory.run(test_config="missing") is None

    def test_run_missing_project_returns_none(self):
        factory, _ = _make_factory(project_id=None)
        assert factory.run(test_config="nightly") is None

    def test_list_runs(self):
        factory, client = _make_factory()
        client.get_projects_test_runs.return_value = {
            "test_runs": [
                {
                    "id": "run-1",
                    "test_config_id": "cfg-1",
                    "dataset_id": "d1",
                    "dataset_version_id": "v1",
                    "status": "completed",
                    "source": "sdk",
                    "user_id": "u1",
                    "project_id": "proj-1",
                }
            ]
        }
        runs = factory.list_runs(test_config_id="cfg-1")
        assert len(runs) == 1
        assert isinstance(runs[0], TestRunInfo)
        assert runs[0].status == "completed"

    def test_get_run(self):
        factory, client = _make_factory()
        client.get_projects_test_runs_by_test_run_id.return_value = {
            "test_run": {
                "id": "run-1",
                "test_config_id": "cfg-1",
                "dataset_id": "d1",
                "dataset_version_id": "v1",
                "status": "running",
            }
        }
        run = factory.get_run("run-1")
        assert run.id == "run-1"
        assert run.status == "running"
