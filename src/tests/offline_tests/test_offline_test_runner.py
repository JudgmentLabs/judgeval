from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from judgeval.data.example import Example
from judgeval.data.scorer_data import ScorerData
from judgeval.exceptions import (
    JudgmentAPIError,
    JudgmentTestError,
    JudgmentValidationError,
)
from judgeval.offline_tests.offline_test_runner import (
    OfflineTestRunner,
    build_agent_kwargs,
    normalize_judge_versions,
)
from judgeval.offline_tests.types import OfflineTestResult, TestConfig

CONFIG = TestConfig(id="cfg-1", name="nightly", dataset_id="d1")

VERSIONS = {
    "versions": [
        {
            "version_id": "v1",
            "dataset_id": "d1",
            "version_number": 1,
            "created_at": "2026-01-01",
        },
    ]
}

PAGE = {
    "dataset": {"dataset_id": "d1", "name": "golden"},
    "entries": [
        {
            "item": {
                "id": "item-1",
                "version_added": 1,
                "example_id": "ex-1",
                "created_at": "2026-01-02",
            },
            "example": {
                "example_id": "ex-1",
                "data": {"input": "q1"},
                "offline_trace_id": None,
                "metadata": {},
                "created_at": "2026-01-01",
            },
        },
    ],
    "metadata": {"hasMore": False, "nextCursor": None},
}

PREPARED = {
    "test_run": {"id": "run-1", "status": "running"},
    "dataset": {"dataset_id": "d1", "name": "golden"},
    "dataset_version": {"version_id": "v1", "version_number": 1},
    "examples": [
        {
            "example_id": "ex-1",
            "data": {"input": "q1"},
            "offline_trace_id": None,
            "created_at": "2026-01-01",
            "user_id": "u1",
        },
    ],
    "judges": [
        {
            "judge_id": "j1",
            "judge_name": "helpfulness",
            "judge_type": "prompt",
            "score_type": "binary",
            "judge_major_version": 1,
            "judge_minor_version": 0,
        }
    ],
    "evaluation_runs": [
        {
            "run_id": "er-1",
            "judge_name": "helpfulness",
            "example_id": "ex-1",
            "test_run_id": "run-1",
            "judge_id": "j1",
            "judge_major_version": 1,
            "judge_minor_version": 0,
        }
    ],
    "ui_results_url": "https://app/tests/run-1",
}

ITEMS = [
    {
        "example_id": "ex-1",
        "agent_offline_trace_id": None,
        "created_at": "2026-01-02",
        "example": {"example_id": "ex-1", "created_at": "2026-01-01", "data": {}},
        "data": {"input": "q1"},
        "scorers": [
            {
                "judge_id": "j1",
                "judge_name": "helpfulness",
                "judge_major_version": 1,
                "judge_minor_version": 0,
                "score_type": "binary",
                "num_value": 0,
                "bool_value": True,
                "str_value": "",
                "reason": json.dumps({"text": "looks good"}),
                "metadata": None,
                "success": None,
                "error": None,
                "created_at": "2026-01-02",
            }
        ],
    }
]


def _make_runner():
    client = MagicMock()
    client.base_url = "http://api.test"
    client._request.return_value = {"updated": 1}
    runner = OfflineTestRunner(
        client=client, project_id="proj-1", project_name="test-project"
    )
    return runner, client


def _stub_raw_request(client, items_pages=None):
    """Route the raw `_request` helper: GET serves items pages, PATCH acks.

    `items_pages` is a list of successive responses for the items GET;
    the last page is repeated if called again.
    """
    if items_pages is None:
        items_pages = [
            {
                "results": [json.loads(json.dumps(item)) for item in ITEMS],
                "has_more": False,
                "next_cursor": None,
                "ui_results_url": "https://app/tests/run-1",
            }
        ]
    state = {"page": 0}

    def _request(method, url, payload, *args, **kwargs):
        if method == "GET":
            page = items_pages[min(state["page"], len(items_pages) - 1)]
            state["page"] += 1
            return page
        return {"updated": 1}

    client._request.side_effect = _request


def _request_calls(client, method):
    return [c for c in client._request.call_args_list if c.args[0] == method]


def _stub_lifecycle(client, status="completed"):
    client.get_projects_datasets_by_dataset_identifier_versions.return_value = (
        json.loads(json.dumps(VERSIONS))
    )
    client.get_projects_datasets_by_dataset_identifier_page.return_value = json.loads(
        json.dumps(PAGE)
    )
    client.post_projects_test_runs.return_value = dict(PREPARED)
    client.get_projects_test_runs_by_test_run_id.return_value = {
        "test_run": {"id": "run-1", "status": status},
        "ui_results_url": "https://app/tests/run-1",
    }
    _stub_raw_request(client)


class TestNormalizeJudgeVersions:
    def test_none_passthrough(self):
        assert normalize_judge_versions(None) is None
        assert normalize_judge_versions([]) is None

    def test_requires_name_or_judge_id(self):
        with pytest.raises(ValueError, match="name"):
            normalize_judge_versions([{"tag": "prod"}])

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            normalize_judge_versions(["helpfulness"])  # type: ignore[list-item]

    def test_keeps_allowed_keys_only(self):
        normalized = normalize_judge_versions(
            [{"name": "j", "tag": "prod", "extra": "x"}]
        )
        assert normalized == [{"name": "j", "tag": "prod"}]


class TestBuildAgentKwargs:
    def test_maps_fields_to_kwargs(self):
        def agent(input):
            return input

        assert build_agent_kwargs(agent, {"input": "q"}) == {"input": "q"}

    def test_unexpected_field_raises(self):
        def agent(input):
            return input

        with pytest.raises(TypeError, match="does not accept"):
            build_agent_kwargs(agent, {"input": "q", "other": 1})

    def test_var_keyword_accepts_everything(self):
        def agent(**kwargs):
            return kwargs

        assert build_agent_kwargs(agent, {"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_missing_required_param_raises(self):
        def agent(input, expected_output):
            return input

        with pytest.raises(TypeError, match="requires parameter"):
            build_agent_kwargs(agent, {"input": "q"})

    def test_defaulted_params_optional(self):
        def agent(input, mode="fast"):
            return input

        assert build_agent_kwargs(agent, {"input": "q"}) == {"input": "q"}


class TestDatasetResolution:
    def test_latest_version_by_default(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": [
                {"version_id": "v2", "version_number": 2},
                {"version_id": "v1", "version_number": 1},
            ]
        }
        version = runner.resolve_dataset_version(CONFIG)
        assert version["version_id"] == "v2"
        kwargs = (
            client.get_projects_datasets_by_dataset_identifier_versions.call_args.kwargs
        )
        assert kwargs["dataset_identifier"] == "d1"

    def test_resolve_by_version_number(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": [
                {"version_id": "v2", "version_number": 2},
                {"version_id": "v1", "version_number": 1},
            ]
        }
        assert runner.resolve_dataset_version(CONFIG, 1)["version_id"] == "v1"

    def test_resolve_by_version_id(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": [
                {"version_id": "v2", "version_number": 2},
                {"version_id": "v1", "version_number": 1},
            ]
        }
        assert runner.resolve_dataset_version(CONFIG, "v1")["version_number"] == 1

    def test_resolve_unknown_version_raises(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = (
            json.loads(json.dumps(VERSIONS))
        )
        with pytest.raises(ValueError, match="does not exist"):
            runner.resolve_dataset_version(CONFIG, 99)
        with pytest.raises(ValueError, match="does not exist"):
            runner.resolve_dataset_version(CONFIG, "missing-id")

    def test_resolve_no_versions_raises(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_versions.return_value = {
            "versions": []
        }
        with pytest.raises(ValueError, match="no versions"):
            runner.resolve_dataset_version(CONFIG)

    def test_fetch_examples_single_page(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_page.return_value = (
            json.loads(json.dumps(PAGE))
        )
        examples = runner.fetch_examples(CONFIG, 1)
        assert examples == [
            {
                "example_id": "ex-1",
                "created_at": "2026-01-01",
                "data": {"input": "q1"},
                "offline_trace_id": None,
            }
        ]
        kwargs = (
            client.get_projects_datasets_by_dataset_identifier_page.call_args.kwargs
        )
        assert kwargs["version"] == "1"
        assert kwargs["cursor_created_at"] is None

    def test_fetch_examples_offline_trace_id_propagated(self):
        """An example carrying an offline_trace_id is forwarded to callers."""
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_page.return_value = {
            "dataset": {"dataset_id": "d1", "name": "golden"},
            "entries": [
                {
                    "item": {
                        "id": "item-1",
                        "version_added": 1,
                        "example_id": "ex-1",
                        "created_at": "2026-01-02",
                    },
                    "example": {
                        "example_id": "ex-1",
                        "data": {"input": "q1"},
                        "offline_trace_id": "trace-abc",
                        "metadata": {},
                        "created_at": "2026-01-01",
                    },
                }
            ],
            "metadata": {"hasMore": False, "nextCursor": None},
        }
        examples = runner.fetch_examples(CONFIG, 1)
        assert examples[0]["offline_trace_id"] == "trace-abc"
        assert examples[0]["example_id"] == "ex-1"
        assert examples[0]["data"] == {"input": "q1"}

    def test_fetch_examples_paginates_with_cursor(self):
        runner, client = _make_runner()

        def _make_entry(example_id, created_at, data):
            return {
                "item": {
                    "id": f"item-{example_id}",
                    "version_added": 1,
                    "example_id": example_id,
                    "created_at": "item-ts",
                },
                "example": {
                    "example_id": example_id,
                    "data": data,
                    "offline_trace_id": None,
                    "metadata": {},
                    "created_at": created_at,
                },
            }

        page_1 = {
            "entries": [
                _make_entry("ex-1", "t1", {"input": "q1"}),
                _make_entry("ex-2", "t2", {"input": "q2"}),
            ],
            "metadata": {
                "hasMore": True,
                "nextCursor": {"created_at": "t2", "example_id": "ex-2"},
            },
        }
        page_2 = {
            "entries": [
                _make_entry("ex-3", "t3", {"input": "q3"}),
            ],
            "metadata": {"hasMore": False, "nextCursor": None},
        }
        client.get_projects_datasets_by_dataset_identifier_page.side_effect = [
            page_1,
            page_2,
        ]
        examples = runner.fetch_examples(CONFIG, 1)
        assert [e["example_id"] for e in examples] == ["ex-1", "ex-2", "ex-3"]
        second_call = (
            client.get_projects_datasets_by_dataset_identifier_page.call_args_list[
                1
            ].kwargs
        )
        # cursor params must come from metadata.nextCursor, not the last entry
        assert second_call["cursor_created_at"] == "t2"
        assert second_call["cursor_example_id"] == "ex-2"

    def test_fetch_examples_parses_string_data(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_page.return_value = {
            "entries": [
                {
                    "item": {
                        "id": "item-1",
                        "version_added": 1,
                        "example_id": "ex-1",
                        "created_at": "item-ts",
                    },
                    "example": {
                        "example_id": "ex-1",
                        "data": json.dumps({"input": "q1"}),
                        "offline_trace_id": None,
                        "metadata": {},
                        "created_at": "t1",
                    },
                }
            ],
            "metadata": {"hasMore": False, "nextCursor": None},
        }
        examples = runner.fetch_examples(CONFIG, 1)
        assert examples[0]["data"] == {"input": "q1"}

    def test_fetch_examples_maps_api_error(self):
        runner, client = _make_runner()
        client.get_projects_datasets_by_dataset_identifier_page.side_effect = (
            JudgmentAPIError(422, "Version 9 does not exist", None)
        )
        with pytest.raises(JudgmentValidationError):
            runner.fetch_examples(CONFIG, 9)


class TestFetchItems:
    def _items_page(self, example_ids, has_more, next_cursor, url="https://app/run-1"):
        return {
            "results": [
                {"example_id": example_id, "data": {}, "scorers": []}
                for example_id in example_ids
            ],
            "has_more": has_more,
            "next_cursor": next_cursor,
            "ui_results_url": url,
        }

    def test_single_page(self):
        runner, client = _make_runner()
        client._request.return_value = self._items_page(["ex-1"], False, None)
        items, url = runner.fetch_items("run-1")
        assert [i["example_id"] for i in items] == ["ex-1"]
        assert url == "https://app/run-1"
        client._request.assert_called_once()
        args = client._request.call_args.args
        assert args[0] == "GET"
        assert args[1] == "http://api.test/v1/projects/proj-1/test-runs/run-1/items"
        # limit is always sent explicitly; no cursor on the first page
        assert args[2] == {"limit": 200}

    def test_paginates_until_has_more_false(self):
        runner, client = _make_runner()
        client._request.side_effect = [
            self._items_page(["ex-1", "ex-2"], True, "ex-2"),
            self._items_page(["ex-3"], False, None),
        ]
        items, url = runner.fetch_items("run-1")
        assert [i["example_id"] for i in items] == ["ex-1", "ex-2", "ex-3"]
        assert url == "https://app/run-1"
        assert client._request.call_count == 2
        second_params = client._request.call_args_list[1].args[2]
        assert second_params == {"limit": 200, "cursor": "ex-2"}

    def test_old_server_without_pagination_fields(self):
        """Servers predating pagination omit has_more/next_cursor entirely."""
        runner, client = _make_runner()
        client._request.return_value = {
            "results": [{"example_id": "ex-1", "data": {}, "scorers": []}],
            "ui_results_url": "https://app/run-1",
        }
        items, url = runner.fetch_items("run-1")
        assert [i["example_id"] for i in items] == ["ex-1"]
        assert url == "https://app/run-1"
        client._request.assert_called_once()

    def test_null_next_cursor_stops_despite_has_more(self):
        """A defective has_more=True with no cursor must not loop forever."""
        runner, client = _make_runner()
        client._request.return_value = self._items_page(["ex-1"], True, None)
        items, _ = runner.fetch_items("run-1")
        assert len(items) == 1
        client._request.assert_called_once()


class TestCreateTestRun:
    def test_payload_includes_source_and_config(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        runner.create_test_run(CONFIG)
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload == {"test_config_id": "cfg-1", "source": "sdk"}

    def test_dataset_version_number_and_id(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        runner.create_test_run(CONFIG, dataset_version=3)
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["dataset_version_number"] == 3

        runner.create_test_run(CONFIG, dataset_version="ver-uuid")
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["dataset_version_id"] == "ver-uuid"

    def test_duplicate_judge_versions_pass_through_unchanged(self):
        # Multi-version runs need no extra flags: queue refs carry the
        # pinned judge versions server-side.
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        runner.create_test_run(
            CONFIG,
            judge_versions=[
                {"name": "j", "tag": "prod"},
                {"name": "j", "version": "0.1"},
            ],
        )
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["judge_versions"] == [
            {"name": "j", "tag": "prod"},
            {"name": "j", "version": "0.1"},
        ]
        assert "versioned_results" not in payload

    def test_agent_traces_attached_to_payload(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        runner.create_test_run(CONFIG, agent_traces={"ex-1": "trace-abc"})
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["agent_traces"] == [
            {"example_id": "ex-1", "agent_offline_trace_id": "trace-abc"}
        ]

    def test_empty_agent_traces_omitted(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        runner.create_test_run(CONFIG, agent_traces={})
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert "agent_traces" not in payload

    def test_maps_422_to_validation_error(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.side_effect = JudgmentAPIError(
            422, "Judge 'x' has no version v1.2", None
        )
        with pytest.raises(JudgmentValidationError):
            runner.create_test_run(CONFIG)


class TestBuildResults:
    def test_builds_scoring_results(self):
        runner, _ = _make_runner()
        results = runner.build_results(ITEMS, agent_traces={})
        assert len(results) == 1
        result = results[0]
        assert isinstance(result.data_object, Example)
        assert result.data_object.example_id == "ex-1"
        assert result.data_object["input"] == "q1"
        scorer = result.scorers_data[0]
        assert scorer.name == "helpfulness"
        assert scorer.value == "Yes"
        assert scorer.score_type == "binary"
        assert scorer.success is None
        assert scorer.additional_metadata["reason"] == "looks good"

    def test_pass_condition_sets_success(self):
        runner, _ = _make_runner()

        def pass_fn(fields, scorers):
            assert fields == {"input": "q1"}
            assert all(isinstance(s, ScorerData) for s in scorers)
            return scorers[0].value == "Yes"

        results = runner.build_results(
            ITEMS, agent_traces={}, pass_condition_fn=pass_fn
        )
        assert results[0].scorers_data[0].success is True

    def test_agent_trace_recorded_on_result(self):
        runner, _ = _make_runner()
        results = runner.build_results(ITEMS, agent_traces={"ex-1": "trace-abc"})
        assert results[0].trace_id == "trace-abc"


class TestReportSuccess:
    def test_patches_success_per_evaluation_run(self):
        runner, client = _make_runner()
        results = runner.build_results(
            ITEMS,
            agent_traces={},
            pass_condition_fn=lambda fields, scorers: True,
        )
        runner.report_success("run-1", PREPARED, ITEMS, results)
        client._request.assert_called_once()
        args, kwargs = client._request.call_args
        assert args[0] == "PATCH"
        assert args[1] == "http://api.test/v1/projects/proj-1/test-runs/run-1/success"
        assert kwargs["payload"] == {
            "successes": [{"evaluation_run_id": "er-1", "success": True}]
        }

    def test_failure_outcome_is_patched(self):
        runner, client = _make_runner()
        results = runner.build_results(
            ITEMS,
            agent_traces={},
            pass_condition_fn=lambda fields, scorers: False,
        )
        runner.report_success("run-1", PREPARED, ITEMS, results)
        payload = client._request.call_args.kwargs["payload"]
        assert payload == {
            "successes": [{"evaluation_run_id": "er-1", "success": False}]
        }

    def test_matches_evaluation_run_ids_by_judge_version(self):
        runner, client = _make_runner()
        prepared = {
            "evaluation_runs": [
                {
                    "run_id": "er-v1",
                    "judge_name": "helpfulness",
                    "example_id": "ex-1",
                    "judge_id": "j1",
                    "judge_major_version": 1,
                    "judge_minor_version": 0,
                },
                {
                    "run_id": "er-v2",
                    "judge_name": "helpfulness",
                    "example_id": "ex-1",
                    "judge_id": "j1",
                    "judge_major_version": 2,
                    "judge_minor_version": 0,
                },
            ]
        }
        items = [
            {
                "example_id": "ex-1",
                "data": {"input": "q1"},
                "scorers": [
                    {
                        "judge_id": "j1",
                        "judge_name": "helpfulness",
                        "judge_major_version": 2,
                        "judge_minor_version": 0,
                        "score_type": "binary",
                        "bool_value": True,
                    },
                    {
                        "judge_id": "j1",
                        "judge_name": "helpfulness",
                        "judge_major_version": 1,
                        "judge_minor_version": 0,
                        "score_type": "binary",
                        "bool_value": False,
                    },
                ],
            }
        ]
        results = runner.build_results(
            items, agent_traces={}, pass_condition_fn=lambda fields, scorers: True
        )
        runner.report_success("run-1", prepared, items, results)
        payload = client._request.call_args.kwargs["payload"]
        # same judge name twice: version-keyed matching attributes each row
        assert payload == {
            "successes": [
                {"evaluation_run_id": "er-v2", "success": True},
                {"evaluation_run_id": "er-v1", "success": True},
            ]
        }

    def test_falls_back_to_judge_name_match(self):
        runner, client = _make_runner()
        prepared = {
            "evaluation_runs": [
                {
                    "run_id": "er-1",
                    "judge_name": "helpfulness",
                    "example_id": "ex-1",
                }
            ]
        }
        items = [
            {
                "example_id": "ex-1",
                "data": {},
                "scorers": [
                    {
                        "judge_id": "j1",
                        "judge_name": "helpfulness",
                        "judge_major_version": 3,
                        "judge_minor_version": 1,
                        "score_type": "binary",
                        "bool_value": True,
                    }
                ],
            }
        ]
        results = runner.build_results(
            items, agent_traces={}, pass_condition_fn=lambda fields, scorers: True
        )
        runner.report_success("run-1", prepared, items, results)
        payload = client._request.call_args.kwargs["payload"]
        assert payload == {
            "successes": [{"evaluation_run_id": "er-1", "success": True}]
        }

    def test_unmatched_scorer_row_skipped(self):
        runner, client = _make_runner()
        items = [
            {
                "example_id": "ex-other",
                "data": {},
                "scorers": [
                    {
                        "judge_id": "j-unknown",
                        "judge_name": "unknown",
                        "score_type": "binary",
                        "bool_value": True,
                    }
                ],
            }
        ]
        results = runner.build_results(
            items, agent_traces={}, pass_condition_fn=lambda fields, scorers: True
        )
        runner.report_success("run-1", PREPARED, items, results)
        client._request.assert_not_called()

    def test_no_rows_skips_patch(self):
        runner, client = _make_runner()
        runner.report_success("run-1", PREPARED, [], [])
        client._request.assert_not_called()

    def test_maps_422_to_validation_error(self):
        runner, client = _make_runner()
        client._request.side_effect = JudgmentAPIError(
            422, "evaluation_run_id er-x does not belong to run-1", None
        )
        results = runner.build_results(
            ITEMS, agent_traces={}, pass_condition_fn=lambda fields, scorers: True
        )
        with pytest.raises(JudgmentValidationError):
            runner.report_success("run-1", PREPARED, ITEMS, results)


class TestRunOrchestration:
    def test_full_lifecycle_without_agent(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        outcome = runner.run(CONFIG, pass_condition_fn=lambda fields, scorers: True)
        assert isinstance(outcome, OfflineTestResult)
        assert outcome.test_run_id == "run-1"
        assert outcome.status == "completed"
        assert outcome.passed is True
        patches = _request_calls(client, "PATCH")
        assert len(patches) == 1
        args, kwargs = patches[0]
        assert args[1].endswith("/v1/projects/proj-1/test-runs/run-1/success")
        assert kwargs["payload"] == {
            "successes": [{"evaluation_run_id": "er-1", "success": True}]
        }
        client.post_projects_eval_results.assert_not_called()

    def test_skips_success_patch_without_pass_condition(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        outcome = runner.run(CONFIG)
        assert outcome.passed is None
        assert _request_calls(client, "PATCH") == []
        client.post_projects_eval_results.assert_not_called()

    def test_assert_test_requires_pass_condition(self):
        runner, _ = _make_runner()
        with pytest.raises(ValueError, match="pass_condition_fn"):
            runner.run(CONFIG, assert_test=True)

    def test_assert_test_raises_on_failure(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        with pytest.raises(JudgmentTestError):
            runner.run(
                CONFIG,
                pass_condition_fn=lambda fields, scorers: False,
                assert_test=True,
            )

    def test_assert_test_passes(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        outcome = runner.run(
            CONFIG,
            pass_condition_fn=lambda fields, scorers: True,
            assert_test=True,
        )
        assert outcome.passed is True

    def test_assert_test_raises_on_error_status(self):
        runner, client = _make_runner()
        _stub_lifecycle(client, status="error")
        with pytest.raises(JudgmentTestError, match="error"):
            runner.run(
                CONFIG,
                pass_condition_fn=lambda fields, scorers: True,
                assert_test=True,
            )

    def test_agent_function_invoked_per_fetched_example(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        calls = []

        def agent(input):
            calls.append(input)
            return f"answer to {input}"

        def fake_run_agent(agent_function, examples, progress=None):
            # examples come from the dataset fetch, not the prepare response
            assert [e["example_id"] for e in examples] == ["ex-1"]
            for example in examples:
                agent_function(**example["data"])
            return {"ex-1": "trace-abc"}

        with patch.object(OfflineTestRunner, "run_agent", side_effect=fake_run_agent):
            outcome = runner.run(CONFIG, agent_function=agent)

        assert calls == ["q1"]
        assert outcome.agent_offline_trace_ids == {"ex-1": "trace-abc"}
        # agent traces were attached at run creation; without a pass
        # condition nothing is reported back after the run
        assert _request_calls(client, "PATCH") == []
        client.post_projects_eval_results.assert_not_called()

    def test_agent_with_pass_condition_patches_success_only(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)

        def fake_run_agent(agent_function, examples, progress=None):
            return {"ex-1": "trace-abc"}

        with patch.object(OfflineTestRunner, "run_agent", side_effect=fake_run_agent):
            runner.run(
                CONFIG,
                agent_function=lambda input: input,
                pass_condition_fn=lambda fields, scorers: True,
            )

        patches = _request_calls(client, "PATCH")
        assert len(patches) == 1
        payload = patches[0].kwargs["payload"]
        # only evaluation_run_id + success cross the wire -- no trace echo
        assert payload == {
            "successes": [{"evaluation_run_id": "er-1", "success": True}]
        }
        client.post_projects_eval_results.assert_not_called()

    def test_agent_runs_before_run_creation_and_traces_attached(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        order = []

        def fake_run_agent(agent_function, examples, progress=None):
            order.append("agent")
            assert client.post_projects_test_runs.call_count == 0
            return {"ex-1": "trace-abc"}

        def fake_post(**kwargs):
            order.append("create")
            return dict(PREPARED)

        client.post_projects_test_runs.side_effect = fake_post

        with patch.object(OfflineTestRunner, "run_agent", side_effect=fake_run_agent):
            runner.run(CONFIG, agent_function=lambda input: input)

        assert order == ["agent", "create"]
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["agent_traces"] == [
            {"example_id": "ex-1", "agent_offline_trace_id": "trace-abc"}
        ]
        assert payload["dataset_version_number"] == 1

    def test_no_agent_traces_key_without_agent(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        runner.run(CONFIG)
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert "agent_traces" not in payload

    def test_run_pins_resolved_dataset_version(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)

        # default: latest version number is pinned explicitly
        runner.run(CONFIG)
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["dataset_version_number"] == 1

        # a version id passes through as dataset_version_id
        runner.run(CONFIG, dataset_version="v1")
        payload = client.post_projects_test_runs.call_args.kwargs["payload"]
        assert payload["dataset_version_id"] == "v1"
        assert "dataset_version_number" not in payload

    def test_traces_flushed_before_run_creation(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        client.api_key = "key"
        client.organization_id = "org"
        client.base_url = "http://localhost"

        tracer = MagicMock()

        def fake_create(**kwargs):
            dataset = kwargs["dataset"]

            def observe(func, span_type=None):
                def wrapper(**call_kwargs):
                    result = func(**call_kwargs)
                    dataset.append(Example.create(offline_trace_id="trace-abc"))
                    return result

                return wrapper

            tracer.observe = observe
            return tracer

        def fake_post(**kwargs):
            # by the time the run is created, the offline tracer has been
            # flushed and its provider shut down
            tracer.force_flush.assert_called_once()
            tracer._tracer_provider.shutdown.assert_called_once()
            return dict(PREPARED)

        client.post_projects_test_runs.side_effect = fake_post

        with patch(
            "judgeval.trace.offline_tracer.OfflineTracer.create",
            side_effect=fake_create,
        ):
            outcome = runner.run(CONFIG, agent_function=lambda input: input)

        assert outcome.agent_offline_trace_ids == {"ex-1": "trace-abc"}
        client.post_projects_test_runs.assert_called_once()

    def test_timeout_raises(self):
        runner, client = _make_runner()
        _stub_lifecycle(client, status="running")
        with pytest.raises(TimeoutError):
            runner.run(CONFIG, timeout_seconds=0)


class TestRunAgentLoop:
    def test_run_agent_wraps_with_offline_tracer(self):
        runner, client = _make_runner()
        client.api_key = "key"
        client.organization_id = "org"
        client.base_url = "http://localhost"

        tracer = MagicMock()
        captured_examples = []

        def fake_create(**kwargs):
            captured_examples.append(kwargs["dataset"])
            dataset = kwargs["dataset"]

            def observe(func, span_type=None):
                def wrapper(**call_kwargs):
                    result = func(**call_kwargs)
                    dataset.append(
                        Example.create(offline_trace_id=f"trace-{len(dataset)}")
                    )
                    return result

                return wrapper

            tracer.observe = observe
            return tracer

        examples = [
            {"example_id": "ex-1", "data": {"input": "q1"}},
            {"example_id": "ex-2", "data": {"input": "q2"}},
        ]

        seen = []

        def agent(input):
            seen.append(input)
            return input

        with patch(
            "judgeval.trace.offline_tracer.OfflineTracer.create",
            side_effect=fake_create,
        ):
            traces = runner.run_agent(agent, examples)

        assert seen == ["q1", "q2"]
        assert traces == {"ex-1": "trace-0", "ex-2": "trace-1"}
        tracer.force_flush.assert_called_once()

    def test_run_agent_continues_after_agent_error(self):
        runner, client = _make_runner()
        client.api_key = "key"
        client.organization_id = "org"
        client.base_url = "http://localhost"

        tracer = MagicMock()

        def fake_create(**kwargs):
            tracer.observe = lambda func, span_type=None: func
            return tracer

        def agent(input):
            if input == "q1":
                raise RuntimeError("boom")
            return input

        examples = [
            {"example_id": "ex-1", "data": {"input": "q1"}},
            {"example_id": "ex-2", "data": {"input": "q2"}},
        ]

        with patch(
            "judgeval.trace.offline_tracer.OfflineTracer.create",
            side_effect=fake_create,
        ):
            traces = runner.run_agent(agent, examples)

        assert traces == {}

    def test_run_agent_restores_previous_active_tracer(self):
        from judgeval.trace.judgment_tracer_provider import JudgmentTracerProvider

        runner, client = _make_runner()
        client.api_key = "key"
        client.organization_id = "org"
        client.base_url = "http://localhost"

        proxy = JudgmentTracerProvider.get_instance()
        original = proxy.get_active_tracer()
        previous = MagicMock()
        proxy.set_active(previous)

        tracer = MagicMock()

        def fake_create(**kwargs):
            assert kwargs["set_active"] is True
            proxy.set_active(tracer)
            tracer.observe = lambda func, span_type=None: func
            return tracer

        try:
            with patch(
                "judgeval.trace.offline_tracer.OfflineTracer.create",
                side_effect=fake_create,
            ):
                runner.run_agent(
                    lambda input: input,
                    [{"example_id": "ex-1", "data": {"input": "q1"}}],
                )
            assert proxy.get_active_tracer() is previous
            tracer._tracer_provider.shutdown.assert_called_once()

            # restored even when the agent loop raises
            with patch(
                "judgeval.trace.offline_tracer.OfflineTracer.create",
                side_effect=fake_create,
            ):
                with pytest.raises(TypeError, match="does not accept"):
                    runner.run_agent(
                        lambda input: input,
                        [{"example_id": "ex-1", "data": {"other": 1}}],
                    )
            assert proxy.get_active_tracer() is previous
        finally:
            proxy.deregister(previous)
            proxy.deregister(tracer)
            proxy.restore_active(original)

    def test_run_agent_signature_mismatch_raises(self):
        runner, client = _make_runner()
        client.api_key = "key"
        client.organization_id = "org"
        client.base_url = "http://localhost"

        tracer = MagicMock()
        tracer.observe = lambda func, span_type=None: func

        def agent(question):
            return question

        with patch(
            "judgeval.trace.offline_tracer.OfflineTracer.create",
            return_value=tracer,
        ):
            with pytest.raises(TypeError, match="does not accept"):
                runner.run_agent(
                    agent, [{"example_id": "ex-1", "data": {"input": "q1"}}]
                )
