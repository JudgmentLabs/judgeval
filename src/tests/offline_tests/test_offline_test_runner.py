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
    has_duplicate_judges,
    normalize_judge_versions,
)
from judgeval.offline_tests.types import OfflineTestResult, TestConfig

CONFIG = TestConfig(id="cfg-1", name="nightly", dataset_id="d1")

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
    runner = OfflineTestRunner(
        client=client, project_id="proj-1", project_name="test-project"
    )
    return runner, client


def _stub_lifecycle(client, status="completed"):
    client.post_projects_test_runs.return_value = dict(PREPARED)
    client.get_projects_test_runs_by_test_run_id.return_value = {
        "test_run": {"id": "run-1", "status": status},
        "ui_results_url": "https://app/tests/run-1",
    }
    client.get_projects_test_runs_by_test_run_id_items.return_value = {
        "results": [json.loads(json.dumps(item)) for item in ITEMS],
        "ui_results_url": "https://app/tests/run-1",
    }
    client.post_projects_eval_results.return_value = {
        "test_run_id": "run-1",
        "ui_results_url": "https://app/tests/run-1",
    }


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

    def test_duplicate_judges_detected(self):
        assert has_duplicate_judges(
            [{"name": "j", "tag": "prod"}, {"name": "j", "version": "1.0"}]
        )
        assert not has_duplicate_judges([{"name": "a"}, {"name": "b"}])


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

    def test_versioned_results_auto_set_for_duplicate_judges(self):
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
        assert payload["versioned_results"] is True

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


class TestReportResults:
    def test_echoes_evaluation_run_id_success_and_agent_trace(self):
        runner, client = _make_runner()
        results = runner.build_results(
            ITEMS,
            agent_traces={"ex-1": "trace-abc"},
            pass_condition_fn=lambda fields, scorers: True,
        )
        runner.report_results(
            "run-1", PREPARED, ITEMS, results, agent_traces={"ex-1": "trace-abc"}
        )
        payload = client.post_projects_eval_results.call_args.kwargs["payload"]
        assert payload["test_run_id"] == "run-1"
        wire = payload["results"][0]
        scorer = wire["scorers_data"][0]
        assert scorer["evaluation_run_id"] == "er-1"
        assert scorer["success"] is True
        assert scorer["scorer_name"] == "helpfulness"
        assert scorer["score_type"] == "binary"
        assert scorer["bool_value"] is True
        assert scorer["reason"] == {"text": "looks good"}
        assert wire["data_object"]["example_id"] == "ex-1"
        assert wire["data_object"]["agent_offline_trace_id"] == "trace-abc"
        assert wire["data_object"]["input"] == "q1"
        assert payload["run"]["project_id"] == "proj-1"

    def test_no_rows_skips_post(self):
        runner, client = _make_runner()
        runner.report_results("run-1", PREPARED, [], [], agent_traces={})
        client.post_projects_eval_results.assert_not_called()


class TestRunOrchestration:
    def test_full_lifecycle_without_agent(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        outcome = runner.run(CONFIG, pass_condition_fn=lambda fields, scorers: True)
        assert isinstance(outcome, OfflineTestResult)
        assert outcome.test_run_id == "run-1"
        assert outcome.status == "completed"
        assert outcome.passed is True
        client.post_projects_eval_results.assert_called_once()

    def test_skips_reporting_without_pass_condition_or_agent(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        outcome = runner.run(CONFIG)
        assert outcome.passed is None
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

    def test_agent_function_invoked_per_example(self):
        runner, client = _make_runner()
        _stub_lifecycle(client)
        calls = []

        def agent(input):
            calls.append(input)
            return f"answer to {input}"

        def fake_run_agent(agent_function, examples, progress=None):
            for example in examples:
                agent_function(**example["data"])
            return {"ex-1": "trace-abc"}

        with patch.object(OfflineTestRunner, "run_agent", side_effect=fake_run_agent):
            outcome = runner.run(CONFIG, agent_function=agent)

        assert calls == ["q1"]
        assert outcome.agent_offline_trace_ids == {"ex-1": "trace-abc"}
        # agent traces are reported even without a pass condition
        client.post_projects_eval_results.assert_called_once()
        payload = client.post_projects_eval_results.call_args.kwargs["payload"]
        data_object = payload["results"][0]["data_object"]
        assert data_object["agent_offline_trace_id"] == "trace-abc"

    def test_timeout_raises(self):
        runner, client = _make_runner()
        client.post_projects_test_runs.return_value = dict(PREPARED)
        client.get_projects_test_runs_by_test_run_id.return_value = {
            "test_run": {"id": "run-1", "status": "running"}
        }
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
