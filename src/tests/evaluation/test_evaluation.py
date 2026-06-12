from __future__ import annotations

from unittest.mock import MagicMock

from judgeval.data.example import Example
from judgeval.evaluation.evaluation import Evaluation


def _make_evaluation():
    client = MagicMock()
    return Evaluation(client=client, project_id="proj-1", project_name="test-project")


class TestEvaluationRouting:
    def test_mixed_scorers_returns_empty(self):
        from judgeval.judges import Judge

        eval_ = _make_evaluation()
        judge_mock = MagicMock(spec=Judge)
        examples = [Example.create(input="q", output="a")]
        result = eval_.run(
            examples=examples,
            scorers=["hosted-scorer", judge_mock],
            eval_run_name="run-1",
        )
        assert result == []

    def test_empty_scorers_returns_empty(self):
        eval_ = _make_evaluation()
        result = eval_.run(
            examples=[Example.create(input="q")],
            scorers=[],
            eval_run_name="run-1",
        )
        assert result == []

    def test_hosted_scorers_delegates(self):
        eval_ = _make_evaluation()
        eval_._hosted.run = MagicMock(return_value=[])
        eval_.run(
            examples=[Example.create(input="q")],
            scorers=["scorer-a"],
            eval_run_name="run-1",
        )
        eval_._hosted.run.assert_called_once()

    def test_local_scorers_delegates(self):
        from judgeval.judges import Judge

        eval_ = _make_evaluation()
        eval_._local.run = MagicMock(return_value=[])
        judge = MagicMock(spec=Judge)
        eval_.run(
            examples=[Example.create(input="q")],
            scorers=[judge],
            eval_run_name="run-1",
        )
        eval_._local.run.assert_called_once()


class TestHostedSubmitAndPoll:
    def test_hosted_scorers_queue_and_poll_results(self):
        client = MagicMock()
        eval_ = Evaluation(
            client=client, project_id="proj-1", project_name="test-project"
        )
        client.get_projects_experiments_by_run_id.return_value = {
            "results": [
                {
                    "scorers": [
                        {
                            "judge_name": "faithfulness",
                            "score_type": "binary",
                            "bool_value": True,
                            "success": True,
                        }
                    ]
                }
            ],
            "ui_results_url": "https://app/experiments/run-1",
        }
        results = eval_.run(
            examples=[Example.create(input="q")],
            scorers=["faithfulness"],
            eval_run_name="run-1",
        )
        client.post_projects_eval_queue_examples.assert_called_once()
        queue_kwargs = client.post_projects_eval_queue_examples.call_args.kwargs
        assert queue_kwargs["project_id"] == "proj-1"
        assert queue_kwargs["payload"]["judgment_scorers"] == [{"name": "faithfulness"}]
        client.get_projects_experiments_by_run_id.assert_called()
        assert len(results) == 1
        scorer = results[0].scorers_data[0]
        assert scorer.name == "faithfulness"
        assert scorer.value == "Yes"
        assert scorer.success is True


class TestEvaluationFactory:
    def test_create_returns_evaluation(self):
        from judgeval.evaluation.evaluation_factory import EvaluationFactory

        client = MagicMock()
        factory = EvaluationFactory(client=client, project_id="p", project_name="n")
        evaluation = factory.create()
        assert isinstance(evaluation, Evaluation)
