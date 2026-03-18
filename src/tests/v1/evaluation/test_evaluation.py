from __future__ import annotations

from unittest.mock import MagicMock


from judgeval.v1.data.example import Example
from judgeval.v1.evaluation.evaluation import Evaluation


def _make_evaluation():
    client = MagicMock()
    return Evaluation(client=client, project_id="proj-1", project_name="test-project")


class TestEvaluationRouting:
    def test_mixed_scorers_returns_empty(self):
        from judgeval.v1.judges import Judge

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
        from judgeval.v1.judges import Judge

        eval_ = _make_evaluation()
        eval_._local.run = MagicMock(return_value=[])
        judge = MagicMock(spec=Judge)
        eval_.run(
            examples=[Example.create(input="q")],
            scorers=[judge],
            eval_run_name="run-1",
        )
        eval_._local.run.assert_called_once()


class TestEvaluationFactory:
    def test_create_returns_evaluation(self):
        from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory

        client = MagicMock()
        factory = EvaluationFactory(client=client, project_id="p", project_name="n")
        evaluation = factory.create()
        assert isinstance(evaluation, Evaluation)
