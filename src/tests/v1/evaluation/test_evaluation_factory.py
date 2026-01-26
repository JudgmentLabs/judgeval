import pytest
from unittest.mock import MagicMock
from judgeval.v1.evaluation.evaluation_factory import EvaluationFactory
from judgeval.v1.evaluation.evaluation import Evaluation
from judgeval.v1.data.example import Example
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def evaluation_factory(mock_client):
    return EvaluationFactory(mock_client)


@pytest.fixture
def sample_examples():
    return [
        Example(name="ex1")
        .set_property("input", "test")
        .set_property("actual_output", "result"),
    ]


@pytest.fixture
def sample_scorers():
    return [AnswerRelevancyScorer(threshold=0.5)]


def test_factory_create(evaluation_factory, mock_client):
    evaluation = evaluation_factory.create()

    assert isinstance(evaluation, Evaluation)
    assert evaluation._client == mock_client


def test_factory_run(evaluation_factory, mock_client, sample_examples, sample_scorers):
    mock_client.fetch_experiment_run.return_value = {
        "results": [
            {
                "scorers": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.5,
                        "success": True,
                        "score": 0.9,
                        "reason": "Relevant",
                        "evaluation_model": "gpt-4o-mini",
                        "error": None,
                        "additional_metadata": {},
                        "scorer_data_id": "1",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            }
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    evaluation = evaluation_factory.create()
    results = evaluation.run(
        examples=sample_examples,
        scorers=sample_scorers,
        project_name="test_project",
        eval_run_name="test_run",
    )

    assert len(results) == 1
    assert results[0].success
    mock_client.add_to_run_eval_queue_examples.assert_called_once()
