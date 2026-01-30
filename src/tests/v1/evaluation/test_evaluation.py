import pytest
from unittest.mock import MagicMock
from judgeval.v1.evaluation.evaluation import Evaluation
from judgeval.v1.data.example import Example
from judgeval.v1.data.scoring_result import ScoringResult
from judgeval.v1.data.scorer_data import ScorerData
from judgeval.v1.scorers.built_in.answer_relevancy import AnswerRelevancyScorer


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def evaluation(mock_client):
    return Evaluation(
        client=mock_client,
        project_id="test_project_id",
        project_name="test_project",
    )


@pytest.fixture
def sample_examples():
    return [
        Example(name="ex1")
        .set_property("input", "What is 2+2?")
        .set_property("actual_output", "4"),
        Example(name="ex2")
        .set_property("input", "What is 3+3?")
        .set_property("actual_output", "6"),
    ]


@pytest.fixture
def sample_scorers():
    return [AnswerRelevancyScorer(threshold=0.5)]


def test_evaluation_run_success(
    evaluation, mock_client, sample_examples, sample_scorers
):
    mock_client.get_projects_experiments_by_run_id.return_value = {
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
            },
            {
                "scorers": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.5,
                        "success": True,
                        "score": 0.8,
                        "reason": "Relevant",
                        "evaluation_model": "gpt-4o-mini",
                        "error": None,
                        "additional_metadata": {},
                        "scorer_data_id": "2",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            },
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    results = evaluation.run(
        examples=sample_examples,
        scorers=sample_scorers,
        eval_run_name="test_run",
        model="gpt-4o-mini",
        timeout_seconds=10,
    )

    assert len(results) == 2
    assert all(isinstance(r, ScoringResult) for r in results)
    assert all(r.success for r in results)
    mock_client.post_projects_eval_queue_examples.assert_called_once()
    mock_client.get_projects_experiments_by_run_id.assert_called()


def test_evaluation_run_with_failures(
    evaluation, mock_client, sample_examples, sample_scorers
):
    mock_client.get_projects_experiments_by_run_id.return_value = {
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
            },
            {
                "scorers": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.5,
                        "success": False,
                        "score": 0.3,
                        "reason": "Not relevant",
                        "evaluation_model": "gpt-4o-mini",
                        "error": None,
                        "additional_metadata": {},
                        "scorer_data_id": "2",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            },
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    results = evaluation.run(
        examples=sample_examples,
        scorers=sample_scorers,
        eval_run_name="test_run",
    )

    assert len(results) == 2
    assert results[0].success
    assert not results[1].success


def test_evaluation_run_with_assert_mode_success(
    evaluation, mock_client, sample_examples, sample_scorers
):
    mock_client.get_projects_experiments_by_run_id.return_value = {
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
            },
            {
                "scorers": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.5,
                        "success": True,
                        "score": 0.8,
                        "reason": "Relevant",
                        "evaluation_model": "gpt-4o-mini",
                        "error": None,
                        "additional_metadata": {},
                        "scorer_data_id": "2",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            },
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    results = evaluation.run(
        examples=sample_examples,
        scorers=sample_scorers,
        eval_run_name="test_run",
        assert_test=True,
    )

    assert len(results) == 2
    assert all(r.success for r in results)


def test_evaluation_run_with_assert_mode_failure(
    evaluation, mock_client, sample_examples, sample_scorers
):
    mock_client.get_projects_experiments_by_run_id.return_value = {
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
            },
            {
                "scorers": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.5,
                        "success": False,
                        "score": 0.3,
                        "reason": "Not relevant",
                        "evaluation_model": "gpt-4o-mini",
                        "error": None,
                        "additional_metadata": {},
                        "scorer_data_id": "2",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            },
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    with pytest.raises(AssertionError, match="Evaluation failed"):
        evaluation.run(
            examples=sample_examples,
            scorers=sample_scorers,
            eval_run_name="test_run",
            assert_test=True,
        )


def test_evaluation_timeout(evaluation, mock_client, sample_examples, sample_scorers):
    mock_client.get_projects_experiments_by_run_id.return_value = {
        "results": [],
        "ui_results_url": "http://test.example.com/results",
    }

    with pytest.raises(TimeoutError, match="Evaluation timed out"):
        evaluation.run(
            examples=sample_examples,
            scorers=sample_scorers,
            eval_run_name="test_run",
            timeout_seconds=1,
        )


def test_evaluation_scorer_data_parsing(
    evaluation, mock_client, sample_examples, sample_scorers
):
    mock_client.get_projects_experiments_by_run_id.return_value = {
        "results": [
            {
                "scorers": [
                    {
                        "name": "Test Scorer",
                        "threshold": 0.7,
                        "success": True,
                        "score": 0.85,
                        "reason": "Test reason",
                        "evaluation_model": "test-model",
                        "error": None,
                        "additional_metadata": {"key": "value"},
                        "scorer_data_id": "test-id",
                        "minimum_score_range": 0,
                        "maximum_score_range": 1,
                    }
                ]
            }
        ],
        "ui_results_url": "http://test.example.com/results",
    }

    results = evaluation.run(
        examples=[sample_examples[0]],
        scorers=sample_scorers,
        eval_run_name="test_run",
    )

    assert len(results) == 1
    scorer_data = results[0].scorers_data[0]
    assert isinstance(scorer_data, ScorerData)
    assert scorer_data.name == "Test Scorer"
    assert scorer_data.threshold == 0.7
    assert scorer_data.success is True
    assert scorer_data.score == 0.85
    assert scorer_data.reason == "Test reason"
    assert scorer_data.evaluation_model == "test-model"
    assert scorer_data.additional_metadata == {"key": "value"}
    assert scorer_data.id == "test-id"
