"""
Tests to verify that all Pydantic models properly reject invalid keys.
"""

import pytest
from pydantic import ValidationError

# Import all the models that should reject invalid keys
from judgeval.data import Example
from judgeval.scorers import BaseScorer, APIScorerConfig
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.trace_run import TraceRun
from judgeval.judgment_client import EvalRunRequestBody, DeleteEvalRunRequestBody
from judgeval.utils.alerts import AlertResult, AlertStatus
from judgeval.common.utils import CustomModelParameters, ChatCompletionRequest


class TestValidationFixes:
    """Test that all Pydantic models reject invalid keys."""

    def test_example_rejects_invalid_keys(self):
        """Test that Example rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            Example(invalid_key='test')
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_base_scorer_rejects_invalid_keys(self):
        """Test that BaseScorer rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            BaseScorer(score_type='test', invalid_key='test')
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_api_scorer_config_rejects_invalid_keys(self):
        """Test that APIScorerConfig rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            APIScorerConfig(score_type='faithfulness', invalid_key='test')
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_evaluation_run_rejects_invalid_keys(self):
        """Test that EvaluationRun rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationRun(
                examples=[Example(input='test')],
                scorers=[APIScorerConfig(score_type='faithfulness')],
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_trace_run_rejects_invalid_keys(self):
        """Test that TraceRun rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            TraceRun(
                scorers=[APIScorerConfig(score_type='faithfulness')],
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_eval_run_request_body_rejects_invalid_keys(self):
        """Test that EvalRunRequestBody rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            EvalRunRequestBody(
                eval_name='test',
                project_name='test',
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_delete_eval_run_request_body_rejects_invalid_keys(self):
        """Test that DeleteEvalRunRequestBody rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            DeleteEvalRunRequestBody(
                eval_names=['test'],
                project_name='test',
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_alert_result_rejects_invalid_keys(self):
        """Test that AlertResult rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            AlertResult(
                rule_name='test',
                status=AlertStatus.TRIGGERED,
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_custom_model_parameters_rejects_invalid_keys(self):
        """Test that CustomModelParameters rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            CustomModelParameters(
                model_name='test',
                secret_key='test',
                litellm_base_url='test',
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_chat_completion_request_rejects_invalid_keys(self):
        """Test that ChatCompletionRequest rejects invalid keys."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                model='gpt-3.5-turbo',
                messages=[{'role': 'user', 'content': 'test'}],
                invalid_key='test'
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_valid_models_still_work(self):
        """Test that valid usage still works for all models."""
        # Test Example
        example = Example(input='test')
        assert example.input == 'test'

        # Test BaseScorer
        scorer = BaseScorer(score_type='test')
        assert scorer.score_type == 'test'

        # Test APIScorerConfig
        api_scorer = APIScorerConfig(score_type='faithfulness')
        assert api_scorer.score_type == 'faithfulness'

        # Test EvaluationRun
        eval_run = EvaluationRun(
            examples=[example],
            scorers=[api_scorer]
        )
        assert len(eval_run.examples) == 1
        assert len(eval_run.scorers) == 1

        # Test AlertResult
        alert = AlertResult(
            rule_name='test',
            status=AlertStatus.TRIGGERED
        )
        assert alert.rule_name == 'test'
        assert alert.status == AlertStatus.TRIGGERED