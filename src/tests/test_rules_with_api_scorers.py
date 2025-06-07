"""Tests to verify rules work correctly with API scorers."""

import pytest
from unittest.mock import MagicMock, patch

from judgeval.rules import Rule, Condition, RulesEngine
from judgeval.utils.alerts import AlertStatus
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example

class TestRulesWithAPIScorers:
    
    def test_rules_with_api_scorers(self):
        """Test that rules work correctly with API scorers."""
        # Create API scorers
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        
        # Create rules with API scorers
        rules = {
            "quality_check": Rule(
                name="Quality Check",
                conditions=[
                    Condition(metric=faithfulness_scorer),
                    Condition(metric=relevancy_scorer)
                ],
                combine_type="all"
            )
        }
        
        # Create RulesEngine
        engine = RulesEngine(rules)
        
        # Create scores dictionary that matches the metric names
        # Important: the key should match what's returned by the metric_name property
        scores = {
            "faithfulness": 0.8,  # This should match faithfulness_scorer.score_type
            "answer_relevancy": 0.9  # This should match relevancy_scorer.score_type
        }
        
        # Print the metric names we're expecting
        print(f"Faithfulness metric name: {faithfulness_scorer.score_type}")
        print(f"Relevancy metric name: {relevancy_scorer.score_type}")
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Verify results
        assert "quality_check" in results
        assert results["quality_check"].status == AlertStatus.TRIGGERED
        assert len(results["quality_check"].conditions_result) == 2
        assert results["quality_check"].conditions_result[0]["passed"] is True
        assert results["quality_check"].conditions_result[1]["passed"] is True
    
    def test_rule_metric_name_resolution(self):
        """Test that rule condition metrics correctly resolve to their score_type."""
        # Create API scorers
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create condition with scorer
        condition = Condition(metric=faithfulness_scorer)
        
        # Check the metric_name property
        assert condition.metric_name == "faithfulness"
        
        # Check with direct assignment
        condition.metric = faithfulness_scorer
        assert condition.metric_name == "faithfulness"
    
    @patch('judgeval.judgment_client.run_eval')
    def test_judgment_client_with_rules(self, mock_run_eval):
        """Test that JudgmentClient works with API scorers in rules."""
        # Mock the run_eval function
        mock_result = MagicMock()
        mock_result.alert_results = {
            "rule_0": {
                "status": "triggered",
                "rule_name": "Quality Check",
                "conditions_result": [
                    {"metric": "faithfulness", "value": 0.8, "threshold": 0.7, "passed": True, "skipped": False},
                    {"metric": "answer_relevancy", "value": 0.9, "threshold": 0.8, "passed": True, "skipped": False}
                ]
            }
        }
        mock_run_eval.return_value = [mock_result]
        
        # Create client with patched _validate_api_key method
        client = JudgmentClient(judgment_api_key="test_key")
    
        # Create example
        example = Example(
            input="Test input",
            actual_output="Test output",
            expected_output="Expected output"
        )
    
        # Create scorers
        scorers = [
            FaithfulnessScorer(threshold=0.7),
            AnswerRelevancyScorer(threshold=0.8)
        ]
    
        # Create rules
        rules = [
            Rule(
                name="Quality Check",
                conditions=[
                    Condition(metric=FaithfulnessScorer(threshold=0.7)),
                    Condition(metric=AnswerRelevancyScorer(threshold=0.8))
                ],
                combine_type="all"
            )
        ]
    
        # Run evaluation
        result = client.run_evaluation(
            examples=[example],
            scorers=scorers,
            model="gpt-3.5-turbo",
            rules=rules
        )
    
        # Verify run_eval was called with the expected arguments
        assert mock_run_eval.called
        call_args = mock_run_eval.call_args[0][0]
        assert hasattr(call_args, 'rules')
                
    def test_validation_with_correct_import(self):
        """Test that using JudgevalScorer with rules raises an error (with correct import)."""
        # Import JudgevalScorer from the correct path
        from judgeval.scorers.judgeval_scorer import JudgevalScorer
        
        # Create a mock JudgevalScorer instance
        mock_judgeval_scorer = MagicMock(spec=JudgevalScorer)
        # Ensure isinstance checks will work properly
        mock_judgeval_scorer.__class__ = JudgevalScorer
        
        # Create client with patched _validate_api_key method
        client = JudgmentClient(judgment_api_key="test_key")
    
        # Create example
        example = Example(
            input="Test input",
            actual_output="Test output",
            expected_output="Expected output"
        )
    
        # Create scorers - mixing API scorer and JudgevalScorer
        scorers = [
            FaithfulnessScorer(threshold=0.7),  # API scorer
            mock_judgeval_scorer  # JudgevalScorer (mocked)
        ]
    
        # Create rules
        rules = [
            Rule(
                name="Quality Check",
                conditions=[
                    Condition(metric=FaithfulnessScorer(threshold=0.7))
                ],
                combine_type="all"
            )
        ]
    
        # Run evaluation should raise ValueError
        with pytest.raises(ValueError, match="Cannot use Judgeval scorers .* when using rules"):
            client.run_evaluation(
                examples=[example],
                scorers=scorers,
                model="gpt-3.5-turbo",
                rules=rules
            ) 
