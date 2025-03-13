"""Integration tests for the notification functionality without using Tracer."""

import pytest
from unittest.mock import patch, MagicMock
import os
import asyncio
import json
from typing import Dict, List, Optional

from judgeval.rules import Rule, Condition, NotificationConfig, AlertStatus, RulesEngine
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer, AnswerCorrectnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example


class TestDirectNotificationIntegration:
    """Integration tests for notifications using the rules engine directly."""
    
    def test_rules_engine_with_notification(self):
        """Test rules engine with notification configurations."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        # Manually set score_type if needed for testing
        if not hasattr(faithfulness_scorer, 'score_type'):
            faithfulness_scorer.score_type = "faithfulness"
        
        # Create notification config
        notification_config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Rule '{rule_name}' was triggered with score {score}",
            email_addresses=["test@example.com"],
            send_at=None
        )
        
        # Create rule with notification config
        rule = Rule(
            name="Faithfulness Check",
            description="Check if faithfulness meets threshold",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification_config
        )
        
        # Create engine with the rule
        engine = RulesEngine({"test_rule": rule})
        
        # Create scores that will trigger the rule
        scores = {"faithfulness": 0.8}
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Verify the notification is included in the triggered rule
        assert "test_rule" in results
        assert results["test_rule"].status == AlertStatus.TRIGGERED
        assert results["test_rule"].notification is not None
        assert results["test_rule"].notification.enabled is True
        assert results["test_rule"].notification.communication_methods == ["slack", "email"]
        assert results["test_rule"].notification.email_addresses == ["test@example.com"]
    
    def test_rules_engine_parallel_evaluation_with_notifications(self):
        """Test rules engine's parallel evaluation with notification configurations."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        # Manually set score_type if needed for testing
        if not hasattr(faithfulness_scorer, 'score_type'):
            faithfulness_scorer.score_type = "faithfulness"
        
        # Create notification config
        notification_config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Rule '{rule_name}' was triggered with score {score}",
            email_addresses=["test@example.com"],
            send_at=None
        )
        
        # Create rule with notification config
        rule = Rule(
            name="Faithfulness Check",
            description="Check if faithfulness meets threshold",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification_config
        )
        
        # Create engine with the rule
        engine = RulesEngine({"test_rule": rule})
        
        # Create example scores and metadata for multiple examples
        example_scores = {
            "example1": {"faithfulness": 0.8},  # Will trigger
            "example2": {"faithfulness": 0.6}   # Won't trigger
        }
        
        example_metadata = {
            "example1": {"id": "example1", "timestamp": "20240101_120000"},
            "example2": {"id": "example2", "timestamp": "20240101_120100"}
        }
        
        # Run parallel evaluation
        async def run_parallel_evaluation():
            results = await engine.evaluate_rules_parallel(
                example_scores=example_scores,
                example_metadata=example_metadata
            )
            return results
        
        results = asyncio.run(run_parallel_evaluation())
        
        # Verify results for example1 (triggered)
        assert "example1" in results
        assert "test_rule" in results["example1"]
        assert results["example1"]["test_rule"].status == AlertStatus.TRIGGERED
        assert results["example1"]["test_rule"].notification is not None
        assert results["example1"]["test_rule"].notification.enabled is True
        
        # Verify results for example2 (not triggered)
        assert "example2" in results
        assert "test_rule" in results["example2"]
        assert results["example2"]["test_rule"].status == AlertStatus.NOT_TRIGGERED
        assert results["example2"]["test_rule"].notification is None
    
    def test_multiple_rules_with_different_notification_configs(self):
        """Test multiple rules with different notification configurations."""
        # Use direct scorer instances
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        # Manually set score_type if needed for testing
        if not hasattr(faithfulness_scorer, 'score_type'):
            faithfulness_scorer.score_type = "faithfulness"
        
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        # Manually set score_type if needed for testing
        if not hasattr(relevancy_scorer, 'score_type'):
            relevancy_scorer.score_type = "answer_relevancy"
        
        # Create notification configs
        notification1 = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            message_template="Slack notification",
            email_addresses=None,
            send_at=None
        )
        
        notification2 = NotificationConfig(
            enabled=True,
            communication_methods=["email"],
            message_template="Email notification",
            email_addresses=["email@example.com"],
            send_at=None
        )
        
        # Create rules with different notification configs
        rule1 = Rule(
            name="Faithfulness Rule",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification1
        )
        
        rule2 = Rule(
            name="Relevancy Rule",
            conditions=[
                Condition(metric=relevancy_scorer)
            ],
            combine_type="all",
            notification=notification2
        )
        
        # Create engine with both rules
        engine = RulesEngine({
            "rule1": rule1,
            "rule2": rule2
        })
        
        # Create scores that will trigger both rules
        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.9
        }
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Verify rule1 has the correct notification config
        assert "rule1" in results
        assert results["rule1"].status == AlertStatus.TRIGGERED
        assert results["rule1"].notification is not None
        assert results["rule1"].notification.communication_methods == ["slack"]
        assert results["rule1"].notification.message_template == "Slack notification"
        
        # Verify rule2 has the correct notification config
        assert "rule2" in results
        assert results["rule2"].status == AlertStatus.TRIGGERED
        assert results["rule2"].notification is not None
        assert results["rule2"].notification.communication_methods == ["email"]
        assert results["rule2"].notification.message_template == "Email notification"
        assert results["rule2"].notification.email_addresses == ["email@example.com"]


@patch('requests.post')
class TestNotificationWithAPICalls:
    """Tests for notifications with API calls to external services."""
    
    @patch('judgeval.scorers.judgeval_scorers.ScorerWrapper.load_implementation')
    def test_judgment_client_with_notification_rules(self, mock_load_implementation, mock_post):
        """Test JudgmentClient with notification rules."""
        # Mock API responses
        mock_auth_response = MagicMock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {"detail": {"user_name": "test_user"}}
        
        # Create proper structured mock response with required fields
        mock_eval_response = MagicMock()
        mock_eval_response.status_code = 200
        mock_eval_response.json.return_value = {
            "results": [
                {
                    "example_id": "example_123",
                    "success": True,  # Required field
                    "scores": {
                        "faithfulness": 0.8
                    },
                    "scorers_data": [  # Required field with correct structure
                        {
                            "scorer_type": "faithfulness",
                            "score": 0.8,
                            "explanation": "Test explanation",
                            "success": True,
                            "error": None,
                            "name": "faithfulness",  # Required field
                            "threshold": 0.7  # Required field
                        }
                    ],
                    "alert_results": {
                        "rule_1": {
                            "rule_name": "Faithfulness Rule",
                            "status": "triggered",
                            "conditions_result": [
                                {"metric": "faithfulness", "value": 0.8, "threshold": 0.7, "passed": True}
                            ],
                            "notification": {
                                "enabled": True,
                                "communication_methods": ["slack", "email"],
                                "message_template": "Rule triggered",
                                "email_addresses": ["test@example.com"]
                            }
                        }
                    }
                }
            ]
        }
        
        # Configure mock to return different responses for different API calls
        def mock_post_side_effect(url, *args, **kwargs):
            if "/auth/validate" in url:
                return mock_auth_response
            else:
                return mock_eval_response
        
        mock_post.side_effect = mock_post_side_effect
        
        # Create a mock implementation that will be returned
        mock_implementation = MagicMock()
        mock_implementation.score_type = "faithfulness"
        mock_load_implementation.return_value = mock_implementation
        
        # Create JudgmentClient
        with patch.object(JudgmentClient, '_validate_api_key', return_value=(True, {"detail": {"user_name": "test_user"}})):
            client = JudgmentClient(judgment_api_key="test_key")
            
            # Create example
            example = Example(
                input="Test input",
                actual_output="Test output",
                expected_output="Expected output"
            )
            
            # Create notification config
            notification = NotificationConfig(
                enabled=True,
                communication_methods=["slack", "email"],
                message_template="Rule triggered",
                email_addresses=["test@example.com"]
            )
            
            # Create rule with notification
            rule = Rule(
                name="Faithfulness Rule",
                conditions=[
                    Condition(metric=FaithfulnessScorer(threshold=0.7))
                ],
                combine_type="all",
                notification=notification
            )
            
            # Run evaluation
            result = client.run_evaluation(
                examples=[example],
                scorers=[FaithfulnessScorer(threshold=0.7)],
                model="gpt-3.5-turbo",
                rules=[rule]
            )
            
            # Verify the API was called
            assert mock_post.called
            
            # Get the evaluation call (should be at least the second call)
            eval_call = mock_post.call_args_list[1]
            
            # Extract the JSON payload - use 'json' parameter instead of 'data'
            if 'json' in eval_call[1]:
                payload = eval_call[1]["json"]
            else:
                payload = {}  # Default empty to avoid errors
            
            # Verify rules and notification config were included in the API call
            assert "rules" in payload
            assert len(payload["rules"]) == 1
            rule_data = payload["rules"][0]
            assert "notification" in rule_data
            assert rule_data["notification"]["enabled"] is True
            assert rule_data["notification"]["communication_methods"] == ["slack", "email"]
            assert rule_data["notification"]["email_addresses"] == ["test@example.com"]
    
    @patch('judgeval.scorers.judgeval_scorers.ScorerWrapper.load_implementation')
    def test_notification_with_multiple_methods(self, mock_load_implementation, mock_post):
        """Test notifications with multiple communication methods."""
        # Mock API responses (same as before but with multiple methods and proper structure)
        mock_auth_response = MagicMock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {"detail": {"user_name": "test_user"}}
        
        mock_eval_response = MagicMock()
        mock_eval_response.status_code = 200
        mock_eval_response.json.return_value = {
            "results": [
                {
                    "example_id": "example_123",
                    "success": True,  # Required field
                    "scores": {
                        "faithfulness": 0.8
                    },
                    "scorers_data": [  # Required field with correct structure
                        {
                            "scorer_type": "faithfulness",
                            "score": 0.8,
                            "explanation": "Test explanation",
                            "success": True,
                            "error": None,
                            "name": "faithfulness",  # Required field
                            "threshold": 0.7  # Required field
                        }
                    ],
                    "alert_results": {
                        "rule_1": {
                            "rule_name": "Faithfulness Rule",
                            "status": "triggered",
                            "conditions_result": [
                                {"metric": "faithfulness", "value": 0.8, "threshold": 0.7, "passed": True}
                            ],
                            "notification": {
                                "enabled": True,
                                "communication_methods": ["slack", "email", "broadcast_slack", "broadcast_email"],
                                "message_template": "Rule triggered with multiple methods",
                                "email_addresses": ["test1@example.com", "test2@example.com"]
                            }
                        }
                    }
                }
            ]
        }
        
        def mock_post_side_effect(url, *args, **kwargs):
            if "/auth/validate" in url:
                return mock_auth_response
            else:
                return mock_eval_response
        
        mock_post.side_effect = mock_post_side_effect
            
        # Create a mock implementation that will be returned
        mock_implementation = MagicMock()
        mock_implementation.score_type = "faithfulness"
        mock_load_implementation.return_value = mock_implementation
        
        # Create JudgmentClient
        with patch.object(JudgmentClient, '_validate_api_key', return_value=(True, {"detail": {"user_name": "test_user"}})):
            client = JudgmentClient(judgment_api_key="test_key")
            
            # Create example
            example = Example(
                input="Test input",
                actual_output="Test output",
                expected_output="Expected output"
            )
            
            # Create notification config with multiple methods
            notification = NotificationConfig(
                enabled=True,
                communication_methods=["slack", "email", "broadcast_slack", "broadcast_email"],
                message_template="Rule triggered with multiple methods",
                email_addresses=["test1@example.com", "test2@example.com"]
            )
            
            # Create rule with notification
            rule = Rule(
                name="Faithfulness Rule",
                conditions=[
                    Condition(metric=FaithfulnessScorer(threshold=0.7))
                ],
                combine_type="all",
                notification=notification
            )
            
            # Run evaluation
            result = client.run_evaluation(
                examples=[example],
                scorers=[FaithfulnessScorer(threshold=0.7)],
                model="gpt-3.5-turbo",
                rules=[rule]
            )
            
            # Verify the API was called
            assert mock_post.called
            
            # Get the evaluation call
            eval_call = mock_post.call_args_list[1]
            
            # Extract the JSON payload - use 'json' parameter instead of 'data'
            if 'json' in eval_call[1]:
                payload = eval_call[1]["json"]
            else:
                payload = {}  # Default empty to avoid errors
            
            # Verify notification config with multiple methods was included
            assert "rules" in payload
            rule_data = payload["rules"][0]
            assert "notification" in rule_data
            assert rule_data["notification"]["enabled"] is True
            assert len(rule_data["notification"]["communication_methods"]) == 4
            assert "slack" in rule_data["notification"]["communication_methods"]
            assert "email" in rule_data["notification"]["communication_methods"]
            assert "broadcast_slack" in rule_data["notification"]["communication_methods"]
            assert "broadcast_email" in rule_data["notification"]["communication_methods"]
            assert len(rule_data["notification"]["email_addresses"]) == 2 