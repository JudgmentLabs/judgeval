"""Unit tests for the notification functionality in the rules system."""

import pytest
from unittest.mock import MagicMock, patch
import uuid
from typing import Dict, List, Any, Optional

from judgeval.rules import (
    Rule, 
    Condition, 
    RulesEngine, 
    AlertStatus, 
    NotificationConfig
)
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import FaithfulnessScorer
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import AnswerRelevancyScorer
from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example


class TestNotificationConfig:
    """Tests for the NotificationConfig class."""
    
    def test_notification_config_creation(self):
        """Test creating a notification config with different parameters."""
        # Default config (minimal)
        config = NotificationConfig()
        assert config.enabled is True
        assert config.communication_methods == []
        assert config.message_template is None
        assert config.email_addresses is None
        assert config.send_at is None
        
        # Full config
        email_addresses = ["test@example.com", "user@example.com"]
        config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Rule {rule_name} was triggered with score {score}",
            email_addresses=email_addresses,
            send_at=1632150000
        )
        
        assert config.enabled is True
        assert config.communication_methods == ["slack", "email"]
        assert config.message_template == "Rule {rule_name} was triggered with score {score}"
        assert config.email_addresses == email_addresses
        assert config.send_at == 1632150000
    
    def test_notification_config_serialization(self):
        """Test that NotificationConfig can be serialized to a dictionary."""
        config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Test message",
            email_addresses=["test@example.com"],
            send_at=1632150000
        )
        
        # Test the model_dump method
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["enabled"] is True
        assert data["communication_methods"] == ["slack", "email"]
        assert data["message_template"] == "Test message"
        assert data["email_addresses"] == ["test@example.com"]
        assert data["send_at"] == 1632150000


class TestRuleWithNotification:
    """Tests for rules with notification configuration."""
    
    def test_rule_with_notification(self):
        """Test creating a rule with notification configuration."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Rule {rule_name} was triggered",
            email_addresses=["test@example.com"]
        )
        
        # Create rule with notification config
        rule = Rule(
            name="Test Rule",
            description="Rule for testing notifications",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification
        )
        
        # Verify the notification config was set correctly
        assert rule.notification is not None
        assert rule.notification.enabled is True
        assert rule.notification.communication_methods == ["slack", "email"]
        assert rule.notification.message_template == "Rule {rule_name} was triggered"
        assert rule.notification.email_addresses == ["test@example.com"]
        
        # Check that the rule can be serialized correctly with its notification config
        data = rule.model_dump()
        assert "notification" in data
        assert data["notification"]["enabled"] is True
        assert data["notification"]["communication_methods"] == ["slack", "email"]
    
    def test_rule_without_notification(self):
        """Test creating a rule without notification configuration."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create rule without notification config
        rule = Rule(
            name="Test Rule Without Notification",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all"
        )
        
        # Verify no notification config is set
        assert rule.notification is None
        
        # Check serialization
        data = rule.model_dump()
        assert "notification" in data
        assert data["notification"] is None


class TestRulesEngineNotification:
    """Tests for notification configuration in RulesEngine."""
    
    def test_configure_notification(self):
        """Test configuring notifications for a rule."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create rule without notification
        rule = Rule(
            name="Test Rule",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all"
        )
        
        # Create engine with the rule
        rule_id = f"rule_{uuid.uuid4()}"
        engine = RulesEngine({rule_id: rule})
        
        # Configure notification for the rule
        engine.configure_notification(
            rule_id=rule_id,
            enabled=True,
            communication_methods=["slack", "email"],
            message_template="Custom notification message",
            email_addresses=["configured@example.com"],
            send_at=None
        )
        
        # Verify notification was configured correctly
        configured_rule = engine.rules[rule_id]
        assert configured_rule.notification is not None
        assert configured_rule.notification.enabled is True
        assert configured_rule.notification.communication_methods == ["slack", "email"]
        assert configured_rule.notification.message_template == "Custom notification message"
        assert configured_rule.notification.email_addresses == ["configured@example.com"]
    
    def test_configure_all_notifications(self):
        """Test configuring notifications for all rules at once."""
        # Use direct scorer instances 
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        
        # Create multiple rules without notification
        rule1 = Rule(
            name="Rule 1",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all"
        )
        
        rule2 = Rule(
            name="Rule 2",
            conditions=[
                Condition(metric=relevancy_scorer)
            ],
            combine_type="all"
        )
        
        # Create engine with the rules
        rule_ids = {
            f"rule1_{uuid.uuid4()}": rule1,
            f"rule2_{uuid.uuid4()}": rule2
        }
        engine = RulesEngine(rule_ids)
        
        # Configure notifications for all rules
        engine.configure_all_notifications(
            enabled=True,
            communication_methods=["email"],
            email_addresses=["global@example.com"]
        )
        
        # Verify all rules have the notification setting
        for rule_id, rule in engine.rules.items():
            assert rule.notification is not None
            assert rule.notification.enabled is True
            assert rule.notification.communication_methods == ["email"]
            assert rule.notification.email_addresses == ["global@example.com"]


class TestNotificationInAlertResults:
    """Tests for inclusion of notification configuration in alert results."""
    
    def test_notification_in_alert_results(self):
        """Test that notification config is included in alert results when rule is triggered."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            message_template="Alert message",
            email_addresses=["alert@example.com"]
        )
        
        # Create rule with notification
        rule = Rule(
            name="Test Rule",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification
        )
        
        # Create engine with the rule
        engine = RulesEngine({"test_rule": rule})
        
        # Create scores that will trigger the rule
        scores = {"faithfulness": 0.8}  # Above threshold, will trigger
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Check that notification config is in the alert result
        assert "test_rule" in results
        assert results["test_rule"].status == AlertStatus.TRIGGERED
        assert hasattr(results["test_rule"], "notification")
        assert results["test_rule"].notification is not None
        assert results["test_rule"].notification.enabled is True
        assert results["test_rule"].notification.communication_methods == ["slack"]
        
        # Test serialization of alert result with notification
        data = results["test_rule"].model_dump()
        assert "notification" in data
        assert data["notification"]["enabled"] is True
        assert data["notification"]["communication_methods"] == ["slack"]
        assert data["notification"]["message_template"] == "Alert message"
        assert data["notification"]["email_addresses"] == ["alert@example.com"]
    
    def test_notification_not_included_when_not_triggered(self):
        """Test that notification config is NOT included in alert results when rule is not triggered."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        
        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            message_template="Alert message",
            email_addresses=["alert@example.com"]
        )
        
        # Create rule with notification
        rule = Rule(
            name="Test Rule",
            conditions=[
                Condition(metric=faithfulness_scorer)
            ],
            combine_type="all",
            notification=notification
        )
        
        # Create engine with the rule
        engine = RulesEngine({"test_rule": rule})
        
        # Create scores that will NOT trigger the rule
        scores = {"faithfulness": 0.5}  # Below threshold, will not trigger
        
        # Evaluate rules
        results = engine.evaluate_rules(scores)
        
        # Check that notification config is not in the alert result
        assert "test_rule" in results
        assert results["test_rule"].status == AlertStatus.NOT_TRIGGERED
        
        # In the not triggered case, we still have the notification, but it won't be used
        # for sending notifications
        assert hasattr(results["test_rule"], "notification")
        assert results["test_rule"].notification is None


@patch('judgeval.judgment_client.run_eval')
class TestNotificationWithJudgmentClient:
    """Tests for notification with JudgmentClient."""
    
    @patch('judgeval.scorers.judgeval_scorers.ScorerWrapper.load_implementation')
    def test_judgment_client_with_rules_and_notification(self, mock_load_implementation, mock_run_eval):
        """Test that JudgmentClient works with rules that have notification configs."""
        # Mock the implementation of ScorerWrapper.load_implementation
        mock_implementation = MagicMock()
        mock_implementation.score_type = "faithfulness"
        mock_load_implementation.return_value = mock_implementation
        
        # Mock the run_eval function
        mock_result = MagicMock()
        mock_result.alert_results = {
            "rule_0": {
                "status": "triggered",
                "rule_name": "Quality Check",
                "conditions_result": [
                    {"metric": "faithfulness", "value": 0.8, "threshold": 0.7, "passed": True, "skipped": False}
                ],
                "notification": {
                    "enabled": True,
                    "communication_methods": ["slack", "email"],
                    "message_template": "Rule triggered",
                    "email_addresses": ["test@example.com"],
                    "send_at": None
                }
            }
        }
        mock_run_eval.return_value = [mock_result]
        
        # Create client with patched _validate_api_key method
        with patch.object(JudgmentClient, '_validate_api_key', return_value=(True, {"detail": {"user_name": "test_user"}})):
            client = JudgmentClient(judgment_api_key="test_key")
        
            # Create example
            example = Example(
                input="Test input",
                actual_output="Test output",
                expected_output="Expected output"
            )
        
            # Create scorers
            scorers = [FaithfulnessScorer(threshold=0.7)]
        
            # Create rules with notification
            notification = NotificationConfig(
                enabled=True,
                communication_methods=["slack", "email"],
                message_template="Rule triggered",
                email_addresses=["test@example.com"]
            )
            
            rules = [
                Rule(
                    name="Quality Check",
                    conditions=[
                        Condition(metric=FaithfulnessScorer(threshold=0.7))
                    ],
                    combine_type="all",
                    notification=notification
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
            
            # Check that rules in call_args have notification configs
            assert len(call_args.rules) == 1
            rule = call_args.rules[0]
            assert rule.notification is not None
            assert rule.notification.enabled is True
            assert rule.notification.communication_methods == ["slack", "email"] 