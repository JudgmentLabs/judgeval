"""Unit tests for the notification functionality in the rules system."""

import pytest
import uuid

from judgeval.rules import (
    Rule,
    Condition,
    RulesEngine,
    NotificationConfig,
    PagerDutyConfig,
)
from judgeval.utils.alerts import AlertStatus
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import (
    FaithfulnessScorer,
)
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import (
    AnswerRelevancyScorer,
)


@pytest.fixture
def mock_validate_api_key(monkeypatch):
    """Mock the validate_api_key function and organization ID."""

    def _mock_validate_api_key(judgment_api_key):
        return True, "Valid API key"

    monkeypatch.setattr(
        "judgeval.common.utils.validate_api_key", _mock_validate_api_key
    )
    monkeypatch.setenv("JUDGMENT_ORG_ID", "test_org_id")
    return _mock_validate_api_key


class TestPagerDutyConfig:
    """Tests for the PagerDutyConfig class."""

    def test_pagerduty_config_creation(self):
        """Test creating a PagerDuty config with different parameters."""
        # Minimal config (only routing key required)
        config = PagerDutyConfig(routing_key="R0ABCD1234567890123456789")
        assert config.routing_key == "R0ABCD1234567890123456789"
        assert config.severity == "error"  # default
        assert config.source == "judgeval"  # default
        assert config.component is None
        assert config.group is None
        assert config.class_type is None

        # Full config
        config = PagerDutyConfig(
            routing_key="R0ABCD1234567890123456789",
            severity="critical",
            source="ml-pipeline",
            component="faithfulness-checker",
            group="production",
            class_type="ml-alert",
        )

        assert config.routing_key == "R0ABCD1234567890123456789"
        assert config.severity == "critical"
        assert config.source == "ml-pipeline"
        assert config.component == "faithfulness-checker"
        assert config.group == "production"
        assert config.class_type == "ml-alert"

    def test_pagerduty_config_serialization(self):
        """Test that PagerDutyConfig can be serialized to a dictionary."""
        config = PagerDutyConfig(
            routing_key="R0ABCD1234567890123456789",
            severity="critical",
            component="test-component",
        )

        # Test the model_dump method
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["routing_key"] == "R0ABCD1234567890123456789"
        assert data["severity"] == "critical"
        assert data["source"] == "judgeval"
        assert data["component"] == "test-component"
        assert data["group"] is None
        assert data["class_type"] is None


class TestNotificationConfig:
    """Tests for the NotificationConfig class."""

    def test_notification_config_creation(self):
        """Test creating a notification config with different parameters."""
        # Default config (minimal)
        config = NotificationConfig()
        assert config.enabled is True
        assert config.communication_methods == []
        assert config.email_addresses is None
        assert config.send_at is None

        # Full config
        email_addresses = ["test@example.com", "user@example.com"]
        config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=email_addresses,
            send_at=1632150000,
        )

        assert config.enabled is True
        assert config.communication_methods == ["slack", "email"]
        assert config.email_addresses == email_addresses
        assert config.send_at == 1632150000

    def test_notification_config_with_pagerduty(self):
        """Test creating a notification config with PagerDuty configuration."""
        # Create PagerDuty config
        pagerduty_config = PagerDutyConfig(
            routing_key="R0ABCD1234567890123456789", severity="critical"
        )

        # Create notification config with PagerDuty
        config = NotificationConfig(
            enabled=True,
            communication_methods=["pagerduty", "email"],
            email_addresses=["test@example.com"],
            pagerduty_config=pagerduty_config,
        )

        assert config.enabled is True
        assert config.communication_methods == ["pagerduty", "email"]
        assert config.email_addresses == ["test@example.com"]
        assert config.pagerduty_config is not None
        assert config.pagerduty_config.routing_key == "R0ABCD1234567890123456789"
        assert config.pagerduty_config.severity == "critical"

        # Test serialization includes PagerDuty config
        data = config.model_dump()
        assert data["pagerduty_config"] is not None
        assert data["pagerduty_config"]["routing_key"] == "R0ABCD1234567890123456789"
        assert data["pagerduty_config"]["severity"] == "critical"

    def test_notification_config_serialization(self):
        """Test that NotificationConfig can be serialized to a dictionary."""
        config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["test@example.com"],
            send_at=1632150000,
        )

        # Test the model_dump method
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["enabled"] is True
        assert data["communication_methods"] == ["slack", "email"]
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
            email_addresses=["test@example.com"],
        )

        # Create rule with notification config
        rule = Rule(
            name="Test Rule",
            description="Rule for testing notifications",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification,
        )

        # Verify the notification config was set correctly
        assert rule.notification is not None
        assert rule.notification.enabled is True
        assert rule.notification.communication_methods == ["slack", "email"]
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
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
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
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
        )

        # Create engine with the rule
        rule_id = f"rule_{uuid.uuid4()}"
        engine = RulesEngine({rule_id: rule})

        # Configure notification for the rule
        engine.configure_notification(
            rule_id=rule_id,
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["configured@example.com"],
            send_at=None,
        )

        # Verify notification was configured correctly
        configured_rule = engine.rules[rule_id]
        assert configured_rule.notification is not None
        assert configured_rule.notification.enabled is True
        assert configured_rule.notification.communication_methods == ["slack", "email"]
        assert configured_rule.notification.email_addresses == [
            "configured@example.com"
        ]

    def test_configure_all_notifications(self):
        """Test configuring notifications for all rules at once."""
        # Use direct scorer instances
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)

        # Create multiple rules without notification
        rule1 = Rule(
            name="Rule 1",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
        )

        rule2 = Rule(
            name="Rule 2",
            conditions=[Condition(metric=relevancy_scorer)],
            combine_type="all",
        )

        # Create engine with the rules
        rule_ids = {f"rule1_{uuid.uuid4()}": rule1, f"rule2_{uuid.uuid4()}": rule2}
        engine = RulesEngine(rule_ids)

        # Configure notifications for all rules
        engine.configure_all_notifications(
            enabled=True,
            communication_methods=["email"],
            email_addresses=["global@example.com"],
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
            email_addresses=["alert@example.com"],
        )

        # Create rule with notification
        rule = Rule(
            name="Test Rule",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification,
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
        assert data["notification"]["email_addresses"] == ["alert@example.com"]

    def test_notification_not_included_when_not_triggered(self):
        """Test that notification config is NOT included in alert results when rule is not triggered."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)

        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            email_addresses=["alert@example.com"],
        )

        # Create rule with notification
        rule = Rule(
            name="Test Rule",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification,
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
