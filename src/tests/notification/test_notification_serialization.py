"""Tests for serialization and deserialization of notification configurations."""

import json

from judgeval.rules import Rule, Condition, NotificationConfig
from judgeval.utils.alerts import AlertStatus, AlertResult
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import (
    FaithfulnessScorer,
)
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import (
    AnswerRelevancyScorer,
)


class TestNotificationSerialization:
    """Tests for serialization of notification configurations."""

    def test_notification_config_serialization(self):
        """Test that NotificationConfig can be properly serialized to JSON."""
        # Create a full notification config
        config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email", "broadcast_slack"],
            email_addresses=["user1@example.com", "user2@example.com"],
            send_at=1632150000,
        )

        # Serialize to dict and then to JSON
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict)

        # Deserialize from JSON to dict
        deserialized_dict = json.loads(config_json)

        # Verify the serialization/deserialization maintained data integrity
        assert deserialized_dict["enabled"] is True
        assert len(deserialized_dict["communication_methods"]) == 3
        assert "slack" in deserialized_dict["communication_methods"]
        assert "email" in deserialized_dict["communication_methods"]
        assert "broadcast_slack" in deserialized_dict["communication_methods"]
        assert len(deserialized_dict["email_addresses"]) == 2
        assert "user1@example.com" in deserialized_dict["email_addresses"]
        assert "user2@example.com" in deserialized_dict["email_addresses"]
        assert deserialized_dict["send_at"] == 1632150000

    def test_rule_with_notification_serialization(self):
        """Test that a Rule with NotificationConfig can be properly serialized to JSON."""
        # Use a direct faithfulness scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)

        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["test@example.com"],
            send_at=None,
        )

        # Create rule with notification
        rule = Rule(
            name="Test Rule",
            description="Rule for testing serialization",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification,
        )

        # Serialize to dict and then to JSON
        rule_dict = rule.model_dump()
        rule_json = json.dumps(rule_dict)

        # Deserialize from JSON to dict
        deserialized_dict = json.loads(rule_json)

        # Verify the serialization/deserialization maintained data integrity
        assert deserialized_dict["name"] == "Test Rule"
        assert deserialized_dict["description"] == "Rule for testing serialization"
        assert deserialized_dict["combine_type"] == "all"

        # Verify notification was properly serialized
        assert "notification" in deserialized_dict
        notification_dict = deserialized_dict["notification"]
        assert notification_dict["enabled"] is True
        assert notification_dict["communication_methods"] == ["slack", "email"]
        assert notification_dict["email_addresses"] == ["test@example.com"]
        assert notification_dict["send_at"] is None

    def test_alert_result_with_notification_serialization(self):
        """Test that AlertResult with NotificationConfig can be properly serialized to JSON."""
        # Create notification config
        notification = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            email_addresses=["alert@example.com"],
        )

        # Create an AlertResult with notification
        alert_result = AlertResult(
            status=AlertStatus.TRIGGERED,
            rule_name="Test Rule",
            rule_id="rule_123",
            conditions_result=[
                {
                    "metric": "faithfulness",
                    "value": 0.8,
                    "threshold": 0.7,
                    "passed": True,
                }
            ],
            metadata={"example_id": "example_123", "timestamp": "20240101_120000"},
            notification=notification,
        )

        # Serialize to dict and then to JSON
        result_dict = alert_result.model_dump()
        result_json = json.dumps(result_dict)

        # Deserialize from JSON to dict
        deserialized_dict = json.loads(result_json)

        # Verify the serialization/deserialization maintained data integrity
        assert deserialized_dict["status"] == "triggered"
        assert deserialized_dict["rule_name"] == "Test Rule"
        assert deserialized_dict["rule_id"] == "rule_123"
        assert len(deserialized_dict["conditions_result"]) == 1
        assert deserialized_dict["conditions_result"][0]["metric"] == "faithfulness"
        assert deserialized_dict["conditions_result"][0]["value"] == 0.8

        # Verify notification was properly serialized
        assert "notification" in deserialized_dict
        notification_dict = deserialized_dict["notification"]
        assert notification_dict["enabled"] is True
        assert notification_dict["communication_methods"] == ["slack"]
        assert notification_dict["email_addresses"] == ["alert@example.com"]

    def test_serialization_with_null_notification(self):
        """Test serialization when notification is None."""
        # Use a direct faithfulness scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)

        # Create a rule without notification
        rule = Rule(
            name="Rule Without Notification",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=None,
        )

        # Serialize to dict and then to JSON
        rule_dict = rule.model_dump()
        rule_json = json.dumps(rule_dict)

        # Deserialize from JSON to dict
        deserialized_dict = json.loads(rule_json)

        # Verify the notification is None in the serialized data
        assert "notification" in deserialized_dict
        assert deserialized_dict["notification"] is None

    def test_complex_serialization_with_multiple_rules(self):
        """Test serialization of complex data with multiple rules and notification configs."""
        # Use direct scorer instances
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)

        # Create notification configs
        notification1 = NotificationConfig(
            enabled=True, communication_methods=["slack"], email_addresses=None
        )

        notification2 = NotificationConfig(
            enabled=True,
            communication_methods=["email"],
            email_addresses=["email@example.com"],
        )

        # Create rules with different notification configs
        rule1 = Rule(
            name="Rule 1",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification1,
        )

        rule2 = Rule(
            name="Rule 2",
            conditions=[Condition(metric=relevancy_scorer)],
            combine_type="all",
            notification=notification2,
        )

        # Create a complex data structure with multiple rules
        data = {"rules": [rule1.model_dump(), rule2.model_dump()]}

        # Serialize to JSON
        data_json = json.dumps(data)

        # Deserialize from JSON
        deserialized_data = json.loads(data_json)

        # Verify the serialization/deserialization maintained data integrity
        assert len(deserialized_data["rules"]) == 2

        # Verify first rule
        rule1_dict = deserialized_data["rules"][0]
        assert rule1_dict["name"] == "Rule 1"
        assert rule1_dict["notification"]["communication_methods"] == ["slack"]
        assert rule1_dict["notification"]["email_addresses"] is None

        # Verify second rule
        rule2_dict = deserialized_data["rules"][1]
        assert rule2_dict["name"] == "Rule 2"
        assert rule2_dict["notification"]["communication_methods"] == ["email"]
        assert rule2_dict["notification"]["email_addresses"] == ["email@example.com"]
