"""Integration tests for the notification functionality without using Tracer."""

import pytest
import asyncio

from judgeval.rules import Rule, Condition, NotificationConfig, RulesEngine
from judgeval.utils.alerts import AlertStatus
from judgeval.scorers import AnswerRelevancyScorer, FaithfulnessScorer


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


class TestDirectNotificationIntegration:
    """Integration tests for notifications using the rules engine directly."""

    def test_rules_engine_with_notification(self):
        """Test rules engine with notification configurations."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        # Manually set score_type if needed for testing
        if not hasattr(faithfulness_scorer, "score_type"):
            faithfulness_scorer.score_type = "faithfulness"

        # Create notification config
        notification_config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["test@example.com"],
            send_at=None,
        )

        # Create rule with notification config
        rule = Rule(
            name="Faithfulness Check",
            description="Check if faithfulness meets threshold",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification_config,
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
        assert results["test_rule"].notification.communication_methods == [
            "slack",
            "email",
        ]
        assert results["test_rule"].notification.email_addresses == ["test@example.com"]

    def test_rules_engine_parallel_evaluation_with_notifications(self):
        """Test rules engine's parallel evaluation with notification configurations."""
        # Use a direct scorer instance
        faithfulness_scorer = FaithfulnessScorer(threshold=0.7)
        # Manually set score_type if needed for testing
        if not hasattr(faithfulness_scorer, "score_type"):
            faithfulness_scorer.score_type = "faithfulness"

        # Create notification config
        notification_config = NotificationConfig(
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["test@example.com"],
            send_at=None,
        )

        # Create rule with notification config
        rule = Rule(
            name="Faithfulness Check",
            description="Check if faithfulness meets threshold",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification_config,
        )

        # Create engine with the rule
        engine = RulesEngine({"test_rule": rule})

        # Create example scores and metadata for multiple examples
        example_scores = {
            "example1": {"faithfulness": 0.8},  # Will trigger
            "example2": {"faithfulness": 0.6},  # Won't trigger
        }

        example_metadata = {
            "example1": {"id": "example1", "timestamp": "20240101_120000"},
            "example2": {"id": "example2", "timestamp": "20240101_120100"},
        }

        # Run parallel evaluation
        async def run_parallel_evaluation():
            results = await engine.evaluate_rules_parallel(
                example_scores=example_scores, example_metadata=example_metadata
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
        if not hasattr(faithfulness_scorer, "score_type"):
            faithfulness_scorer.score_type = "faithfulness"

        relevancy_scorer = AnswerRelevancyScorer(threshold=0.8)
        # Manually set score_type if needed for testing
        if not hasattr(relevancy_scorer, "score_type"):
            relevancy_scorer.score_type = "answer_relevancy"

        # Create notification configs
        notification1 = NotificationConfig(
            enabled=True,
            communication_methods=["slack"],
            email_addresses=None,
            send_at=None,
        )

        notification2 = NotificationConfig(
            enabled=True,
            communication_methods=["email"],
            email_addresses=["email@example.com"],
            send_at=None,
        )

        # Create rules with different notification configs
        rule1 = Rule(
            name="Faithfulness Rule",
            conditions=[Condition(metric=faithfulness_scorer)],
            combine_type="all",
            notification=notification1,
        )

        rule2 = Rule(
            name="Relevancy Rule",
            conditions=[Condition(metric=relevancy_scorer)],
            combine_type="all",
            notification=notification2,
        )

        # Create engine with both rules
        engine = RulesEngine({"rule1": rule1, "rule2": rule2})

        # Create scores that will trigger both rules
        scores = {"faithfulness": 0.9, "answer_relevancy": 0.9}

        # Evaluate rules
        results = engine.evaluate_rules(scores)

        # Verify rule1 has the correct notification config
        assert "rule1" in results
        assert results["rule1"].status == AlertStatus.TRIGGERED
        assert results["rule1"].notification is not None
        assert results["rule1"].notification.communication_methods == ["slack"]

        # Verify rule2 has the correct notification config
        assert "rule2" in results
        assert results["rule2"].status == AlertStatus.TRIGGERED
        assert results["rule2"].notification is not None
        assert results["rule2"].notification.communication_methods == ["email"]
        assert results["rule2"].notification.email_addresses == ["email@example.com"]
