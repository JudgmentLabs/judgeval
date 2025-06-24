"""Tests to verify rules work correctly with API scorers."""

from judgeval.rules import Rule, Condition, RulesEngine
from judgeval.utils.alerts import AlertStatus
from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import (
    FaithfulnessScorer,
)
from judgeval.scorers.judgeval_scorers.api_scorers.answer_relevancy import (
    AnswerRelevancyScorer,
)


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
                    Condition(metric=relevancy_scorer),
                ],
                combine_type="all",
            )
        }

        # Create RulesEngine
        engine = RulesEngine(rules)

        # Create scores dictionary that matches the metric names
        # Important: the key should match what's returned by the metric_name property
        scores = {
            "faithfulness": 0.8,  # This should match faithfulness_scorer.score_type
            "answer_relevancy": 0.9,  # This should match relevancy_scorer.score_type
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
