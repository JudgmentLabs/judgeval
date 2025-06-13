"""
Tests for the Rule Engine's ability to skip missing metrics.
"""

from judgeval.rules import Rule, Condition, RulesEngine
from judgeval.utils.alerts import AlertStatus
from judgeval.scorers import APIJudgmentScorer


# Mock Scorer objects for testing
class MockFaithfulnessScorer(APIJudgmentScorer):
    def __init__(self):
        super().__init__(score_type="faithfulness", threshold=0.7, strict_mode=True)


class MockRelevancyScorer(APIJudgmentScorer):
    def __init__(self):
        super().__init__(score_type="answer_relevancy", threshold=0.8, strict_mode=True)


class MockMetric1Scorer(APIJudgmentScorer):
    def __init__(self):
        super().__init__(
            score_type="contextual_recall", threshold=0.7, strict_mode=True
        )


class MockMetric2Scorer(APIJudgmentScorer):
    def __init__(self):
        super().__init__(
            score_type="contextual_relevancy", threshold=0.7, strict_mode=True
        )


class MockMetric3Scorer(APIJudgmentScorer):
    def __init__(self):
        super().__init__(
            score_type="contextual_precision", threshold=0.7, strict_mode=True
        )


# Create scorer instances
faithfulness_scorer = MockFaithfulnessScorer()
relevancy_scorer = MockRelevancyScorer()
metric1_scorer = MockMetric1Scorer()
metric2_scorer = MockMetric2Scorer()
metric3_scorer = MockMetric3Scorer()


def test_skip_missing_metrics_all():
    """Test that missing metrics are skipped in 'all' combine type rules."""
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

    engine = RulesEngine(rules)

    # Test when one metric is missing
    scores = {"faithfulness": 0.8}  # Missing relevancy
    results = engine.evaluate_rules(scores)

    assert "quality_check" in results
    assert (
        results["quality_check"].status == AlertStatus.NOT_TRIGGERED
    )  # Missing metric causes rule not to trigger

    # Verify the condition results
    condition_results = results["quality_check"].conditions_result
    assert len(condition_results) == 2

    # Find the missing metric condition
    relevancy_condition = next(
        (c for c in condition_results if c["metric"] == "answer_relevancy"), None
    )
    assert relevancy_condition is not None
    assert relevancy_condition["skipped"] is True

    # Find the existing metric condition
    faithfulness_condition = next(
        (c for c in condition_results if c["metric"] == "faithfulness"), None
    )
    assert faithfulness_condition is not None
    assert faithfulness_condition["passed"] is True


def test_skip_missing_metrics_any():
    """Test that missing metrics are skipped in 'any' combine type rules."""
    rules = {
        "quality_check": Rule(
            name="Quality Check",
            conditions=[
                Condition(metric=faithfulness_scorer),
                Condition(metric=relevancy_scorer),
            ],
            combine_type="any",
        )
    }

    engine = RulesEngine(rules)

    # Test when one metric is missing but other passes
    scores = {"faithfulness": 0.8}  # Missing relevancy
    results = engine.evaluate_rules(scores)

    assert "quality_check" in results
    assert (
        results["quality_check"].status == AlertStatus.TRIGGERED
    )  # Should trigger based on faithfulness only

    # Test when one metric is missing and other fails
    scores = {"faithfulness": 0.6}  # Missing relevancy and faithfulness fails
    results = engine.evaluate_rules(scores)

    assert "quality_check" in results
    assert results["quality_check"].status == AlertStatus.NOT_TRIGGERED


def test_rules_all_metrics_missing():
    """Test when all metrics are missing."""
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

    engine = RulesEngine(rules)

    # Test when all metrics are missing
    scores = {}
    results = engine.evaluate_rules(scores)

    assert "quality_check" in results
    assert results["quality_check"].status == AlertStatus.NOT_TRIGGERED


def test_mixed_conditions():
    """Test with a mix of missing, passing, and failing metrics."""
    rules = {
        "mixed_rule": Rule(
            name="Mixed Rule",
            conditions=[
                Condition(metric=metric1_scorer),
                Condition(metric=metric2_scorer),
                Condition(metric=metric3_scorer),
            ],
            combine_type="all",
        )
    }

    engine = RulesEngine(rules)

    # Test with one missing, one passing, one failing
    scores = {"contextual_recall": 0.8, "contextual_precision": 0.6}

    results = engine.evaluate_rules(scores)

    assert "mixed_rule" in results
    assert results["mixed_rule"].status == AlertStatus.NOT_TRIGGERED

    # Check each condition
    condition_results = results["mixed_rule"].conditions_result

    metric1_condition = next(
        (c for c in condition_results if c["metric"] == "contextual_recall"), None
    )
    assert metric1_condition["passed"] is True
    assert metric1_condition["skipped"] is False

    metric2_condition = next(
        (c for c in condition_results if c["metric"] == "contextual_relevancy"), None
    )
    assert metric2_condition["passed"] is None
    assert metric2_condition["skipped"] is True

    metric3_condition = next(
        (c for c in condition_results if c["metric"] == "contextual_precision"), None
    )
    assert metric3_condition["passed"] is False
    assert metric3_condition["skipped"] is False
