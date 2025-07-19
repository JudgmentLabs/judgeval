"""
Rules system for Judgeval that enables alerts based on metric thresholds.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

from judgeval.scorers import APIScorerConfig, BaseScorer
from judgeval.utils.alerts import AlertStatus, AlertResult


class Condition(BaseModel):
    """
    A single metric condition.

    Example:
        {
            "metric": FaithfulnessScorer(threshold=0.7)  # Must be a scorer object: APIScorerConfig, BaseScorer
        }

    The Condition class uses the scorer's threshold and success function internally.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metric: Union[APIScorerConfig, BaseScorer]

    @property
    def metric_name(self) -> str:
        """Get the name of the metric for lookups in scores dictionary."""
        if hasattr(self.metric, "score_type"):
            # Handle APIScorerConfig and BaseScorer which have score_type
            return self.metric.score_type
        elif hasattr(self.metric, "__name__"):
            # Handle cases where metric has a __name__ attribute
            return self.metric.__name__
        # Fallback to string representation
        return str(self.metric)

    @property
    def threshold(self) -> float:
        """Get the threshold from the metric."""
        return self.metric.threshold if hasattr(self.metric, "threshold") else 0.5

    def evaluate(self, value: float) -> bool:
        """
        Evaluate the condition against a value.
        Returns True if the condition passes, False otherwise.
        Uses the scorer's success check function if available.
        """
        # Store the value in the scorer
        if hasattr(self.metric, "score"):
            self.metric.score = value

        # Use the scorer's success check function if available
        if hasattr(self.metric, "success_check"):
            return self.metric.success_check()
        elif hasattr(self.metric, "success_check"):
            return self.metric.success_check()
        else:
            # Fallback to default comparison (greater than or equal)
            return value >= self.threshold if self.threshold is not None else False


class PagerDutyConfig(BaseModel):
    """
    Configuration for PagerDuty notifications.

    Attributes:
        routing_key: PagerDuty integration routing key
        severity: Severity level (critical, error, warning, info)
        source: Source of the alert (defaults to "judgeval")
        component: Optional component that triggered the alert
        group: Optional logical grouping for the alert
        class_type: Optional class/type of alert event
    """

    routing_key: str
    severity: str = "error"  # critical, error, warning, info
    source: str = "judgeval"
    component: Optional[str] = None
    group: Optional[str] = None
    class_type: Optional[str] = None

    def model_dump(self, **kwargs):
        """Convert the PagerDutyConfig to a dictionary for JSON serialization."""
        return {
            "routing_key": self.routing_key,
            "severity": self.severity,
            "source": self.source,
            "component": self.component,
            "group": self.group,
            "class_type": self.class_type,
        }


class NotificationConfig(BaseModel):
    """
    Configuration for notifications when a rule is triggered.

    Example:
        {
            "enabled": true,
            "communication_methods": ["email", "broadcast_slack", "broadcast_email", "pagerduty"],
            "email_addresses": ["user1@example.com", "user2@example.com"],
            "pagerduty_config": {
                "routing_key": "R0ABCD1234567890123456789",
                "severity": "error"
            },
            "send_at": 1632150000  # Unix timestamp (specific date/time)
        }

    Communication Methods:
        - "email": Send emails to specified email addresses
        - "broadcast_slack": Send broadcast notifications to all configured Slack channels
        - "broadcast_email": Send broadcast emails to all organization emails
        - "pagerduty": Send alerts to PagerDuty using the configured routing key
    """

    enabled: bool = True
    communication_methods: List[str] = []
    email_addresses: Optional[List[str]] = None
    pagerduty_config: Optional[PagerDutyConfig] = None
    send_at: Optional[int] = None  # Unix timestamp for scheduled notifications

    def model_dump(self, **kwargs):
        """Convert the NotificationConfig to a dictionary for JSON serialization."""
        return {
            "enabled": self.enabled,
            "communication_methods": self.communication_methods,
            "email_addresses": self.email_addresses,
            "pagerduty_config": self.pagerduty_config.model_dump()
            if self.pagerduty_config
            else None,
            "send_at": self.send_at,
        }


class Rule(BaseModel):
    """
    Configuration for a single rule.

    Example:
        {
            "rule_id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "Quality Check",
            "description": "Check if quality metrics meet thresholds",
            "conditions": [
                {"metric": FaithfulnessScorer(threshold=0.7)},
                {"metric": AnswerRelevancyScorer(threshold=0.8)}
            ],
            "combine_type": "all",  # "all" or "any"
            "notification": {
                "enabled": true,
                "communication_methods": ["slack", "email"],
                "email_addresses": ["user1@example.com", "user2@example.com"]
            }
        }
    """

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )  # Random UUID string as default value
    name: str
    description: Optional[str] = None
    conditions: List[Condition]
    combine_type: str = Field(..., pattern="^(all|any)$")  # all = AND, any = OR
    notification: Optional[NotificationConfig] = None  # Configuration for notifications

    def model_dump(self, **kwargs):
        """
        Custom serialization that properly handles condition serialization.
        """
        data = super().model_dump(**kwargs)

        # Special handling for conditions with complex metric objects
        if "conditions" in data:
            for i, condition in enumerate(data["conditions"]):
                if "metric" in condition:
                    # Get the actual metric object
                    metric_obj = self.conditions[i].metric

                    # Create standardized metric representation needed by server API
                    metric_data = {"score_type": "", "threshold": 0.0, "name": ""}

                    # First try to use object's own serialization methods
                    if hasattr(metric_obj, "to_dict"):
                        orig_data = metric_obj.to_dict()
                        # Copy any existing fields
                        for key, value in orig_data.items():
                            metric_data[key] = value
                    elif hasattr(metric_obj, "model_dump"):
                        orig_data = metric_obj.model_dump()
                        # Copy any existing fields
                        for key, value in orig_data.items():
                            metric_data[key] = value

                    # If we already have data from original serialization methods but missing required fields
                    if "name" in metric_data and "score_type" not in metric_data:
                        metric_data["score_type"] = metric_data["name"]

                    # Ensure required fields have values by checking various sources
                    if not metric_data["score_type"]:
                        # Try to get score_type from different possible attributes
                        if hasattr(metric_obj, "score_type"):
                            metric_data["score_type"] = metric_obj.score_type
                        elif hasattr(metric_obj, "name"):
                            metric_data["score_type"] = metric_obj.name
                        else:
                            # Last resort: use string representation
                            metric_data["score_type"] = str(metric_obj)

                    # Make sure threshold is set
                    if (
                        not metric_data.get("threshold")
                        and metric_data.get("threshold") != 0.0
                    ):
                        if hasattr(metric_obj, "threshold"):
                            metric_data["threshold"] = metric_obj.threshold
                        else:
                            # Use condition threshold if metric doesn't have one
                            metric_data["threshold"] = self.conditions[i].threshold

                    # Make sure name is set
                    if not metric_data.get("name"):
                        if hasattr(metric_obj, "__name__"):
                            metric_data["name"] = metric_obj.__name__
                        elif hasattr(metric_obj, "name"):
                            metric_data["name"] = metric_obj.name
                        else:
                            # Fallback to score_type if available
                            metric_data["name"] = metric_data.get(
                                "score_type", str(metric_obj)
                            )

                    # Update the condition with our properly serialized metric
                    condition["metric"] = metric_data

        return data


class RulesEngine:
    """
    Engine for creating and evaluating rules against metrics.

    Example:
        ```python
        # Define rules
        rules = {
            "1": Rule(
                name="Quality Check",
                description="Check if quality metrics meet thresholds",
                conditions=[
                    Condition(metric=FaithfulnessScorer(threshold=0.7)),
                    Condition(metric=AnswerRelevancyScorer(threshold=0.8))
                ],
                combine_type="all"
            )
        }

        # Create rules engine
        engine = RulesEngine(rules)

        # Configure notifications
        engine.configure_notification(
            rule_id="1",
            enabled=True,
            communication_methods=["slack", "email"],
            email_addresses=["user@example.com"]
        )

        # Evaluate rules
        scores = {"faithfulness": 0.65, "relevancy": 0.85}
        results = engine.evaluate_rules(scores, {"example_id": "example_123"})
        ```
    """

    def __init__(self, rules: Dict[str, Rule]):
        """
        Initialize the rules engine.

        Args:
            rules: Dictionary mapping rule IDs to Rule objects
        """
        self.rules = rules

    def configure_notification(
        self,
        rule_id: str,
        enabled: bool = True,
        communication_methods: List[str] | None = None,
        email_addresses: List[str] | None = None,
        send_at: Optional[int] = None,
    ) -> None:
        """
        Configure notification settings for a specific rule.

        Args:
            rule_id: ID of the rule to configure notifications for
            enabled: Whether notifications are enabled for this rule
            communication_methods: List of notification methods (e.g., ["slack", "email"])
            email_addresses: List of email addresses to send notifications to
            send_at: Optional Unix timestamp for when to send the notification
        """
        if rule_id not in self.rules:
            raise ValueError(f"Rule ID '{rule_id}' not found")

        rule = self.rules[rule_id]

        # Create notification configuration if it doesn't exist
        if rule.notification is None:
            rule.notification = NotificationConfig()

        # Set notification parameters
        rule.notification.enabled = enabled

        if communication_methods is not None:
            rule.notification.communication_methods = communication_methods

        if email_addresses is not None:
            rule.notification.email_addresses = email_addresses

        if send_at is not None:
            rule.notification.send_at = send_at

    def configure_all_notifications(
        self,
        enabled: bool = True,
        communication_methods: List[str] | None = None,
        email_addresses: List[str] | None = None,
        send_at: Optional[int] = None,
    ) -> None:
        """
        Configure notification settings for all rules.

        Args:
            enabled: Whether notifications are enabled
            communication_methods: List of notification methods (e.g., ["slack", "email"])
            email_addresses: List of email addresses to send notifications to
            send_at: Optional Unix timestamp for when to send the notification
        """
        for rule_id, rule in self.rules.items():
            self.configure_notification(
                rule_id=rule_id,
                enabled=enabled,
                communication_methods=communication_methods,
                email_addresses=email_addresses,
                send_at=send_at,
            )

    def evaluate_rules(
        self,
        scores: Dict[str, float],
        example_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, AlertResult]:
        """
        Evaluate all rules against a set of scores.
        Returns mapping of rule IDs to their alert results.

        Args:
            scores: Dictionary of metric names to their score values
            example_metadata: Optional dictionary containing example metadata (example_id, timestamp)
        """
        results = {}

        for rule_id, rule in self.rules.items():
            # Evaluate each condition
            condition_results = []
            passed_conditions = []

            for condition in rule.conditions:
                # Get the metric name for lookup
                metric_name = condition.metric_name
                value = scores.get(metric_name)

                if value is None:
                    # Skip this condition instead of evaluating it as false
                    condition_results.append(
                        {
                            "metric": metric_name,
                            "value": None,
                            "threshold": condition.threshold,
                            "passed": None,  # Using None to indicate the condition was skipped
                            "skipped": True,  # Add a flag to indicate this condition was skipped
                        }
                    )
                    continue  # Skip adding to passed_conditions
                else:
                    passed = condition.evaluate(value)
                    condition_results.append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": condition.threshold,
                            "passed": passed,
                            "skipped": False,  # Indicate this condition was evaluated
                        }
                    )
                    passed_conditions.append(passed)

            # Determine if alert should trigger - only consider conditions that weren't skipped
            if not passed_conditions:
                # If all conditions were skipped, the rule doesn't trigger
                triggered = False
            else:
                if rule.combine_type == "all":
                    # For "all" combine_type:
                    # - All evaluated conditions must pass
                    # - All conditions must have been evaluated (none skipped)
                    all_conditions_passed = all(passed_conditions)
                    all_conditions_evaluated = len(passed_conditions) == len(
                        rule.conditions
                    )
                    triggered = all_conditions_passed and all_conditions_evaluated
                else:
                    # For "any" combine_type, at least one condition must pass
                    triggered = any(passed_conditions)

            # Create alert result with example metadata
            notification_config = None
            if triggered and rule.notification:
                # If rule has a notification config and the alert is triggered, include it in the result
                notification_config = rule.notification

            # Set the alert status based on whether the rule was triggered using proper enum values
            status = AlertStatus.TRIGGERED if triggered else AlertStatus.NOT_TRIGGERED

            # Create the alert result
            alert_result = AlertResult(
                status=status,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                conditions_result=condition_results,
                notification=notification_config,
                metadata=example_metadata or {},
                combine_type=rule.combine_type,
                project_id=example_metadata.get("project_id")
                if example_metadata
                else None,
                trace_span_id=example_metadata.get("trace_span_id")
                if example_metadata
                else None,
            )

            results[rule_id] = alert_result

        return results

    async def evaluate_rules_parallel(
        self,
        example_scores: Dict[str, Dict[str, float]],
        example_metadata: Dict[str, Dict[str, Any]],
        max_concurrent: int = 100,
    ) -> Dict[str, Dict[str, AlertResult]]:
        """
        Evaluate all rules against multiple examples in parallel.

        Args:
            example_scores: Dictionary mapping example_ids to their score dictionaries
            example_metadata: Dictionary mapping example_ids to their metadata
            max_concurrent: Maximum number of concurrent evaluations

        Returns:
            Dictionary mapping example_ids to dictionaries of rule_ids and their alert results
        """
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        tasks = []

        # Create a task for each example
        for example_id, scores in example_scores.items():
            metadata = example_metadata.get(example_id, {})
            task = self._evaluate_with_semaphore(
                semaphore=semaphore,
                example_id=example_id,
                scores=scores,
                metadata=metadata,
            )
            tasks.append(task)

        # Run all tasks and collect results
        example_results = await asyncio.gather(*tasks)

        # Organize results by example_id
        for example_id, result in example_results:
            results[example_id] = result

        return results

    async def _evaluate_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        example_id: str,
        scores: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> Tuple[str, Dict[str, AlertResult]]:
        """
        Helper method to evaluate rules for an example with semaphore control.

        Args:
            semaphore: Semaphore to control concurrency
            example_id: ID of the example being evaluated
            scores: Dictionary of scores for this example
            metadata: Metadata for this example

        Returns:
            Tuple of (example_id, rule_results)
        """
        async with semaphore:
            # Run the evaluation in a thread pool to avoid blocking the event loop
            # for CPU-bound operations
            with ThreadPoolExecutor() as executor:
                rule_results = await asyncio.get_event_loop().run_in_executor(
                    executor, self.evaluate_rules, scores, metadata
                )

                return (example_id, rule_results)
