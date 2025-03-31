"""
Rules system for Judgeval that enables alerts based on metric thresholds.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import uuid

from judgeval.scorers import APIJudgmentScorer, JudgevalScorer, ScorerWrapper

class AlertStatus(str, Enum):
    """Status of an alert evaluation."""
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"

class Operator(str, Enum):
    """Comparison operators for conditions."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    EQ = "=="
    NEQ = "!="

class Condition(BaseModel):
    """
    A single metric condition.
    
    Example:
        {
            "metric": FaithfulnessScorer(threshold=0.7)  # Must be a scorer object: APIJudgmentScorer, JudgevalScorer, or ScorerWrapper
            "operator": ">=",
            "threshold": 0.7
        }
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metric: Union[APIJudgmentScorer, JudgevalScorer, ScorerWrapper]  
    operator: Operator
    threshold: float

    @property
    def metric_name(self) -> str:
        """Get the name of the metric for lookups in scores dictionary."""
        if isinstance(self.metric, ScorerWrapper):
            # Handle ScorerWrapper case specifically
            return self.metric.scorer.score_type if hasattr(self.metric.scorer, 'score_type') else str(self.metric.scorer)
        elif hasattr(self.metric, 'score_type'):
            # Handle APIJudgmentScorer and JudgevalScorer which have score_type
            return self.metric.score_type
        elif hasattr(self.metric, '__name__'):
            # Handle cases where metric has a __name__ attribute
            return self.metric.__name__
        # Fallback to string representation
        return str(self.metric)

    def evaluate(self, value: float) -> bool:
        """
        Evaluate the condition against a value.
        Returns True if the condition passes, False otherwise.
        """
        if self.operator == Operator.GT:
            return value > self.threshold
        elif self.operator == Operator.GTE:
            return value >= self.threshold
        elif self.operator == Operator.LT:
            return value < self.threshold
        elif self.operator == Operator.LTE:
            return value <= self.threshold
        elif self.operator == Operator.EQ:
            return value == self.threshold
        elif self.operator == Operator.NEQ:
            return value != self.threshold
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")

class NotificationConfig(BaseModel):
    """
    Configuration for notifications when a rule is triggered.
    
    Example:
        {
            "enabled": true,
            "communication_methods": ["slack", "email"],
            "message_template": "Rule '{rule_name}' was triggered with score {score}",
            "email_addresses": ["user1@example.com", "user2@example.com"],
            "send_at": 1632150000  # Unix timestamp (specific date/time)
        }
    """
    enabled: bool = True
    communication_methods: List[str] = []
    message_template: Optional[str] = None
    email_addresses: Optional[List[str]] = None
    send_at: Optional[int] = None  # Unix timestamp for scheduled notifications

class Rule(BaseModel):
    """
    Configuration for a single rule.
    
    Example:
        {
            "rule_id": "123e4567-e89b-12d3-a456-426614174000",
            "name": "Quality Check",
            "description": "Check if quality metrics meet thresholds",
            "conditions": [
                {"metric": FaithfulnessScorer(threshold=0.7), "operator": ">=", "threshold": 0.7},
                {"metric": AnswerRelevancyScorer(threshold=0.8), "operator": ">=", "threshold": 0.8}
            ],
            "combine_type": "all",  # "all" or "any"
            "notification": {
                "enabled": true,
                "communication_methods": ["slack", "email"],
                "message_template": "Quality check failed: {condition_results}",
                "email_addresses": ["user1@example.com", "user2@example.com"],
                "send_at": 1632150000  # Unix timestamp (specific date/time)
            }
        }
    """
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Random UUID string as default value
    name: str
    description: Optional[str] = None
    conditions: List[Condition]
    combine_type: str = Field(..., pattern="^(all|any)$")  # all = AND, any = OR
    notification: Optional[NotificationConfig] = None  # Configuration for notifications

    @field_validator('conditions')
    def validate_conditions_not_empty(cls, v):
        if not v:
            raise ValueError("Conditions list cannot be empty")
        return v

    @field_validator('combine_type')
    def validate_combine_type(cls, v):
        if v not in ["all", "any"]:
            raise ValueError(f"combine_type must be 'all' or 'any', got: {v}")
        return v


class AlertResult(BaseModel):
    """
    Result of evaluating a rule.
    
    Example:
        {
            "status": "triggered",
            "rule_name": "Quality Check",
            "conditions_result": [
                {"metric": "faithfulness", "value": 0.6, "threshold": 0.7, "passed": False},
                {"metric": "relevancy", "value": 0.9, "threshold": 0.8, "passed": True}
            ],
            "rule_id": "123e4567-e89b-12d3-a456-426614174000",
            "metadata": {
                "example_id": "example_123",
                "timestamp": "20240321_123456"
            },
            "notification": {
                "enabled": true,
                "communication_methods": ["slack", "email"],
                "message_template": "Rule {rule_name} was triggered with score {score}",
                "email_addresses": ["user1@example.com", "user2@example.com"]
            }
        }
    """
    status: AlertStatus
    rule_id: Optional[str] = None  # The unique identifier of the rule
    rule_name: str
    conditions_result: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    notification: Optional[NotificationConfig] = None  # Configuration for notifications
    
    @property
    def example_id(self) -> Optional[str]:
        """Get example_id from metadata for backward compatibility"""
        return self.metadata.get("example_id")
        
    @property
    def timestamp(self) -> Optional[str]:
        """Get timestamp from metadata for backward compatibility"""
        return self.metadata.get("timestamp")

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
                    Condition(metric=FaithfulnessScorer(threshold=0.7), operator=">=", threshold=0.7),
                    Condition(metric=AnswerRelevancyScorer(threshold=0.8), operator=">=", threshold=0.8)
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
            message_template="Quality check failed for example {example_id}",
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

    def configure_notification(self, rule_id: str, enabled: bool = True, 
                              communication_methods: List[str] = None, 
                              message_template: str = None,
                              email_addresses: List[str] = None,
                              send_at: Optional[int] = None) -> None:
        """
        Configure notification settings for a specific rule.
        
        Args:
            rule_id: ID of the rule to configure notifications for
            enabled: Whether notifications are enabled for this rule
            communication_methods: List of notification methods (e.g., ["slack", "email"])
            message_template: Template string for the notification message
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
            
        if message_template is not None:
            rule.notification.message_template = message_template
            
        if email_addresses is not None:
            rule.notification.email_addresses = email_addresses
            
        if send_at is not None:
            rule.notification.send_at = send_at
    
    def configure_all_notifications(self, enabled: bool = True, 
                                   communication_methods: List[str] = None,
                                   email_addresses: List[str] = None,
                                   send_at: Optional[int] = None) -> None:
        """
        Configure notification settings for all rules.
        
        Args:
            enabled: Whether notifications are enabled
            communication_methods: List of notification methods (e.g., ["slack", "email"])
            email_addresses: List of email addresses to send notifications to
            send_at: Optional Unix timestamp for when to send the notification
        """
        for rule_id, rule in self.rules.items():
            # Create a custom message template based on the rule name
            message_template = f"Rule '{rule.name}' was triggered"
            
            self.configure_notification(
                rule_id=rule_id,
                enabled=enabled,
                communication_methods=communication_methods,
                message_template=message_template,
                email_addresses=email_addresses,
                send_at=send_at
            )
    
    def evaluate_rules(self, scores: Dict[str, float], example_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, AlertResult]:
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
                    condition_results.append({
                        "metric": metric_name,
                        "value": None,
                        "threshold": condition.threshold,
                        "operator": condition.operator,
                        "passed": None,  # Using None to indicate the condition was skipped
                        "skipped": True  # Add a flag to indicate this condition was skipped
                    })
                    continue  # Skip adding to passed_conditions
                else:
                    passed = condition.evaluate(value)
                    condition_results.append({
                        "metric": metric_name,
                        "value": value,
                        "threshold": condition.threshold,
                        "operator": condition.operator,
                        "passed": passed,
                        "skipped": False  # Indicate this condition was evaluated
                    })
                    passed_conditions.append(passed)
            
            # Determine if alert should trigger - only consider conditions that weren't skipped
            if not passed_conditions:
                # If all conditions were skipped, the rule doesn't trigger
                triggered = False
            else:
                triggered = all(passed_conditions) if rule.combine_type == "all" else any(passed_conditions)
            
            # Create alert result with example metadata
            notification_config = None
            if triggered and rule.notification:
                # If rule has a notification config and the alert is triggered, include it in the result
                notification_config = rule.notification
            
            # Set the alert status based on whether the rule was triggered
            status = AlertStatus.TRIGGERED if triggered else AlertStatus.NOT_TRIGGERED
            
            # Create the alert result
            alert_result = AlertResult(
                status=status,
                rule_id=rule.rule_id,
                rule_name=rule.name,
                conditions_result=condition_results,
                notification=notification_config,
                metadata=example_metadata or {}
            )
            
            results[rule_id] = alert_result
            
        return results
    
    async def evaluate_rules_parallel(self, 
                               example_scores: Dict[str, Dict[str, float]], 
                               example_metadata: Dict[str, Dict[str, Any]],
                               max_concurrent: int = 100) -> Dict[str, Dict[str, AlertResult]]:
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
                metadata=metadata
            )
            tasks.append(task)
        
        # Run all tasks and collect results
        example_results = await asyncio.gather(*tasks)
        
        # Organize results by example_id
        for example_id, result in example_results:
            results[example_id] = result
            
        return results
    
    async def _evaluate_with_semaphore(self, 
                                semaphore: asyncio.Semaphore, 
                                example_id: str, 
                                scores: Dict[str, float], 
                                metadata: Dict[str, Any]) -> Tuple[str, Dict[str, AlertResult]]:
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
                start_time = time.perf_counter()
                rule_results = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    self.evaluate_rules,
                    scores,
                    metadata
                )
                end_time = time.perf_counter()
                
                # Could log performance metrics here if needed
                # debug(f"Rule evaluation for example {example_id} took {end_time - start_time:.4f} seconds")
                
                return (example_id, rule_results) 