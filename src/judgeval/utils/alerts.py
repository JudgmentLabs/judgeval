"""
Handling alerts in Judgeval.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class AlertStatus(str, Enum):
    """Status of an alert evaluation."""

    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"


class AlertResult(BaseModel):
    """
    Result of a rule evaluation.

    Attributes:
        rule_name: Name of the rule that was evaluated
        rule_id: Unique identifier of the rule
        status: Status of the alert (triggered or not)
        conditions_result: List of condition evaluation results
        metadata: Dictionary containing example_id, timestamp, and other metadata
        notification: Optional notification configuration for triggered alerts
        combine_type: The combination type used ("all" or "any")
        project_id: Optional project identifier
        trace_span_id: Optional trace span identifier
    """

    rule_name: str
    rule_id: Optional[str] = None  # The unique identifier of the rule
    status: AlertStatus
    conditions_result: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    notification: Optional[Any] = (
        None  # NotificationConfig when triggered, None otherwise
    )
    combine_type: Optional[str] = None  # "all" or "any"
    project_id: Optional[str] = None  # Project identifier
    trace_span_id: Optional[str] = None  # Trace span identifier

    @property
    def example_id(self) -> Optional[str]:
        """Get example_id from metadata for backward compatibility"""
        return self.metadata.get("example_id")

    @property
    def timestamp(self) -> Optional[str]:
        """Get timestamp from metadata for backward compatibility"""
        return self.metadata.get("timestamp")

    @property
    def conditions_results(self) -> List[Dict[str, Any]]:
        """Backwards compatibility property for the conditions_result field"""
        return self.conditions_result

    def model_dump(self, **kwargs):
        """
        Convert the AlertResult to a dictionary for JSON serialization.

        Args:
            **kwargs: Additional arguments to pass to Pydantic's model_dump

        Returns:
            dict: Dictionary representation of the AlertResult
        """
        data = (
            super().model_dump(**kwargs)
            if hasattr(super(), "model_dump")
            else super().dict(**kwargs)
        )

        # Handle the NotificationConfig object if it exists
        if hasattr(self, "notification") and self.notification is not None:
            if hasattr(self.notification, "model_dump"):
                data["notification"] = self.notification.model_dump()
            elif hasattr(self.notification, "dict"):
                data["notification"] = self.notification.dict()
            else:
                # Manually convert the notification to a dictionary
                notif = self.notification
                data["notification"] = {
                    "enabled": notif.enabled,
                    "communication_methods": notif.communication_methods,
                    "email_addresses": notif.email_addresses,
                    "slack_channels": getattr(notif, "slack_channels", []),
                    "send_at": notif.send_at,
                }

        return data
