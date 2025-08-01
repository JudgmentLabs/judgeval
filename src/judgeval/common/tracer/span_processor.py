"""
Base class for span processors with default no-op implementations.

This eliminates the need for optional typing and null checks.
When monitoring is disabled, we use this base class directly.
When monitoring is enabled, we use JudgmentSpanProcessor which overrides the methods.
"""

from typing import Dict, Any, Callable

from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


class SpanProcessorBase:
    """
    Base class for Judgment span processors with default no-op implementations.

    This eliminates the need for optional typing and null checks.
    When monitoring is disabled, we use this base class directly.
    When monitoring is enabled, we use JudgmentSpanProcessor which overrides the methods.
    """

    def queue_span_update(self, span: TraceSpan, span_state: str = "input") -> None:
        pass

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> None:
        pass

    def queue_trace_upsert(
        self,
        trace_data: Dict[str, Any],
        upsert_callback: Callable[[Dict[str, Any]], Dict[str, Any]],
        final_save: bool = False,
    ) -> None:
        """
        Queue a trace upsert to be processed in the background.

        Args:
            trace_data: The trace data to upsert
            upsert_callback: Callback function that performs the actual upsert
            final_save: Whether this is the final save
        """
        pass

    def flush_pending_spans(self) -> None:
        pass

    def flush_pending_traces(self) -> None:
        """Flush all pending trace upserts."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        pass
