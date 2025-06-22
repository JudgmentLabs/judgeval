from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from judgeval.common.tracer.model import TraceSave


class ABCStorage(ABC):
    """
    Abstract base class for storage systems, responsible for storing data needed for judgeval operations.
    """

    @abstractmethod
    def save_trace(
        self, trace_data: TraceSave, trace_id: str, project_name: str
    ) -> str:
        """
        Save trace data to the storage system.

        Args:
            trace_data (TraceSave): The trace data to be saved.
            trace_id (str): Unique identifier for the trace.
            project_name (str): Name of the project associated with the trace.

        Returns:
            str: The URL or identifier of the saved trace.
        """
        ...
