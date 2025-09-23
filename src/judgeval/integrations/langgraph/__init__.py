from __future__ import annotations

from abc import ABC
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Type
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult, ChatGeneration
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        FunctionMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "Judgeval's langgraph integration requires langchain to be installed. Please install it with `pip install judgeval[langchain]`"
    ) from e
import os


class Langgraph(ABC):
    @staticmethod
    def initialize(otel_only: bool = True):
        os.environ["LANGSMITH_OTEL_ENABLED"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        if otel_only:
            os.environ["LANGSMITH_OTEL_ONLY"] = "true"
