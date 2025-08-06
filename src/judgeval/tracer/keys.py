"""
Identifiers used by Judgeval to store specific types of data in the spans.
"""

import enum
from enum import Enum


@enum.unique
class EventKeys(str, Enum):
    JUDGMENT_INPUT = "judgment.input"
    JUDGMENT_OUTPUT = "judgment.output"
    LLM_USAGE = "llm.usage"


@enum.unique
class ResourceKeys(str, Enum):
    JUDGMENT_PROJECT_ID = "judgment.project_id"
