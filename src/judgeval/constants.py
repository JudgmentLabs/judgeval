from __future__ import annotations

from enum import Enum

JUDGEVAL_TRACER_INSTRUMENTING_MODULE_NAME = "judgeval"


class APIScorerType(str, Enum):
    """
    Collection of proprietary scorers implemented by Judgment.

    These are ready-made evaluation scorers that can be used to evaluate
    Examples via the Judgment API.
    """

    PROMPT_SCORER = "Prompt Scorer"
    FAITHFULNESS = "Faithfulness"
    ANSWER_RELEVANCY = "Answer Relevancy"
    ANSWER_CORRECTNESS = "Answer Correctness"
    INSTRUCTION_ADHERENCE = "Instruction Adherence"
    EXECUTION_ORDER = "Execution Order"
    CUSTOM = "Custom"

    @classmethod
    def __missing__(cls, value: str) -> APIScorerType:
        for member in cls:
            if member.value == value.lower():
                return member

        raise ValueError(f"Invalid scorer type: {value}")
