import asyncio
from typing import List

from judgeval.v1 import Judgeval
from judgeval.v1.data import Example
from judgeval.v1.judges import Judge, NumericResponse


def test_basic_custom_scorer(client: Judgeval, random_name: str):
    class HappinessJudge(Judge[NumericResponse]):
        async def score(self, data: Example) -> NumericResponse:
            actual_output = data._properties.get("actual_output") or ""
            if "happy" in actual_output:
                return NumericResponse(value=1.0, reason="happy detected")
            elif "sad" in actual_output:
                return NumericResponse(value=0.0, reason="sad detected")
            else:
                return NumericResponse(value=0.5, reason="neutral")

    examples: List[Example] = [
        Example.create(actual_output="I'm happy"),
        Example.create(actual_output="I'm sad"),
        Example.create(actual_output="I dont know"),
    ]

    judge = HappinessJudge()
    results = []
    for example in examples:
        results.append(asyncio.run(judge.score(example)))

    assert results[0].value == 1.0
    assert results[1].value == 0.0
    assert results[2].value == 0.5
