from __future__ import annotations

import ast

from judgeval.cli.upload_judge import parse_judge
from judgeval.judges import ArbitraryResponse

SOURCE = """
class ToolCallCount(Judge[ArbitraryResponse]):
    async def score(self, data: Example) -> ArbitraryResponse:
        return ArbitraryResponse(value=2, reason="two")
"""


def test_arbitrary_judge_parses_and_keeps_value_type():
    assert parse_judge(ast.parse(SOURCE)) == ("ToolCallCount", None, "arbitrary", None)
    assert type(ArbitraryResponse(value=2, reason="").value) is int
    assert ArbitraryResponse(value=True, reason="").value is True
    assert ArbitraryResponse(value={"city": "Paris"}, reason="").value == {
        "city": "Paris"
    }
    assert ArbitraryResponse(value=[1, "a"], reason="").value == [1, "a"]
