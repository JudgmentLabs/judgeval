from __future__ import annotations

import ast

from judgeval.cli.upload_judge import parse_judge
from judgeval.judges import LiteralResponse

SOURCE = """
class ToolCallCount(Judge[LiteralResponse]):
    async def score(self, data: Example) -> LiteralResponse:
        return LiteralResponse(value=2, reason="two")
"""


def test_literal_judge_parses_and_keeps_value_type():
    assert parse_judge(ast.parse(SOURCE)) == ("ToolCallCount", None, "literal", None)
    assert type(LiteralResponse(value=2, reason="").value) is int
    assert LiteralResponse(value=True, reason="").value is True
    assert LiteralResponse(value={"city": "Paris"}, reason="").value == {
        "city": "Paris"
    }
    assert LiteralResponse(value=[1, "a"], reason="").value == [1, "a"]
