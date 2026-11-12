import pytest

from judgeval.data.example import Example
from judgeval.hosted.responses import BinaryResponse
from judgeval.judges import Judge


class _ToolUseJudge(Judge[BinaryResponse]):
    def __init__(self) -> None:
        # Deliberately skips super().__init__(), as customer judges may.
        self.limit = 5

    async def score(self, data: Example) -> BinaryResponse:
        self.metric("tool_calls", 3)
        self.metric("tool_calls", 7)
        self.metric("cost_usd", 0.25)
        return BinaryResponse(value=7 <= self.limit, reason="7 tool calls")


@pytest.mark.asyncio
async def test_metric_records_latest_number_per_name_on_the_judge():
    judge, other = _ToolUseJudge(), _ToolUseJudge()

    await judge.score(Example.create(input="hi"))

    assert judge.recorded_metrics == {"tool_calls": 7.0, "cost_usd": 0.25}
    assert not hasattr(other, "recorded_metrics")
    with pytest.raises(ValueError, match="finite"):
        judge.metric("ratio", float("nan"))
