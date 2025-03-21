from typing import List

from judgeval.constants import APIScorer
from judgeval.scorers.utils import (
    scorer_progress_meter,
    create_verbose_logs,
    check_example_params
)
from judgeval.data import Example, ExampleParams
from judgeval.scorers import JudgevalScorer


required_params = [
    ExampleParams.INPUT,
    ExampleParams.ACTUAL_OUTPUT,
    ExampleParams.EXPECTED_TOOLS,
    ExampleParams.TOOLS_CALLED,
]


def get_lcs(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]


class ExecutionOrderScorer(JudgevalScorer):
    def __init__(
        self,
        threshold: float = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        should_exact_match: bool = False,
        should_consider_ordering: bool = False,
    ):
        super().__init__(
            score_type=APIScorer.EXECUTION_ORDER,
            threshold=1 if strict_mode else threshold,
            evaluation_model=None,
            include_reason=include_reason,
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode
        )
        self.should_exact_match = should_exact_match
        self.should_consider_ordering = should_consider_ordering

    def measure(
        self,
        example: Example,
        _show_indicator: bool = True,
    ) -> float:
        check_example_params(example, required_params, self)

        with scorer_progress_meter(self, display_meter=_show_indicator):
            self.tools_called: List[str] = example.tools_called
            self.expected_tools: List[str] = example.expected_tools
            self.score = self._calculate_score()
            self.reason = self._generate_reason()
            self.success = self.score >= self.threshold
            self.verbose_logs = create_verbose_logs(
                self,
                steps=[
                    f"Expected Tools:\n{self.expected_tools}",
                    f"Tools Called:\n{self.tools_called}",
                    f"Score: {self.score}\nReason: {self.reason}",
                ],
            )
            return self.score

    async def a_measure(
        self, test_case: Example, _show_indicator: bool = True
    ) -> float:
        check_example_params(test_case, required_params, self)
        return self.measure(test_case, _show_indicator=_show_indicator)

    def _generate_reason(self):
        if self.should_exact_match:
            return f"{'Exact match' if self.tools_called == self.expected_tools else 'Not an exact match'}: expected {self.expected_tools}, called {self.tools_called}."

        elif self.should_consider_ordering:
            lcs = get_lcs(self.expected_tools, self.tools_called)
            missing = set(self.expected_tools) - set(self.tools_called)
            out_of_order = set(self.expected_tools) - set(lcs)

            if len(lcs) == len(self.expected_tools):
                return f"Correct ordering: all expected tools {self.expected_tools} were called in the correct order."
            else:
                issues = []
                if missing:
                    issues.append(f"missing tools {list(missing)}")
                if out_of_order:
                    issues.append(f"out-of-order tools {list(out_of_order)}")
                return f"Incorrect tool usage: {' and '.join(issues)}; expected {self.expected_tools}, called {self.tools_called}."

        else:
            used_expected = set(self.tools_called).intersection(
                set(self.expected_tools)
            )
            missing = set(self.expected_tools) - used_expected

            if len(used_expected) == len(self.expected_tools):
                return f"All expected tools {self.expected_tools} were called (order not considered)."
            else:
                return f"Incomplete tool usage: missing tools {list(missing)}; expected {self.expected_tools}, called {self.tools_called}."

    def _calculate_score(self):
        if self.should_exact_match:
            return 1.0 if self.tools_called == self.expected_tools else 0.0

        elif self.should_consider_ordering:
            longest_common_subsequence = get_lcs(
                self.expected_tools, self.tools_called
            )
            score = len(longest_common_subsequence) / len(self.expected_tools)

        else:
            used_expected_tools = set(self.tools_called).intersection(
                set(self.expected_tools)
            )
            score = len(used_expected_tools) / len(self.expected_tools)
        return 0 if self.strict_mode and score < self.threshold else score

    def _success_check(self) -> bool:
        try:
            self.success = self.score >= self.threshold
        except:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Execution Order"
    