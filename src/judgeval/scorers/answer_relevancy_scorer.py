from judgeval.scorers.judgeval_scorer import JudgevalScorer
from judgeval.data import Example
from difflib import SequenceMatcher

class AnswerRelevancyScorer(JudgevalScorer):
    def __init__(self, threshold: float):
        super().__init__(score_type="answer_relevancy", threshold=threshold)

    def score_example(self, example: Example, *args, **kwargs) -> float:
        expected = example.expected_output or ""
        actual = getattr(example, "actual_output", "") or ""
        score = SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        self.score = score
        return score

    # async def a_score_example(self, example: Example, *args, **kwargs) -> float:
    #     return self.score_example(example, *args, **kwargs)
    pass

    def _success_check(self) -> bool:
        return self.score is not None and self.score >= self.threshold
