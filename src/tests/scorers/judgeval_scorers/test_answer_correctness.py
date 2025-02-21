import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.answer_correctness import AnswerCorrectnessScorer
from judgeval.constants import APIScorer


class TestAnswerCorrectnessScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = AnswerCorrectnessScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.ANSWER_CORRECTNESS

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            AnswerCorrectnessScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            AnswerCorrectnessScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = AnswerCorrectnessScorer(threshold=0.5)
        assert scorer.__name__ == "Answer Correctness"