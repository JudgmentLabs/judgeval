import pytest
from judgeval.scorers.judgeval_scorers.local_implementations.execution_order import ExecutionOrderScorer
from judgeval.constants import APIScorer


class TestExecutionOrderScorer:
    def test_init(self):
        # Test initialization with valid threshold
        threshold = 0.7
        scorer = ExecutionOrderScorer(threshold=threshold)
        
        assert scorer.threshold == threshold
        assert scorer.score_type == APIScorer.EXECUTION_ORDER

    def test_init_invalid_threshold(self):
        # Test initialization with invalid threshold values
        with pytest.raises(ValueError):
            ExecutionOrderScorer(threshold=-0.1)
        
        with pytest.raises(ValueError):
            ExecutionOrderScorer(threshold=1.1)

    def test_name_property(self):
        # Test the __name__ property
        scorer = ExecutionOrderScorer(threshold=0.5)
        assert scorer.__name__ == "Execution Order"
