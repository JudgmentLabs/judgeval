import pytest
from unittest.mock import Mock, patch

from judgeval.scorers.base_scorer import BaseScorer
from judgeval.judges import JudgevalJudge
from judgeval.common.exceptions import InvalidJudgeModelError


class MockJudge(JudgevalJudge):
    """Mock implementation of judgevalJudge for testing"""

    def load_model(self, *args, **kwargs):
        return Mock()

    def generate(self, *args, **kwargs) -> str:
        return "mock response"

    async def a_generate(self, *args, **kwargs) -> str:
        return "mock async response"

    def get_model_name(self, *args, **kwargs) -> str:
        return "mock-model"


class SampleScorer(BaseScorer):
    """Concrete implementation of BaseScorer for testing"""

    def score_example(self, example, *args, **kwargs) -> float:
        return 0.8

    async def a_score_example(self, example, *args, **kwargs) -> float:
        return 0.9

    def success_check(self) -> bool:
        return self.score >= self.threshold if self.score is not None else False


@pytest.fixture
def basic_scorer():
    return SampleScorer(score_type="test_scorer", threshold=0.7)


@pytest.fixture
def mock_judge():
    return MockJudge(model_name="mock-model")


class TestBaseScorer:
    def test_initialization(self):
        """Test basic initialization with minimal parameters"""
        scorer = SampleScorer(score_type="test", threshold=0.5)
        assert scorer.score_type == "test"
        assert scorer.threshold == 0.5
        assert scorer.score is None

    def test_initialization_with_all_params(self):
        """Test initialization with all optional parameters"""
        additional_metadata = {"key": "value"}
        scorer = SampleScorer(
            score_type="test",
            threshold=0.5,
            score=0.8,
            score_breakdown={"detail": 0.8},
            reason="test reason",
            success=True,
            evaluation_model="gpt-4",
            strict_mode=True,
            error=None,
            additional_metadata=additional_metadata,
        )

        assert scorer.score == 0.8
        assert scorer.score_breakdown == {"detail": 0.8}
        assert scorer.reason == "test reason"
        assert scorer.success is True
        assert scorer.strict_mode is True
        assert scorer.additional_metadata == additional_metadata

    @patch("judgeval.scorers.base_scorer.create_judge")
    def test_add_model_success(self, mock_create_judge, mock_judge, basic_scorer):
        """Test successful model addition"""
        mock_create_judge.return_value = (mock_judge, True)

        scorer = basic_scorer
        scorer._add_model("mock-model")

        assert scorer.evaluation_model == "mock-model"
        assert scorer.using_native_model is True
        mock_create_judge.assert_called_once_with("mock-model")

    @patch("judgeval.scorers.base_scorer.create_judge")
    def test_add_model_error(self, mock_create_judge, basic_scorer):
        """Test model addition with invalid model"""
        mock_create_judge.side_effect = InvalidJudgeModelError("Invalid model")

        scorer = basic_scorer
        with pytest.raises(InvalidJudgeModelError):
            scorer._add_model("invalid-model")

    def test_score_example_implementation(self, basic_scorer):
        """Test score_example returns expected value"""
        score = basic_scorer.score_example({"test": "example"})
        assert score == 0.8

    @pytest.mark.asyncio
    async def test_a_score_example_implementation(self, basic_scorer):
        """Test async score_example returns expected value"""
        score = await basic_scorer.a_score_example({"test": "example"})
        assert score == 0.9

    def test_success_check_implementation(self, basic_scorer):
        """Test success_check with various scores"""
        # Test with score above threshold
        basic_scorer.score = 0.8
        assert basic_scorer.success_check() is True

        # Test with score below threshold
        basic_scorer.score = 0.6
        assert basic_scorer.success_check() is False

        # Test with no score
        basic_scorer.score = None
        assert basic_scorer.success_check() is False

    def test_str_representation(self, basic_scorer):
        """Test string representation of scorer"""
        str_rep = str(basic_scorer)
        assert "BaseScorer" in str_rep
        assert "test_scorer" in str_rep
        assert "0.7" in str_rep  # threshold value
