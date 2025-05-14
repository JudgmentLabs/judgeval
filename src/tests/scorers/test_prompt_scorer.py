import pytest
from pydantic import Field
from unittest.mock import MagicMock, AsyncMock
from typing import List, Any

from judgeval.data import Example
from judgeval.scorers.prompt_scorer import PromptScorer, ClassifierScorer

# Test fixtures
@pytest.fixture
def example():
    return Example(
        input="This is a test input",
        actual_output="This is a test response",
        expected_output="Expected response",
        context=["Some context"],
        retrieval_context=["Retrieved context"],
        tools_called=["tool1", "tool2"],
        expected_tools=[{"tool_name": "tool1"}, {"tool_name": "tool2"}]
    )

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate = MagicMock(return_value='{"score": 0.8, "reason": "Test reason"}')
    model.a_generate = AsyncMock(return_value='{"score": 0.8, "reason": "Test reason"}')
    return model

# Simple implementation of PromptScorer for testing
class SampleScorer(PromptScorer):

    model: Any = Field(default=None)

    def __init__(self, mock_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = mock_model

    def _build_measure_prompt(self, example: Example) -> List[dict]:
        return [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": f"Response: {example.actual_output}"}
        ]
    
    def _build_schema(self) -> dict:
        return {"score": float, "reason": str}
    
    def _process_response(self, response: dict):
        return response["score"], response["reason"]
    
    def _success_check(self, **kwargs) -> bool:
        return self._result >= self.threshold

# Tests for PromptScorer
class TestPromptScorer:
    def test_init(self, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model)
        assert scorer.name == "test_scorer"
        assert scorer.threshold == 0.5
        assert scorer.include_reason is True
        assert scorer.async_mode is True
        
    def test_init_strict_mode(self, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model, strict_mode=True)
        assert scorer.threshold == 1
        
    def test_enforce_prompt_format(self, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model)
        prompt = [{"role": "system", "content": "Base prompt"}]
        schema = {"score": float, "reason": str}
        
        formatted = scorer._enforce_prompt_format(prompt, schema)
        assert "JSON format" in formatted[0]["content"]
        assert '"score": <score> (float)' in formatted[0]["content"]
        assert '"reason": <reason> (str)' in formatted[0]["content"]
        
    def test_enforce_prompt_format_invalid_input(self, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model)
        with pytest.raises(TypeError):
            scorer._enforce_prompt_format("invalid", {})
            
    @pytest.mark.asyncio
    async def test_a_score_example(self, example, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model)
        
        result = await scorer.a_score_example(example, _show_indicator=False)
        assert result == 0.8
        assert scorer.reason == "Test reason"
        
    def test_score_example_sync(self, example, mock_model):
        scorer = SampleScorer(name="test_scorer", mock_model=mock_model, async_mode=False)
        
        result = scorer.score_example(example, _show_indicator=False)
        assert result == 0.8
        assert scorer.reason == "Test reason"

# Tests for ClassifierScorer
class TestClassifierScorer:
    @pytest.fixture
    def classifier_conversation(self):
        return [
            {"role": "system", "content": "Evaluate if {{actual_output}} is positive"},
            {"role": "user", "content": "Please analyze."}
        ]
    
    @pytest.fixture
    def classifier_options(self):
        return {"positive": 1.0, "negative": 0.0}
    
    def test_classifier_init(self, classifier_conversation, classifier_options):
        scorer = ClassifierScorer(
            name="test_classifier",
            slug="test_classifier_slug",
            conversation=classifier_conversation,
            options=classifier_options
        )
        assert scorer.conversation == classifier_conversation
        assert scorer.options == classifier_options
        
    def test_build_measure_prompt(self, example, classifier_conversation, classifier_options):
        scorer = ClassifierScorer(
            name="test_classifier",
            slug="test_classifier_slug",
            conversation=classifier_conversation,
            options=classifier_options
        )
        
        prompt = scorer._build_measure_prompt(example)
        assert "This is a test response" in prompt[0]["content"]
        
    def test_process_response(self, classifier_conversation, classifier_options):
        scorer = ClassifierScorer(
            name="test_classifier",
            slug="test_classifier_slug",
            conversation=classifier_conversation,
            options=classifier_options
        )
        
        response = {"choice": "positive", "reason": "Test reason"}
        score, reason = scorer._process_response(response)
        assert score == 1.0
        assert reason == "Test reason"
        
    def test_process_response_invalid_choice(self, classifier_conversation, classifier_options):
        scorer = ClassifierScorer(
            name="test_classifier",
            slug="test_classifier_slug",
            conversation=classifier_conversation,
            options=classifier_options
        )
        
        response = {"choice": "invalid", "reason": "Test reason"}
        with pytest.raises(ValueError):
            scorer._process_response(response)
            
    def test_success_check(self, classifier_conversation, classifier_options):
        scorer = ClassifierScorer(
            name="test_classifier",
            slug="test_classifier_slug",
            conversation=classifier_conversation,
            options=classifier_options
        )
        
        scorer.score = 1.0
        assert scorer._success_check() is True
        
        scorer.score = 0.0
        assert scorer._success_check() is False
