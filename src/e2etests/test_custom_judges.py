"""
Tests for custom judge implementations in the JudgmentClient.
"""

import pytest
import os

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.judges import JudgevalJudge
from judgeval.scorers import FaithfulnessScorer

@pytest.mark.skipif(not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                   reason="VertexAI credentials not configured")
@pytest.mark.custom
class TestCustomJudges:
    def test_custom_judge_vertexai(self, client: JudgmentClient):
        """Test VertexAI custom judge."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            PROJECT_ID = "judgment-labs"
            vertexai.init(project=PROJECT_ID, location="us-west1")

            class VertexAIJudge(JudgevalJudge):

                def __init__(self, model_name: str = "gemini-1.5-flash-002"):
                    self.model_name = model_name
                    self.model = GenerativeModel(self.model_name)

                def load_model(self):
                    return self.model

                def generate(self, prompt) -> str:
                    # prompt is a List[dict] (conversation history)
                    # For models that don't support conversation history, we need to convert to string
                    # If you're using a model that supports chat history, you can just pass the prompt directly
                    response = self.model.generate_content(str(prompt))
                    return response.text

                async def a_generate(self, prompt) -> str:
                    # prompt is a List[dict] (conversation history)
                    # For models that don't support conversation history, we need to convert to string
                    # If you're using a model that supports chat history, you can just pass the prompt directly
                    response = await self.model.generate_content_async(str(prompt))
                    return response.text

                def get_model_name(self) -> str:
                    return self.model_name

            example = Example(
                input={
                    "question": "What is the largest animal in the world?"
                },
                actual_output={
                    "answer": "The blue whale is the largest known animal."
                },
                retrieval_context=["The blue whale is the largest known animal."]
            )

            judge = VertexAIJudge()

            res = client.run_evaluation(
                examples=[example],
                scorers=[FaithfulnessScorer()],
                model=judge,
                eval_run_name="custom_judge_test",
                project_name="custom_judge_test",
                override=True,
                use_judgment=False
            )
            assert res, "Custom judge evaluation failed"
        except ImportError:
            pytest.skip("VertexAI package not installed") 