"""
Sanity checks for judgment client functionality
"""

import os
from pydantic import BaseModel
import pytest

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    HallucinationScorer,
    AnswerRelevancyScorer,
    JSONCorrectnessScorer
)
from judgeval.judges import TogetherJudge, JudgevalJudge
from playground import CustomFaithfulnessMetric
from judgeval.data.datasets.dataset import EvalDataset
from dotenv import load_dotenv
import random
import string

from judgeval.scorers.prompt_scorer import ClassifierScorer

load_dotenv()

def get_client():
    return JudgmentClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"))


def test_dataset(client: JudgmentClient):
    dataset: EvalDataset = client.create_dataset()
    dataset.add_example(Example(input="input 1", actual_output="output 1"))

    client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
    
    # PULL
    dataset = client.pull_dataset(alias="test_dataset_5")
    print(dataset)


def test_run_eval(client: JudgmentClient):
    # Single step in our workflow, an outreach Sales Agent

    example1 = Example(
        input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
        actual_output="Dear Ms. Chen,\n\nI noticed TechCorp's recent launch of your AI analytics platform and was impressed by its enterprise-focused approach. Your experience from Google clearly shines through in building scalable solutions, as evidenced by your impressive 50+ enterprise client base.\n\nWould you be open to a brief call to discuss how we could potentially collaborate?\n\nBest regards,\nAlex",
        retrieval_context=["TechCorp launched AI analytics platform in 2024", "Sarah Chen is CEO, ex-Google executive", "Current client base: 50+ enterprise customers"],
    )

    example2 = Example(
        input="Generate a cold outreach email for GreenEnergy Solutions. Facts: They're developing solar panel technology that's 30% more efficient. They're looking to expand into the European market. They won a sustainability award in 2023.",
        actual_output="Dear GreenEnergy Solutions team,\n\nCongratulations on your 2023 sustainability award! Your innovative solar panel technology with 30% higher efficiency is exactly what the European market needs right now.\n\nI'd love to discuss how we could support your European expansion plans.\n\nBest regards,\nAlex",
        expected_output="A professional cold email mentioning the sustainability award, solar technology innovation, and European expansion plans",
        context=["Business Development"],
        retrieval_context=["GreenEnergy Solutions won 2023 sustainability award", "New solar technology 30% more efficient", "Planning European market expansion"],
    )

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer2 = HallucinationScorer(threshold=0.5)
    # c_scorer = CustomFaithfulnessMetric(threshold=0.6)

    PROJECT_NAME = "OutreachWorkflow"
    EVAL_RUN_NAME = "ColdEmailGenerator-Improve-BasePrompt"
    
    client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer, scorer2],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=True,
    )

    results = client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
    print(f"Evaluation results for {EVAL_RUN_NAME} from database:", results)


def test_assert_test(client: JudgmentClient):

    # Create examples and scorers as before
    example = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
    )

    example1 = Example(
        input="How much are your croissants?",
        actual_output="Sorry, we don't accept electronic returns.",
    )

    example2 = Example(
        input="Who is the best basketball player in the world?",
        actual_output="No, the room is too small.",
    )

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer1 = AnswerRelevancyScorer(threshold=0.5)

    with pytest.raises(AssertionError):
        client.assert_test(
            eval_run_name="test_eval",
            examples=[example, example1, example2],
            scorers=[scorer, scorer1],
            model="QWEN",
        )


def test_json_scorer(client: JudgmentClient):

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output='{"tool": "authentication"}',
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    class SampleSchema(BaseModel):
        tool: str

    scorer = JSONCorrectnessScorer(threshold=0.5, json_schema=SampleSchema)
    PROJECT_NAME = "test_project"
    EVAL_RUN_NAME = "test_json_scorer"
    
    res = client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=True,
        use_judgment=True,
    )

    print(res)



def test_override_eval(client: JudgmentClient):
    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
    )
    
    scorer = FaithfulnessScorer(threshold=0.5)

    PROJECT_NAME = "test_eval_run_naming_collisions"
    EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=12))

    # First run should succeed
    client.run_evaluation(
        examples=[example1],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=True,
        override=False,
    )
    
    # Second run with log_results=False should succeed
    client.run_evaluation(
        examples=[example1],
        scorers=[scorer],
        model="QWEN",
        metadata={"batch": "test"},
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        log_results=False,
        override=False,
    )
    
    # Third run with override=True should succeed
    try:
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="QWEN",
            metadata={"batch": "test"},
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            log_results=True,
            override=True,
        )
    except ValueError as e:
        print(f"Unexpected error in override run: {e}")
        raise
    
    # Final non-override run should fail
    try:
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="QWEN",
            metadata={"batch": "test"},
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            log_results=True,
            override=False,
        )
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as e:
        if "already exists" not in str(e):
            raise
        print(f"Successfully caught expected error: {e}")
    

def test_evaluate_dataset(client: JudgmentClient):

    example1 = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
    )

    example2 = Example(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        name="Password Reset",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"}
    )

    dataset = EvalDataset(examples=[example1, example2])
    res = client.evaluate_dataset(
        dataset=dataset,
        scorers=[FaithfulnessScorer(threshold=0.5)],
        model="QWEN",
        metadata={"batch": "test"},
    )

    print(res)
    

def test_classifier_scorer(client: JudgmentClient):
    # Comment out the fetching of non-existent classifier scorer
    # classifier_scorer = client.fetch_classifier_scorer("tonescorer-pt0z")
    faithfulness_scorer = FaithfulnessScorer(threshold=0.5)
    
    # Creating a classifier scorer from SDK
    classifier_scorer_custom = ClassifierScorer(
        name="Test Classifier Scorer",
        slug="testslug",
        threshold=0.5,
        conversation=[],
        options={}
    )
    
    classifier_scorer_custom.update_conversation(conversation=[{"role": "user", "content": "What is the capital of France?"}])
    classifier_scorer_custom.update_options(options={"yes": 1, "no": 0})
    
    slug = client.push_classifier_scorer(scorer=classifier_scorer_custom)
    
    classifier_scorer_custom = client.fetch_classifier_scorer(slug=slug)
    
    example1 = Example(
        input="What is the capital of France?",
        actual_output="Paris",
        retrieval_context=["The capital of France is Paris."],
    )

    res = client.run_evaluation(
        examples=[example1],
        scorers=[faithfulness_scorer, classifier_scorer_custom],  # Removed classifier_scorer from list
        model="QWEN",
        log_results=True,
        eval_run_name="ToneScorerTest",
        project_name="ToneScorerTest",
        override=True,
    )


def test_custom_judge_vertexai(client: JudgmentClient):

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
        input="What is the largest animal in the world?",
        actual_output="The blue whale is the largest known animal.",
        retrieval_context=["The blue whale is the largest known animal."],
    )

    judge = VertexAIJudge()

    res = client.run_evaluation(
        examples=[example],
        scorers=[CustomFaithfulnessMetric()],
        model=judge,
        eval_run_name="custom_judge_test",
        project_name="custom_judge_test",
        override=True
    )
    print(res)


def test_custom_judge_vertexai(client: JudgmentClient):
    
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
        input="What is the largest animal in the world?",
        actual_output="The blue whale is the largest known animal.",
        retrieval_context=["The blue whale is the largest known animal."],
    )

    judge = VertexAIJudge()

    res = client.run_evaluation(
        examples=[example],
        scorers=[CustomFaithfulnessMetric()],
        model=judge,
        eval_run_name="custom_judge_test",
        project_name="custom_judge_test",
        override=True
    )
    print(res)


if __name__ == "__main__":
    # Test client functionality
    client = get_client()
    print("Client initialized successfully")
    print("*" * 40)

    print("Testing dataset creation, pushing, and pulling")
    test_dataset(client)
    print("Dataset creation, pushing, and pulling successful")
    print("*" * 40)
    
    print("Testing evaluation run")
    test_run_eval(client)
    print("Evaluation run successful")
    print("*" * 40)

    print("Testing assert test")
    test_assert_test(client)
    print("Assert test successful")
    print("*" * 40)

    print("Testing JSON scorer")
    test_json_scorer(client)
    print("JSON scorer test successful")
    print("*" * 40)
    
    print("Testing evaluation run override")
    test_override_eval(client)
    print("Evaluation run override successful")
    print("*" * 40)
    
    print("Testing dataset evaluation")
    test_evaluate_dataset(client)
    print("Dataset evaluation successful")
    print("*" * 40)
    
    print("Testing classifier scorer")
    test_classifier_scorer(client)
    print("Classifier scorer test successful")
    print("*" * 40)

    print("Testing custom judge")
    test_custom_judge_vertexai(client)
    print("Custom judge test successful")
    print("*" * 40)

    print("All tests passed successfully")
