"""
Integration tests for the Judgment client functionality.
Tests all major client operations and edge cases.
"""

import os
import pytest
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import random
import string
import logging
from typing import Generator

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
from judgeval.data.datasets.dataset import EvalDataset, GroundTruthExample
from judgeval.data.datasets.eval_dataset_client import EvalDatasetClient
from judgeval.scorers.prompt_scorer import ClassifierScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("JUDGMENT_API_KEY")

if not API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set", allow_module_level=True)

# Fixtures
@pytest.fixture(scope="session")
def client() -> JudgmentClient:
    """Create a single JudgmentClient instance for all tests."""
    return JudgmentClient(judgment_api_key=API_KEY)

@pytest.fixture
def random_name() -> str:
    """Generate a random name for test resources."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

# Original tests organized into classes
class TestBasicOperations:
    def test_dataset(self, client: JudgmentClient):
        """Test dataset creation and manipulation."""
        dataset: EvalDataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))

        client.push_dataset(alias="test_dataset_5", dataset=dataset, overwrite=False)
        
        dataset = client.pull_dataset(alias="test_dataset_5")
        assert dataset, "Failed to pull dataset"

    def test_pull_all_user_dataset_stats(self, client: JudgmentClient):
        dataset: EvalDataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        dataset.add_example(Example(input="input 3", actual_output="output 3"))
        random_name1 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        client.push_dataset(alias=random_name1, dataset=dataset, overwrite=False)

        dataset: EvalDataset = client.create_dataset()
        dataset.add_example(Example(input="input 1", actual_output="output 1"))
        dataset.add_example(Example(input="input 2", actual_output="output 2"))
        dataset.add_ground_truth(GroundTruthExample(input="input 1", actual_output="output 1"))
        dataset.add_ground_truth(GroundTruthExample(input="input 2", actual_output="output 2"))
        random_name2 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        client.push_dataset(alias=random_name2, dataset=dataset, overwrite=False)
        
        all_datasets_stats = client.pull_all_user_dataset_stats()
        print(all_datasets_stats)
        assert all_datasets_stats, "Failed to pull dataset"
        assert all_datasets_stats[random_name1]["example_count"] == 3, f"{random_name1} should have 3 examples"
        assert all_datasets_stats[random_name1]["ground_truth_count"] == 0, f"{random_name1} should have 0 ground truths"
        assert all_datasets_stats[random_name2]["example_count"] == 2, f"{random_name2} should have 2 examples"
        assert all_datasets_stats[random_name2]["ground_truth_count"] == 2, f"{random_name2} should have 2 ground truths"

    
    def run_eval_helper(self, client: JudgmentClient, project_name: str, eval_run_name: str):
        """Helper function to run evaluation."""
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

        client.run_evaluation(
            examples=[example1, example2],
            scorers=[scorer, scorer2],
            model="QWEN",
            metadata={"batch": "test"},
            project_name=project_name,
            eval_run_name=eval_run_name,
            log_results=True,
            override=True,
        )

    def test_run_eval(self, client: JudgmentClient):
        """Test basic evaluation workflow."""
        PROJECT_NAME = "OutreachWorkflow"
        EVAL_RUN_NAME = "ColdEmailGenerator-Improve-BasePrompt"

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        results = client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
        assert results, f"No evaluation results found for {EVAL_RUN_NAME}"

    def test_delete_eval_by_project_and_run_name(self, client: JudgmentClient):
        """Test delete evaluation by project and run name workflow."""
        PROJECT_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        client.delete_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)

    def test_delete_eval_by_project(self, client: JudgmentClient):
        """Test delete evaluation by project workflow."""
        PROJECT_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME2 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME2)

        client.delete_project_evals(project_name=PROJECT_NAME)
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
        
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME2)

    def test_assert_test(self, client: JudgmentClient):
        """Test assertion functionality."""
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
                project_name="test_project",
                examples=[example, example1, example2],
                scorers=[scorer, scorer1],
                model="QWEN",
                override=True
            )

class TestAdvancedFeatures:
    def test_json_scorer(self, client: JudgmentClient):
        """Test JSON scorer functionality."""
        # Original test content preserved
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
        assert res, "JSON scorer evaluation failed"

    def test_override_eval(self, client: JudgmentClient, random_name: str):
        """Test evaluation override behavior."""
        # Original test content with random name for uniqueness
        example1 = Example(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
            trace_id="2231abe3-e7e0-4909-8ab7-b4ab60b645c6"
        )
        
        scorer = FaithfulnessScorer(threshold=0.5)

        PROJECT_NAME = "test_eval_run_naming_collisions"
        EVAL_RUN_NAME = random_name

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
        with pytest.raises(ValueError, match="already exists") as exc_info:
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

    def test_evaluate_dataset(self, client: JudgmentClient):
        """Test dataset evaluation."""
        # Original test content preserved
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
        assert res, "Dataset evaluation failed"
        
    def test_classifier_scorer(self, client: JudgmentClient, random_name: str):
        """Test classifier scorer functionality."""
        # Original test content with random slug
        random_slug = random_name
        faithfulness_scorer = FaithfulnessScorer(threshold=0.5)
        
        # Creating a classifier scorer from SDK
        classifier_scorer_custom = ClassifierScorer(
            name="Test Classifier Scorer",
            slug=random_slug,  # Use random slug instead of hardcoded value
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
            scorers=[faithfulness_scorer, classifier_scorer_custom],
            model="QWEN",
            log_results=True,
            eval_run_name="ToneScorerTest",
            project_name="ToneScorerTest",
            override=True,
        )

class TestTraceOperations:
    """Tests for trace-related operations."""
    
    def test_fetch_traces_by_time_period(self, client: JudgmentClient):
        """Test successful cases with different time periods."""
        for hours in [1, 3, 6, 12, 24, 72, 168]:
            response = requests.post(
                f"{SERVER_URL}/traces/fetch_by_time_period/",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json={"hours": hours}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_fetch_traces_invalid_period(self, client: JudgmentClient):
        """Test invalid time periods."""
        for hours in [0, 2, 4]:
            response = requests.post(
                f"{SERVER_URL}/traces/fetch_by_time_period/",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}"
                },
                json={"hours": hours}
            )
            assert response.status_code == 400

    def test_fetch_traces_missing_api_key(self, client: JudgmentClient):
        """Test missing API key scenario."""
        response = requests.post(
            f"{SERVER_URL}/traces/fetch_by_time_period/",
            headers={
                "Content-Type": "application/json",
            },
            json={"hours": 12}
        )
        assert response.status_code in [401, 403]
        
@pytest.mark.skipif(not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                   reason="VertexAI credentials not configured")

class TestCustomJudges:
    def test_custom_judge_vertexai(self, client: JudgmentClient):
        """Test VertexAI custom judge."""
        # Original test content preserved
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
            assert res, "Custom judge evaluation failed"
        except ImportError:
            pytest.skip("VertexAI package not installed")

def pytest_configure(config):
    """Add markers for test categories."""
    config.addinivalue_line("markers", "basic: mark test as testing basic functionality")
    config.addinivalue_line("markers", "advanced: mark test as testing advanced features")
    config.addinivalue_line("markers", "custom: mark test as testing custom components")
    config.addinivalue_line("markers", "traces: mark test as testing trace operations")

def pytest_collection_modifyitems(items):
    """Add markers to tests based on their class."""
    for item in items:
        if "TestBasicOperations" in item.nodeid:
            item.add_marker(pytest.mark.basic)
        elif "TestAdvancedFeatures" in item.nodeid:
            item.add_marker(pytest.mark.advanced)
        elif "TestCustomJudges" in item.nodeid:
            item.add_marker(pytest.mark.custom)
        elif "TestTraceOperations" in item.nodeid:
            item.add_marker(pytest.mark.traces)

def run_selected_tests(client, test_names: list[str]):
    """
    Run only the specified tests by name.
    
    Args:
        test_names (list[str]): List of test function names to run (without 'test_' prefix)
    """
    judgeval_client = client
    print("Client initialized successfully")
    print("*" * 40)
    
    test_basic_operations = TestBasicOperations()
    test_advanced_features = TestAdvancedFeatures()
    test_custom_judges = TestCustomJudges()
    test_trace_operations = TestTraceOperations()
    
    test_map = {
        'dataset': test_basic_operations.test_dataset,
        'pull_all_user_dataset_stats': test_basic_operations.test_pull_all_user_dataset_stats,
        'run_eval': test_basic_operations.test_run_eval,
        'delete_eval_by_project_and_run_name': test_basic_operations.test_delete_eval_by_project_and_run_name,
        'delete_eval_by_project': test_basic_operations.test_delete_eval_by_project,
        'assert_test': test_basic_operations.test_assert_test,
        'json_scorer': test_advanced_features.test_json_scorer,
        'override_eval': test_advanced_features.test_override_eval,
        'evaluate_dataset': test_advanced_features.test_evaluate_dataset,
        'classifier_scorer': test_advanced_features.test_classifier_scorer,
        'custom_judge_vertexai': test_custom_judges.test_custom_judge_vertexai,
        'fetch_traces': test_trace_operations.test_fetch_traces_by_time_period,
        'fetch_traces_invalid': test_trace_operations.test_fetch_traces_invalid_period,
        'fetch_traces_missing_key': test_trace_operations.test_fetch_traces_missing_api_key
    }
    
    for test_name in test_names:
        if test_name not in test_map:
            print(f"Warning: Test '{test_name}' not found")
            continue
            
        print(f"Running test: {test_name}")
        test_map[test_name](judgeval_client)
        print(f"{test_name} test successful")
        print("*" * 40)
    
    print("Selected tests completed")

# Can either run as a script or a pytest module
# Script makes it easier to specify which tests to run
if __name__ == "__main__":
    client = JudgmentClient(judgment_api_key=API_KEY)
    
    run_selected_tests(client, [
        'dataset',
        'run_eval', 
        'delete_eval_by_project_and_run_name',
        'delete_eval_by_project',
        'assert_test',
        'json_scorer',
        'override_eval',
        'evaluate_dataset',
        'classifier_scorer',
        'custom_judge_vertexai',
        'fetch_traces',
        'fetch_traces_invalid',
        'fetch_traces_missing_key'
        'pull_all_user_dataset_stats',
    ])
