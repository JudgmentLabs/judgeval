"""
Tests for evaluation operations in the JudgmentClient.
"""

import pytest
import random
import string

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    HallucinationScorer,
    AnswerRelevancyScorer,
    JSONCorrectnessScorer
)
from judgeval.data.datasets.dataset import EvalDataset
from pydantic import BaseModel
from judgeval.scorers.prompt_scorer import ClassifierScorer

@pytest.mark.basic
class TestEvalOperations:
    def run_eval_helper(self, client: JudgmentClient, project_name: str, eval_run_name: str):
        """Helper function to run evaluation."""
        # Single step in our workflow, an outreach Sales Agent
        example1 = Example(

            # input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
            input= {
                "task" : "Generate a cold outreach email",
                "company" : "TechCorp",
                "facts": [
                    "Recently launched an AI-powered analytics platform",
                    "CEO Sarah Chen previously worked at Google",
                    "Has 50+ enterprise clients"
                ]
            },
            actual_output={
                "email": {
                    "recipient": "Ms. Chen",
                    "body": [
                        "Noticed TechCorp's recent launch of your AI analytics platform",
                        "Impressed by enterprise-focused approach",
                        "Mentioned experience from Google and 50+ clients",
                        "Request for collaboration call"
                    ],
                    "signature": "Best regards,\nAlex"
                }
            },
            retrieval_context=[
                "TechCorp launched AI analytics platform in 2024",
                "Sarah Chen is CEO, ex-Google executive", 
                "Current client base: 50+ enterprise customers"],
        )

        example2 = Example(
            input= {
                "task" : "Generate a cold outreach email",
                "company" : "GreenEnergy Solutions",
                "facts": [
                    "They're developing solar panel technology that's 30% more efficient",
                    "They're looking to expand into the European market",
                    "Won a sustainability award in 2023"
                ]
            },
            actual_output={
                "email": {
                    "recipient": "GreenEnergy Solutions team",
                    "body": [
                        "Congratulations on your 2023 sustainability award!",
                        "Your innovative solar panel technology with 30% higher efficiency is exactly what the European market needs right now.",
                        "I'd love to discuss how we could support your European expansion plans.",
                        "Best regards,",
                        "Alex"
                    ],
                    "signature": "Best regards,\nAlex"
                }
            },
            expected_output={
                "description": "A professional cold email mentioning the sustainability award, solar technology innovation, and European expansion plans",
                "required_elements": {
                    "award": "2023 sustainability award",
                    "technology": "Solar panel technology with 30% higher efficiency",
                    "expansion": "European market expansion plans"
                },
                "tone": "professional",
                "structure": {
                    "greeting": "Should address the recipient appropriately",
                    "body": [
                        "Mention the sustainability award",
                        "Highlight the solar technology innovation",
                        "Reference the European expansion plans"
                    ],
                    "call_to_action": "Request for discussion about supporting expansion",
                    "signature": "Professional closing"
                }
            },
            context=["Business Development"],
            retrieval_context=["GreenEnergy Solutions won 2023 sustainability award", "New solar technology 30% more efficient", "Planning European market expansion"],
        )

        scorer = FaithfulnessScorer(threshold=0.5)
        scorer2 = HallucinationScorer(threshold=0.5)

        client.run_evaluation(
            examples=[example1, example2],
            scorers=[scorer, scorer2],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
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

        client.delete_project(project_name=PROJECT_NAME)

    def test_delete_eval_by_project_and_run_name(self, client: JudgmentClient):
        """Test delete evaluation by project and run name workflow."""
        PROJECT_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        client.delete_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
        client.delete_project(project_name=PROJECT_NAME)
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)

    def test_delete_eval_by_project(self, client: JudgmentClient):
        """Test delete evaluation by project workflow."""
        PROJECT_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
        EVAL_RUN_NAME2 = ''.join(random.choices(string.ascii_letters + string.digits, k=20))

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME2)

        client.delete_project(project_name=PROJECT_NAME)
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME)
        
        with pytest.raises(ValueError, match="Error fetching eval results"):
            client.pull_eval(project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME2)

    def test_assert_test(self, client: JudgmentClient):
        """Test assertion functionality."""
        # Create examples and scorers as before
        example = Example(
            input={
                "question": "What if these shoes don't fit?",
                "category": "returns"
            },
            actual_output={
                "response": "We offer a 30-day full refund at no extra cost.",
                "policy": "30-day refund"
            },
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        )

        example1 = Example(
            input={
                "question": "How much are your croissants?",
                "category": "pricing"
            },
            actual_output={
                "response": "Sorry, we don't accept electronic returns.",
                "error": "mismatched_response"
            }
        )

        example2 = Example(
            input={
                "question": "Who is the best basketball player in the world?",
                "category": "sports"
            },
            actual_output={
                "response": "No, the room is too small.",
                "error": "irrelevant_response"
            }
        )

        scorer = FaithfulnessScorer(threshold=0.5)
        scorer1 = AnswerRelevancyScorer(threshold=0.5)

        with pytest.raises(AssertionError):
            client.assert_test(
                eval_run_name="test_eval",
                project_name="test_project",
                examples=[example, example1, example2],
                scorers=[scorer, scorer1],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                override=True
            )

    def test_evaluate_dataset(self, client: JudgmentClient):
        """Test dataset evaluation."""
        example1 = Example(
            input={
                "question": "What if these shoes don't fit?",
                "category": "returns",
                "product": "shoes"
            },
            actual_output={
                "response": "We offer a 30-day full refund at no extra cost.",
                "policy": "30-day refund",
                "conditions": "no extra cost"
            },
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        )

        example2 = Example(
            input={
                "question": "How do I reset my password?",
                "category": "account_management",
                "feature": "password_reset"
            },
            actual_output={
                "instructions": "You can reset your password by clicking on 'Forgot Password' at the login screen.",
                "steps": [
                    "Go to login screen",
                    "Click 'Forgot Password'",
                    "Follow instructions"
                ]
            },
            expected_output={
                "instructions": "You can reset your password by clicking on 'Forgot Password' at the login screen.",
                "steps": [
                    "Go to login screen",
                    "Click 'Forgot Password'",
                    "Follow instructions"
                ]
            },
            name="Password Reset",
            context=["User Account"],
            retrieval_context=["Password reset instructions"],
            tools_called=["authentication"],
            expected_tools=["authentication"],
            additional_metadata={
                "difficulty": "medium",
                "category": "security"
            }
        )

        dataset = EvalDataset(examples=[example1, example2])
        res = client.evaluate_dataset(
            dataset=dataset,
            scorers=[FaithfulnessScorer(threshold=0.5)],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            metadata={"batch": "test"},
            project_name="test_project",
            eval_run_name="test_eval_run"
        )
        assert res, "Dataset evaluation failed"

    def test_override_eval(self, client: JudgmentClient, random_name: str):
        """Test evaluation override behavior."""
        example1 = Example(
            input={
                "question": "What if these shoes don't fit?",
                "category": "returns",
                "product": "shoes"
            },
            actual_output={
                "response": "We offer a 30-day full refund at no extra cost.",
                "policy": "30-day refund",
                "conditions": "no extra cost"
            },
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        )
        
        scorer = FaithfulnessScorer(threshold=0.5)

        PROJECT_NAME = "test_eval_run_naming_collisions"
        EVAL_RUN_NAME = random_name

        # First run should succeed
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
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
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
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
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
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
        with pytest.raises(ValueError, match="already exists"):
            client.run_evaluation(
                examples=[example1],
                scorers=[scorer],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                metadata={"batch": "test"},
                project_name=PROJECT_NAME,
                eval_run_name=EVAL_RUN_NAME,
                log_results=True,
                override=False,
            )

@pytest.mark.advanced
class TestAdvancedEvalOperations:
    def test_json_scorer(self, client: JudgmentClient):
        """Test JSON scorer functionality."""
        example1 = Example(
            input={
                "question": "What if these shoes don't fit?",
                "category": "returns",
                "product": "shoes"
            },
            actual_output={
                "tool": "authentication",
                "action": "process_return",
                "parameters": {
                    "product_type": "shoes",
                    "return_policy": "30-day refund"
                }
            },
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        )

        example2 = Example(
            input={
                "question": "How do I reset my password?",
                "category": "account_management",
                "feature": "password_reset"
            },
            actual_output={
                "instructions": "You can reset your password by clicking on 'Forgot Password' at the login screen.",
                "steps": [
                    "Go to login screen",
                    "Click 'Forgot Password'",
                    "Follow instructions"
                ]
            },
            expected_output={
                "instructions": "You can reset your password by clicking on 'Forgot Password' at the login screen.",
                "steps": [
                    "Go to login screen",
                    "Click 'Forgot Password'",
                    "Follow instructions"
                ]
            },
            name="Password Reset",
            context=["User Account"],
            retrieval_context=["Password reset instructions"],
            tools_called=["authentication"],
            expected_tools=["authentication"],
            additional_metadata={
                "difficulty": "medium",
                "category": "security",
                "priority": "high"
            }
        )

        class SampleSchema(BaseModel):
            tool: str

        scorer = JSONCorrectnessScorer(threshold=0.5, json_schema=SampleSchema)
        PROJECT_NAME = "test_project"
        EVAL_RUN_NAME = "test_json_scorer"
        
        res = client.run_evaluation(
            examples=[example1, example2],
            scorers=[scorer],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            metadata={"batch": "test"},
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            log_results=True,
            override=True,
            use_judgment=True,
        )
        assert res, "JSON scorer evaluation failed"

    def test_classifier_scorer(self, client: JudgmentClient, random_name: str):
        """Test classifier scorer functionality."""
        random_slug = random_name
        faithfulness_scorer = FaithfulnessScorer(threshold=0.5)
        
        # Creating a classifier scorer from SDK
        classifier_scorer_custom = ClassifierScorer(
            name="Test Classifier Scorer",
            slug=random_slug,
            threshold=0.5,
            conversation=[],
            options={}
        )
        
        classifier_scorer_custom.update_conversation(conversation=[{"role": "user", "content": "What is the capital of France?"}])
        classifier_scorer_custom.update_options(options={"yes": 1, "no": 0})
        
        slug = client.push_classifier_scorer(scorer=classifier_scorer_custom)
        
        classifier_scorer_custom = client.fetch_classifier_scorer(slug=slug)
        
        example1 = Example(
            input={
                "question": "What is the capital of France?",
                "category": "geography",
                "type": "capital_city",
                "country": "France"
            },
            actual_output={
                "answer": "Paris",
                "details": {
                    "country": "France",
                    "continent": "Europe",
                    "population": "2.1 million (2023)"
                }
            },
            retrieval_context=["The capital of France is Paris."]
        )

        res = client.run_evaluation(
            examples=[example1],
            scorers=[faithfulness_scorer, classifier_scorer_custom],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            log_results=True,
            eval_run_name="ToneScorerTest",
            project_name="ToneScorerTest",
            override=True,
        )
        assert res, "Classifier scorer evaluation failed" 