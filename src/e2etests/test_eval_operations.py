"""
Tests for evaluation operations in the JudgmentClient.
"""

import pytest
import random
import string
import uuid
import os
import tempfile
import time

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer,
    ToolOrderScorer,
)
from judgeval.data.datasets.dataset import EvalDataset
from judgeval.tracer import Tracer
from judgeval.utils.file_utils import get_examples_from_yaml


@pytest.mark.basic
class TestEvalOperations:
    def run_eval_helper(
        self, client: JudgmentClient, project_name: str, eval_run_name: str
    ):
        """Helper function to run evaluation."""
        # Single step in our workflow, an outreach Sales Agent
        example1 = Example(
            input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
            actual_output="Dear Ms. Chen,\n\nI noticed TechCorp's recent launch of your AI analytics platform and was impressed by its enterprise-focused approach. Your experience from Google clearly shines through in building scalable solutions, as evidenced by your impressive 50+ enterprise client base.\n\nWould you be open to a brief call to discuss how we could potentially collaborate?\n\nBest regards,\nAlex",
            retrieval_context=[
                "TechCorp launched AI analytics platform in 2024",
                "Sarah Chen is CEO, ex-Google executive",
                "Current client base: 50+ enterprise customers",
            ],
        )

        example2 = Example(
            input="Generate a cold outreach email for GreenEnergy Solutions. Facts: They're developing solar panel technology that's 30% more efficient. They're looking to expand into the European market. They won a sustainability award in 2023.",
            actual_output="Dear GreenEnergy Solutions team,\n\nCongratulations on your 2023 sustainability award! Your innovative solar panel technology with 30% higher efficiency is exactly what the European market needs right now.\n\nI'd love to discuss how we could support your European expansion plans.\n\nBest regards,\nAlex",
            expected_output="A professional cold email mentioning the sustainability award, solar technology innovation, and European expansion plans",
            context=["Business Development"],
            retrieval_context=[
                "GreenEnergy Solutions won 2023 sustainability award",
                "New solar technology 30% more efficient",
                "Planning European market expansion",
            ],
        )

        scorer = FaithfulnessScorer(threshold=0.5)
        scorer2 = AnswerRelevancyScorer(threshold=0.5)

        client.run_evaluation(
            examples=[example1, example2],
            scorers=[scorer, scorer2],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            project_name=project_name,
            eval_run_name=eval_run_name,
            override=True,
        )

    def test_run_eval(self, client: JudgmentClient):
        """Test basic evaluation workflow."""
        PROJECT_NAME = "OutreachWorkflow"
        EVAL_RUN_NAME = "ColdEmailGenerator-Improve-BasePrompt"

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        results = client.pull_eval(
            project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME
        )
        assert results, f"No evaluation results found for {EVAL_RUN_NAME}"

        client.delete_project(project_name=PROJECT_NAME)

    def test_run_eval_append(self, client: JudgmentClient):
        """Test evaluation append behavior."""
        PROJECT_NAME = "OutreachWorkflow"
        EVAL_RUN_NAME = "ColdEmailGenerator-Improve-BasePrompt"

        self.run_eval_helper(client, PROJECT_NAME, EVAL_RUN_NAME)
        results = client.pull_eval(
            project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME
        )
        results = results["examples"]
        assert results, f"No evaluation results found for {EVAL_RUN_NAME}"
        assert len(results) == 2

        example1 = Example(
            input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
            actual_output="Dear Ms. Chen,\n\nI noticed TechCorp's recent launch of your AI analytics platform and was impressed by its enterprise-focused approach. Your experience from Google clearly shines through in building scalable solutions, as evidenced by your impressive 50+ enterprise client base.\n\nWould you be open to a brief call to discuss how we could potentially collaborate?\n\nBest regards,\nAlex",
            retrieval_context=[
                "TechCorp launched AI analytics platform in 2024",
                "Sarah Chen is CEO, ex-Google executive",
                "Current client base: 50+ enterprise customers",
            ],
        )
        scorer = FaithfulnessScorer(threshold=0.5)

        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            append=True,
        )

        results = client.pull_eval(
            project_name=PROJECT_NAME, eval_run_name=EVAL_RUN_NAME
        )
        assert results, f"No evaluation results found for {EVAL_RUN_NAME}"
        results = results["examples"]
        assert len(results) == 3
        assert results[0]["scorer_data"][0]["score"] == 1.0
        client.delete_project(project_name=PROJECT_NAME)

    def test_delete_eval_by_project(self, client: JudgmentClient):
        """Test delete evaluation by project and run name workflow."""
        PROJECT_NAME = "".join(
            random.choices(string.ascii_letters + string.digits, k=20)
        )
        EVAL_RUN_NAMES = [
            "".join(random.choices(string.ascii_letters + string.digits, k=20))
            for _ in range(3)
        ]

        for eval_run_name in EVAL_RUN_NAMES:
            self.run_eval_helper(client, PROJECT_NAME, eval_run_name)

        client.delete_project(project_name=PROJECT_NAME)
        for eval_run_name in EVAL_RUN_NAMES:
            with pytest.raises(ValueError, match="Error fetching eval results"):
                client.pull_eval(project_name=PROJECT_NAME, eval_run_name=eval_run_name)

    @pytest.mark.asyncio
    async def test_assert_test(self, client: JudgmentClient, project_name: str):
        """Test assertion functionality."""
        # Create examples and scorers as before
        example = Example(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        )

        example1 = Example(
            input="How much are your croissants?",
            actual_output="Sorry, we don't accept electronic returns.",
        )

        example2 = Example(
            input="Who is the best basketball player in the world?",
            actual_output="No, the room is too small.",
        )

        scorer = AnswerRelevancyScorer(threshold=0.5)

        with pytest.raises(AssertionError):
            await client.assert_test(
                eval_run_name="test_eval",
                project_name=project_name,
                examples=[example, example1, example2],
                scorers=[scorer],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                override=True,
            )

    def test_evaluate_dataset(self, client: JudgmentClient, project_name: str):
        """Test dataset evaluation."""
        example1 = Example(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        )

        example2 = Example(
            input="How do I reset my password?",
            actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
            expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
            name="Password Reset",
            context=["User Account"],
            retrieval_context=["Password reset instructions"],
            additional_metadata={"difficulty": "medium"},
        )

        dataset = EvalDataset(examples=[example1, example2])
        res = client.run_evaluation(
            examples=dataset.examples,
            scorers=[FaithfulnessScorer(threshold=0.5)],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            project_name=project_name,
            eval_run_name="test_eval_run",
            override=True,
        )
        assert res, "Dataset evaluation failed"

    @pytest.mark.asyncio
    async def test_run_trace_eval_from_yaml(
        self, client: JudgmentClient, project_name: str, random_name: str
    ):
        """Test run_trace_evaluation with a YAML configuration file."""
        EVAL_RUN_NAME = random_name

        yaml_content = """
examples:
  - input: "hello from yaml"
    expected_tools:
      - tool_name: "simple_traced_function_for_yaml_eval"
        agent_name: "Agent 1"
        parameters:
          text: "hello from yaml"
    retrieval_context: ["Context for hello from yaml"]
  - input: "another yaml test"
    expected_tools:
      - tool_name: "simple_traced_function_for_yaml_eval"
        agent_name: "Agent 1"
    retrieval_context: ["Context for another yaml test"]
"""
        tracer = Tracer(project_name=project_name)

        @tracer.observe(span_type="tool")
        def simple_traced_function_for_yaml_eval(text: str):
            """A simple function to be traced and evaluated from YAML."""
            time.sleep(0.01)  # Simulate minimal sync work
            return f"Processed: {text.upper()}"

        temp_yaml_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".yaml"
            ) as tmpfile:
                tmpfile.write(yaml_content)
                temp_yaml_file_path = tmpfile.name
            scorer = ToolOrderScorer(threshold=0.5)
            client.run_trace_evaluation(
                examples=get_examples_from_yaml(temp_yaml_file_path),
                function=simple_traced_function_for_yaml_eval,
                tracer=tracer,
                scorers=[scorer],
                project_name=project_name,
                eval_run_name=EVAL_RUN_NAME,
                override=True,
            )

            results = client.pull_eval(
                project_name=project_name, eval_run_name=EVAL_RUN_NAME
            )
            assert results, (
                f"No evaluation results found for {EVAL_RUN_NAME} in project {project_name}"
            )
            assert isinstance(results, list), (
                "Expected results to be a list of experiment/trace data"
            )
            assert len(results) == 2, f"Expected 2 trace results but got {len(results)}"

        finally:
            if temp_yaml_file_path and os.path.exists(temp_yaml_file_path):
                os.remove(temp_yaml_file_path)

    def test_override_eval(
        self, client: JudgmentClient, project_name: str, random_name: str
    ):
        """Test evaluation override behavior."""
        example1 = Example(
            input="What if these shoes don't fit?",
            actual_output="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        )

        scorer = FaithfulnessScorer(threshold=0.5)

        EVAL_RUN_NAME = random_name

        # First run should succeed
        client.run_evaluation(
            examples=[example1],
            scorers=[scorer],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            project_name=project_name,
            eval_run_name=EVAL_RUN_NAME,
            override=False,
        )

        # This should fail because the eval run name already exists
        with pytest.raises(ValueError, match="already exists"):
            client.run_evaluation(
                examples=[example1],
                scorers=[scorer],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                project_name=project_name,
                eval_run_name=EVAL_RUN_NAME,
                override=False,
            )

        # Third run with override=True should succeed
        try:
            example1.example_id = str(uuid.uuid4())
            client.run_evaluation(
                examples=[example1],
                scorers=[scorer],
                model="Qwen/Qwen2.5-72B-Instruct-Turbo",
                project_name=project_name,
                eval_run_name=EVAL_RUN_NAME,
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
                project_name=project_name,
                eval_run_name=EVAL_RUN_NAME,
                override=False,
            )
