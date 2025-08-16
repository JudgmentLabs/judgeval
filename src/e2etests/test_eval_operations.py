"""
Tests for evaluation operations in the JudgmentClient.
"""

import pytest

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import (
    FaithfulnessScorer,
    AnswerRelevancyScorer,
)
from judgeval.scorers.example_scorer import ExampleScorer
from judgeval.dataset import Dataset
from judgeval.constants import DEFAULT_TOGETHER_MODEL


def run_eval_helper(client: JudgmentClient, project_name: str, eval_run_name: str):
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
        retrieval_context=[
            "GreenEnergy Solutions won 2023 sustainability award",
            "New solar technology 30% more efficient",
            "Planning European market expansion",
        ],
    )

    scorer = FaithfulnessScorer(threshold=0.5)
    scorer2 = AnswerRelevancyScorer(threshold=0.5)

    res = client.run_evaluation(
        examples=[example1, example2],
        scorers=[scorer, scorer2],
        model=DEFAULT_TOGETHER_MODEL,
        project_name=project_name,
        eval_run_name=eval_run_name,
    )
    return res


def test_run_eval(client: JudgmentClient, project_name: str, random_name: str):
    """Test basic evaluation workflow."""

    res = run_eval_helper(client, project_name, random_name)
    assert res, f"No evaluation results found for {random_name}"

    res2 = run_eval_helper(client, project_name, random_name)
    assert res2, f"No evaluation results found for {random_name}"


@pytest.mark.asyncio
async def test_assert_test(client: JudgmentClient, project_name: str):
    """Test assertion functionality."""
    # Create examples and scorers as before
    example = Example(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
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
            model=DEFAULT_TOGETHER_MODEL,
        )


def test_evaluate_dataset(client: JudgmentClient, project_name: str, random_name: str):
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
        retrieval_context=["Password reset instructions"],
    )

    Dataset.create(
        name=random_name, project_name=project_name, examples=[example1, example2]
    )
    dataset = Dataset.get(name=random_name, project_name=project_name)
    res = client.run_evaluation(
        examples=dataset.examples,
        scorers=[FaithfulnessScorer(threshold=0.5)],
        model=DEFAULT_TOGETHER_MODEL,
        project_name=project_name,
        eval_run_name=random_name,
    )
    assert res, "Dataset evaluation failed"

    dataset.delete()


def test_evaluate_dataset_custom(
    client: JudgmentClient, project_name: str, random_name: str
):
    """Test dataset evaluation with custom scorers."""

    class CustomExample(Example):
        unique_field: str
        unique_number: int

    class CustomScorer(ExampleScorer):
        async def a_score_example(self, example: CustomExample):
            if example.unique_field == "test":
                if example.unique_number == 1:
                    return 1
                elif example.unique_number == 2:
                    return 0.75
                else:
                    return 0.5
            else:
                return 0

    examples = [
        CustomExample(unique_field="test", unique_number=1),
        CustomExample(unique_field="test", unique_number=2),
        CustomExample(unique_field="test", unique_number=3),
        CustomExample(unique_field="not_test", unique_number=1),
    ]
    Dataset.create(name=random_name, project_name=project_name, examples=examples)
    dataset = Dataset.get(name=random_name, project_name=project_name)
    res = client.run_evaluation(
        examples=dataset.examples,
        scorers=[CustomScorer()],
        model=DEFAULT_TOGETHER_MODEL,
        project_name=project_name,
        eval_run_name=random_name,
    )

    assert res[0].success
    assert res[1].success
    assert res[2].success
    assert not res[3].success

    assert res[0].scorers_data[0].score == 1
    assert res[1].scorers_data[0].score == 0.75
    assert res[2].scorers_data[0].score == 0.5
    assert res[3].scorers_data[0].score == 0

    dataset.delete()
