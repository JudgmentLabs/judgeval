import pytest

from judgeval.v1 import Judgeval
from judgeval.v1.data import Example
from judgeval.v1.scorers.built_in import (
    FaithfulnessScorer,
    AnswerRelevancyScorer,
)


def run_eval_helper(client: Judgeval, eval_run_name: str):
    example1 = Example.create(
        input="Generate a cold outreach email for TechCorp. Facts: They recently launched an AI-powered analytics platform. Their CEO Sarah Chen previously worked at Google. They have 50+ enterprise clients.",
        actual_output="Dear Ms. Chen,\n\nI noticed TechCorp's recent launch of your AI analytics platform and was impressed by its enterprise-focused approach. Your experience from Google clearly shines through in building scalable solutions, as evidenced by your impressive 50+ enterprise client base.\n\nWould you be open to a brief call to discuss how we could potentially collaborate?\n\nBest regards,\nAlex",
        retrieval_context=[
            "TechCorp launched AI analytics platform in 2024",
            "Sarah Chen is CEO, ex-Google executive",
            "Current client base: 50+ enterprise customers",
        ],
    )

    example2 = Example.create(
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

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[example1, example2],
        scorers=[scorer, scorer2],
        eval_run_name=eval_run_name,
    )
    return res


def test_basic_eval(client: Judgeval, random_name: str):
    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=[
            Example.create(
                input="What's the capital of France?",
                actual_output="The capital of France is Paris.",
            )
        ],
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        eval_run_name=random_name,
    )

    assert res, "No evaluation results found"


def test_run_eval(client: Judgeval, random_name: str):
    res = run_eval_helper(client, random_name)
    assert res, f"No evaluation results found for {random_name}"

    res2 = run_eval_helper(client, random_name)
    assert res2, f"No evaluation results found for {random_name}"


def test_assert_test(client: Judgeval):
    example = Example.create(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )

    example1 = Example.create(
        input="How much are your croissants?",
        actual_output="Sorry, we don't accept electronic returns.",
    )

    example2 = Example.create(
        input="Who is the best basketball player in the world?",
        actual_output="No, the room is too small.",
    )

    scorer = AnswerRelevancyScorer(threshold=0.5)

    evaluation = client.evaluation.create()
    with pytest.raises(AssertionError):
        evaluation.run(
            eval_run_name="test_eval",
            examples=[example, example1, example2],
            scorers=[scorer],
            assert_test=True,
        )


def test_evaluate_dataset(client: Judgeval, random_name: str):
    example1 = Example.create(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."
        ],
    )
    example2 = Example.create(
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        retrieval_context=["Password reset instructions"],
    )

    client.datasets.create(name=random_name, examples=[example1, example2])
    dataset = client.datasets.get(name=random_name)
    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=list(dataset),
        scorers=[FaithfulnessScorer(threshold=0.5)],
        eval_run_name=random_name,
    )
    assert res, "Dataset evaluation failed"


def test_dataset_and_evaluation(client: Judgeval, random_name: str):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
    ]
    client.datasets.create(name=random_name, examples=examples)
    dataset = client.datasets.get(name=random_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset) == 2, "Dataset should have 2 examples"

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=examples,
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        eval_run_name=random_name,
    )
    assert res, "Dataset evaluation failed"


def test_dataset_and_double_evaluation(client: Judgeval, random_name: str):
    examples = [
        Example.create(input="input 1", actual_output="output 1"),
        Example.create(input="input 2", actual_output="output 2"),
    ]
    client.datasets.create(name=random_name, examples=examples)
    dataset = client.datasets.get(name=random_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset) == 2, "Dataset should have 2 examples"

    evaluation = client.evaluation.create()
    res = evaluation.run(
        examples=list(dataset),
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        eval_run_name=random_name,
    )
    assert res, "Dataset evaluation failed"

    res2 = evaluation.run(
        examples=list(dataset),
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        eval_run_name=random_name,
    )
    assert res2, "Dataset evaluation failed"
