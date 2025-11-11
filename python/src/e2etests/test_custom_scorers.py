from judgeval.scorers.example_scorer import ExampleScorer
from judgeval import JudgmentClient
from judgeval.data import Example
from typing import Dict, Any, List


def test_basic_custom_scorer(
    client: JudgmentClient, project_name: str, random_name: str
):
    class HappinessScorer(ExampleScorer):
        additional_metadata: Dict[str, Any] = {}

        async def a_score_example(self, example):
            score = 0
            if "happy" in example.actual_output:
                score = 1
            elif "sad" in example.actual_output:
                score = 0
            else:
                score = 0.5
            return score

    class CustomExample(Example):
        actual_output: str

    examples: List[CustomExample] = [
        CustomExample(actual_output="I'm happy"),
        CustomExample(actual_output="I'm sad"),
        CustomExample(actual_output="I dont know"),
    ]

    scorer = HappinessScorer(
        name="Happiness Scorer", model="gpt-4o-mini", threshold=0.2
    )
    scorer2 = HappinessScorer(
        name="Stricter Happiness Scorer", model="gpt-4o-mini", threshold=0.8
    )
    res = client.run_evaluation(
        examples=examples,
        scorers=[scorer, scorer2],
        project_name=project_name,
        eval_run_name=random_name,
    )
    assert res[0].success
    assert not res[1].success
    assert not res[2].success

    scorer_data = res[0].scorers_data
    assert len(scorer_data) == 2
    assert scorer_data[0].name == "Happiness Scorer"
    assert scorer_data[1].name == "Stricter Happiness Scorer"
    assert scorer_data[0].success
    assert scorer_data[1].success

    scorer_data_2 = res[1].scorers_data
    assert not scorer_data_2[0].success
    assert not scorer_data_2[1].success

    scorer_data_3 = res[2].scorers_data
    assert scorer_data_3[0].success
    assert not scorer_data_3[1].success
