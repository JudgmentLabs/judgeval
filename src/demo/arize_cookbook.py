from arize.experimental.datasets import ArizeDatasetsClient
import json
from arize.experimental.datasets.experiments.types import EvaluationResult

client = ArizeDatasetsClient(api_key=os.environ["ARIZE_API_KEY"], developer_key=os.environ["ARIZE_DEVELOPER_KEY"])


from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
judgment_client = JudgmentClient()


from arize.experimental.datasets import ArizeDatasetsClient
from arize.experimental.datasets.utils.constants import GENERATIVE
import pandas as pd
from typing import Dict

def task(dataset_row: Dict):
    return dataset_row

def use_judgment_scorer(input, output, experiment_output):
    print(f"input: {input}  \noutput: {output}\nexperiment_output: {experiment_output}")
    # Parse the input/output/experiment_output data from Arize's Datasets
    input_data = json.loads(input)
    input_str = input_data['messages'][0]['content']

    actual_output_messages = json.loads(output['attributes.llm.output_messages'])
    actual_output_str = actual_output_messages[0]['message.content']

    expected_output_messages = json.loads(experiment_output['attributes.llm.output_messages'])
    expected_output_str = expected_output_messages[0]['message.content']

    # Creates a judgeval example which requires input, actual_output, and expected_output as strings
    example = Example(  
        input=input_str,
        actual_output=actual_output_str,
        expected_output=expected_output_str
        )

    scorer = AnswerRelevancyScorer(threshold=0.5)
    results = judgment_client.run_evaluation(
        project_name="Arize-Cookbook-Eval",
        eval_run_name="Answer Relevancy",
        examples=[example],
        scorers=[scorer],
        override=True,
        model="gpt-4o",
    )
    scorers_data = results[0].scorers_data[0]
    score = scorers_data.score
    success = scorers_data.success
    label = "Pass" if success else "Fail"
    explanation = scorers_data.reason

    return EvaluationResult(score=score, label=label, explanation=explanation)


client.run_experiment(
    space_id="U3BhY2U6MTg2MTI6aitHYQ==",
    dataset_name="eval", 
    task=task, 
    evaluators=[use_judgment_scorer], 
    experiment_name="evals27", 
)