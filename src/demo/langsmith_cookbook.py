from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from openai import OpenAI
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
judgment_client = JudgmentClient()

client = Client()
openai_client = wrappers.wrap_openai(OpenAI())


# For other dataset creation methods, see: https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application

# Create inputs and reference outputs
# examples = [
#   (
#       "Which country is Mount Kilimanjaro located in?",
#       "Mount Kilimanjaro is located in Tanzania.",
#   ),
#   (
#       "What is Earth's lowest point?",
#       "Earth's lowest point is The Dead Sea.",
#   ),
# ]

# inputs = [{"input": input_prompt} for input_prompt, _ in examples]
# outputs = [{"expected": output_answer} for _, output_answer in examples]

# # Programmatically create a dataset in LangSmith
# dataset = client.create_dataset(
# 	dataset_name = "Kilimanjaro2",
# 	description = "A sample dataset in LangSmith."
# )

# # Add examples to the dataset
# client.create_examples(
#     inputs=inputs,
#     outputs=outputs,
#     dataset_id=dataset.id
# )

def target(inputs: dict) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            { "role": "system", "content": "Answer the following question accurately" },
            { "role": "user", "content": inputs["input"] },
        ]
    )
    return { "response": response.choices[0].message.content.strip() }
# Define instructions for the LLM judge evaluator
instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
- False: No conceptual match and similarity
- True: Most or full conceptual match and similarity
- Key criteria: Concept should match, not exact wording.
"""

# Define output schema for the LLM judge

def use_judgment_scorer(run, example):
    input_str = example.inputs.get("input")
    output_str = run.outputs.get("response") or run.outputs.get("output") 
    expected_str = example.outputs.get("expected") if example.outputs else None

    judgeval_example = Example(
        input=input_str,
        expected_output=expected_str,
        actual_output=output_str 
        )

    scorer = AnswerRelevancyScorer(threshold=0.5)
    results = judgment_client.run_evaluation(
        project_name="LangSmith-Cookbook-Eval",
        eval_run_name="Answer Relevancy - LangSmith",
        examples=[judgeval_example], 
        scorers=[scorer],
        override=True,
        model="gpt-4o",
    )
    score = results[0].scorers_data[0].score if results and results[0].scorers_data else None
    
    if score is None:
        print("Warning: Could not extract score from Judgment evaluation results.")
        # Return a default score or raise an error, depending on desired behavior
        return {"key": "judgment_answer_relevancy", "score": 0.0} 
        
    return {"key": "judgment_answer_relevancy", "score": score}

# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data = "Kilimanjaro2",
    evaluators = [
        use_judgment_scorer
    ],
    experiment_prefix = "first-eval-in-langsmith",
    max_concurrency = 2,
)