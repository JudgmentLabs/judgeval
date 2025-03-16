import os
import csv
from judgeval.data import Example
from judgeval import JudgmentClient
from judgeval.scorers import ClassifierScorer, ComparisonScorer
from openai import OpenAI

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = JudgmentClient()

def load_examples():
    """Load and parse the data from CSV file"""
    with open(os.path.join(os.path.dirname(__file__), "data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = list(reader)
    
    examples = []
    for row in data:
        id, draft_text, final_text = row
        example = Example(
            input=str(id),
            actual_output=str(draft_text),
            expected_output=str(final_text),
        )
        examples.append(example)
    return examples

def find_improvements(actual_output: str, criteria: str, description: str, count: int):

    # Creating a classifier scorer from SDK
    classifier_scorer_custom = ClassifierScorer(
        name="Refinement Scorer",
        slug="refinement-scorer",
        threshold=0.5,
        conversation=[],
        options={}
    )

    prompt = f"""==== TASK INSTRUCTIONS ====
You will be presented with a document, an actual_output. 
This actual_output is a rough draft of a document, and you are tasked with finding the improvements that can be made to the actual_output on the basis of {criteria}.

We define {criteria} to be the {description}.

I want you to first output each time you find an area/sentence that needs improvement and the reasoning for it. Then you should count the number of needs improvement areas and output the total number of needs improvement areas.
However, I want you to output that are significant.
==== YOUR TURN ====
Actual output:{actual_output}
Criteria: {criteria}
Definition of criteria: {description}
"""

    
    classifier_scorer_custom.update_conversation(conversation=[{"role": "user", "content": prompt}])
    classifier_scorer_custom.update_options(options={"2 or less occurrences of improvements": 1, "3 or more occurrences of improvements": 0})

    example = Example(
        actual_output=actual_output,
    )
    response = client.run_evaluation(
        examples=[example],
        scorers=[classifier_scorer_custom],
        model="gpt-4o",
        project_name="alma-demo",
        eval_run_name=f"alma-refinement-{count}",
        override=True
    )
    score, reason = response[0].scorers_data[0].score, response[0].scorers_data[0].reason
    return score, reason

def refine_draft(actual_output: str, improvements: str, criteria: str, description: str):
    prompt = f"""==== TASK INSTRUCTIONS ====
You will be presented with a document, an actual_output. 
This actual_output is a rough draft of a document, and you are tasked with implementing the improvements that can be made to the actual_output on the basis of the following criteria:
{criteria}
Definition of criteria: {description}

Here are the improvements that can be made to the actual_output:
{improvements}

==== YOUR TURN ====
Actual output:{actual_output}
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def run_judgment(example: Example, criteria: str, description: str):
    scorer = ComparisonScorer(threshold=0.5, criteria=criteria, description=description)
    output = client.run_evaluation(
        model="gpt-4o",
        examples=[example],
        scorers=[scorer],
        project_name="alma-demo",
        eval_run_name=f"alma-refinement-final", 
        override=True
    )

def main():
    examples = load_examples()
    example = examples[0]
    expected_output = example.expected_output
    example.expected_output = None
    criteria = "Impact and Evidence"
    description = "Emphasize significant achievements, leadership, and innovation with quantitative data and supporting evidence. Highlight influences on societal challenges and industry advancements to underscore impact."  
    
    score = 0
    count = 1
    current_output = example.actual_output
    success = False
    while score < 0.5 and count < 5:  
        score, reason = find_improvements(current_output, criteria, description, count)
        print(f"{count}th iteration: Score: {score}")
        if score > 0.5:
            success = True
            break
        current_output = refine_draft(current_output, reason, criteria, description)
        print(f"{count}th Refined Output")
        count += 1

    if success:
        print("Running Judgment...")
        example.actual_output = current_output
        example.expected_output = expected_output
        run_judgment(example, criteria, description)

if __name__ == "__main__":
    main()
