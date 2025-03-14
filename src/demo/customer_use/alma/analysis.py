from dotenv import load_dotenv
import os
import csv
from judgeval.data import Example
from judgeval import JudgmentClient
from judgeval.scorers import ComparisonScorer
from typing import List
import openai
import json
from concurrent.futures import ThreadPoolExecutor



def load_examples():
    """Load and parse the data from CSV file"""
    with open(os.path.join(os.path.dirname(__file__), "data.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
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

def run_judgment_evaluation(examples: List[Example]):
    """
    Run evaluation using JudgmentClient
    
    Args:
        examples: List of Example objects
        
    Returns:
        List of boolean values indicating if the example is a false negative
    """
    client = JudgmentClient()
    scorer1 = ComparisonScorer(threshold=1.0, criteria="Clarity and Precision", description="Use clear, specific, and factual language to convey key points concisely. Avoid redundancy and unnecessary complexity to enhance understanding and ensure effective communication.")
    scorer2 = ComparisonScorer(threshold=1.0, criteria="Structure and Organization", description="Maintain a logical and well-organized flow with coherent transitions and sections. Present content systematically to enhance readability and coherence, effectively showcasing roles and achievements.")
    scorer3 = ComparisonScorer(threshold=1.0, criteria="Professional Tone and Context", description="Adopt a formal tone suitable for professional communication. Use industry-specific language to enhance credibility and professionalism, aligning with standards and expectations.")
    scorer4 = ComparisonScorer(threshold=1.0, criteria="Impact and Evidence", description="Emphasize significant achievements, leadership, and innovation with quantitative data and supporting evidence. Highlight influences on societal challenges and industry advancements to underscore impact.")
    scorer5 = ComparisonScorer(threshold=1.0, criteria="Engagement and Storytelling", description="Utilize storytelling techniques to craft a compelling narrative. Integrate personal and professional milestones to engage the audience, illustrating significant impact and sustained contributions.")

    for i in range(1):
        output = client.run_evaluation(
            model="osiris-mini",
            examples=examples,
            scorers=[scorer1, scorer2, scorer3, scorer4, scorer5],
            eval_run_name=f"alma-run2", 
            project_name="alma-demo"
        )
    return output

def find_categories(examples: List[Example], current_categories: List[dict] = []):
    """
    Find the categories of the examples in parallel.
    """
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    def process_example(example):
        prompt = f"""
        You will be provided with a rough draft and a final version of a draft, where the final version is considered superior. Your task is to:

        1. Identify the differences between the two drafts based on the existing list of criteria: {current_categories}.
        3. If you find that a difference does not fit within this list, then you can either add a new criteria or update that criteria and description.
        4. Similarly, if criteria are similar or can be generalized, combine them and update the description to reflect the combined criteria.
        5. A sanity check is that the total number of criteria should not exceed 8; if it does, you should combine criteria that are most similar, again updating the description.

        Generate a JSON array of objects with the following format:
        [
            {{
                "criteria": "Criteria Name",
                "description": "Generic description of the criteria",
            }},
            ...
        ]

        Your response should include:
        - A detailed explanation of your reasoning.
        - The JSON array as specified. Ensure it is JSON formatted.

        Here are the drafts:
        Rough Draft: {example.actual_output}
        Final Version: {example.expected_output}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content

        try:
            response = response[response.index('json') + len('json'):].strip()
            response = response[response.index('['):response.rindex(']') + 1]
        except Exception as e:
            print(f"Error indexing JSON response: {response}, skipping example {example.input}")
            return example.input

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response}, skipping example {example.input}")
            return example.input

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(process_example, examples))

    # Filter out None results and combine valid results
    for result in results:
        if result:
            current_categories.extend(result)
        else:
            print("===================")
            print(f"Skipping example {result}")
            print("===================")

    return current_categories

def find_categories_in_batches(examples: List[Example]):
    """
    Process examples in batches of three and combine results.
    """
    current_categories = []
    client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))

    for i in range(0, len(examples), 5):
        batch = examples[i:i + 5]
        batch_categories = find_categories(batch, current_categories)
        
        # Combine the results from the batch into a single prompt
        combined_prompt = f"""
        You have processed the following batches of examples:
        {batch_categories}

        Please combine the results into a single coherent list of categories, ensuring that similar criteria are merged and descriptions are updated accordingly. Keep the formatting:
        A sanity check is that the total number of criteria should not exceed 6; if it does, you should combine criteria that are most similar, again updating the description.
        Also criteria should not be overly complex. Things like "Use of Evidence" and "Tone and Clarity" are good, but something like "Tone, Professionalism, Clarity, and Precision" is not.
        Ensure it is JSON formatted.
        [
            {{
                "criteria": "Criteria Name",
                "description": "description of the criteria",
            }},
            ...
        ]
        """

        response = client.chat.completions.create(
            model="o1-mini",
            messages=[
                {"role": "user", "content": combined_prompt}
            ]
        ).choices[0].message.content

        try:
            response = response[response.index('json') + len('json'):].strip()
            response = response[response.index('['):response.rindex(']') + 1]
        except Exception as e:
            print(f"Error finding json for batch combination: {response}")
            continue

        try:
            current_categories = json.loads(response)
            print(f"Combined categories: {current_categories}")
            print()
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {response}")
            continue

    return current_categories

def main():
    load_dotenv()
    examples = load_examples()

    run_judgment_evaluation(examples)

if __name__ == "__main__":
    main()