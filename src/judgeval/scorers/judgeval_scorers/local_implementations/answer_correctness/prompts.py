"""
Util prompts for AnswerCorrectnessScorer
"""

from typing import List, Tuple
from pydantic import BaseModel


# BaseModels to enforce formatting in LLM JSON response
class Statements(BaseModel):
    statements: List[str]


class ACVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ACVerdict]


class Reason(BaseModel):
    reason: str


class AnswerCorrectnessTemplate:
    @staticmethod
    def deduce_statements(expected_output):
        return f"""You will be presented with a piece of text. Your task is to break down the text and generate a list of statements contained within the text. Single words and ambiguous phrases should be considered statements.

===== START OF EXAMPLES =====
Example 1:
Example text: The weather is sunny today. Temperature is 75 degrees. Don't forget your sunscreen!

Output:
{{
    "statements": ["The weather is sunny today", "Temperature is 75 degrees", "Don't forget your sunscreen!"]
}}

Example 2:
Example text: I love pizza. It has cheese and tomato sauce and the crust is crispy.

Output:
{{
    "statements": ["I love pizza", "It has cheese and tomato sauce", "The crust is crispy"]
}}
===== END OF EXAMPLES =====

        
**
IMPORTANT: Please return your answer in valid JSON format, with the "statements" key mapping to a list of strings. No words or explanation is needed.
**

==== START OF INPUT ====
Text:
{expected_output}
==== END OF INPUT ====

==== YOUR ANSWER ====
JSON:
"""

    @staticmethod
    def generate_verdicts(statements, actual_output):
        return f"""You will be provided with:\n
- a list of statements from a text that we will refer to as expected output
- a text that we will refer to as actual output\n

Your task is to determine whether each statement from the expected output is correct/consistent with the actual output text.
More specifically, you should generate a JSON object with the key "verdicts". "verdicts" will map to a list of nested JSON objects with two keys: `verdict` and `reason`.
The "reason" key should provide an explanation for your choice, regardless of which verdict you select. Try providing quotes from the text(s) to justify your answer where possible.
The "verdict" key be EXACTLY EITHER "yes" or "no". You should select "yes" if the statement is correct/consistent based on the actual output and "no" otherwise.

==== OUTPUT FORMATTING ====
IMPORTANT: Please make sure to only return in JSON format, with the "verdicts" key mapping to a list of JSON objects. Each JSON object should contain keys "verdict" (one of "yes" or "no") and "reason" (str). 

==== START OF EXAMPLES ====
Example input 1: What's the capital of France?
Example expected output statements 1: ["Paris is the capital of France", "It is located in northern France", "The city has a population of over 2 million"]
Example actual output 1: "Paris is the capital city of France. It is situated in the northern part of the country and has over 2 million residents."
Example JSON 1:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The actual output directly states 'Paris is the capital city of France', which matches the statement"
        }},
        {{
            "verdict": "yes", 
            "reason": "The actual output confirms this by saying it is 'situated in the northern part of the country'"
        }},
        {{
            "verdict": "yes",
            "reason": "The actual output mentions the city 'has over 2 million residents', matching the population statement"
        }}
    ]
}}

Example input 2: What is the largest planet in our solar system?
Example expected output statements 2: ["Jupiter is the largest planet", "It is a gas giant", "Jupiter has 79 known moons", "The Great Red Spot is a storm on Jupiter"]
Example actual output 2: "Jupiter is the biggest planet in the solar system. It is made mostly of gas. The planet has many moons orbiting it."
Example JSON 2:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The actual output confirms 'Jupiter is the biggest planet', which is equivalent to it being the largest"
        }},
        {{
            "verdict": "yes",
            "reason": "The actual output states it is 'made mostly of gas', indicating it is a gas giant"
        }},
        {{
            "verdict": "no",
            "reason": "While the actual output mentions Jupiter has 'many moons', it does not specify the exact number of 79 known moons"
        }},
        {{
            "verdict": "no",
            "reason": "The actual output makes no mention of the Great Red Spot or any storms on Jupiter"
        }}
    ]
}}
==== END OF EXAMPLES ====

** LASTLY **
Since you are tasked to choose a verdict for each statement, the number of "verdicts" SHOULD BE EXACTLY EQUAL to the number of "statements".


==== YOUR TURN =====

Statements:
{statements}

Actual output:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(incorrect_statements: List[Tuple[str, str]], score: float):
        incorrect_statements = "\n".join([f"statement: {statement}\nreason: {reason}\n------" for statement, reason in incorrect_statements])
        return f"""==== TASK INSTRUCTIONS ====\nYou will provided with two inputs: an answer correctness score and a list of inconsistent/incorrect statements made in a model's output (with the reason why it's irrelevant). Your task is to provide a CLEAR and CONCISE reason for the answer correctness score. 
For context, there were a list of statements generated from an expected output. The model's actual output was then compared to the expected output, and we collected a list of claims made in the expected output that were either incorrect or inconsistent with the actual output. 
The score represents how well the model's output matches the expected output.
You should explain why the score is not higher, but also include why its current score is fair.
The incorrect statements represent parts of the model output that are incorrect or inconsistent with the expected output. The incorrect statement will be paired with the reason why it's incorrect.
If there are no incorrect statements, instead respond with a positive remark with an upbeat encouraging tone (but don't overblow the kind attitude).


==== FORMATTING YOUR ANSWER ====
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <answer_relevancy_score> because <your_reason>."
}}

==== YOUR TURN ====
---- ANSWER CORRECTNESS SCORE ----
{score}

---- INCORRECT STATEMENTS ----
{incorrect_statements}

---- YOUR RESPONSE ----
JSON:
"""

