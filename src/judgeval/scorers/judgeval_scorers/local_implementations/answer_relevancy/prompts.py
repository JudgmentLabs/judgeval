"""
Util prompts for AnswerRelevancyScorer
"""

from typing import List, Tuple
from pydantic import BaseModel


# BaseModels to enforce formatting in LLM JSON response
class Statements(BaseModel):
    statements: List[str]


class ARVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ARVerdict]


class Reason(BaseModel):
    reason: str


class AnswerRelevancyTemplate:
    @staticmethod
    def deduce_statements(actual_output):
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
{actual_output}
==== END OF INPUT ====

==== YOUR ANSWER ====
JSON:
"""

    @staticmethod
    def generate_verdicts(input, actual_output):
        return f"""You will be provided with a list of statements from a response; your task is to determine whether each statement is relevant with respect to a provided input.
More specifically, you should generate a JSON object with the key "verdicts". "verdicts" will map to a list of nested JSON objects with two keys: `verdict` and `reason`.
The "verdict" key be ONE OF THE FOLLOWING: ["yes", "no", "idk"]. You should select "yes" if the statement is relevant to addressing the original input, "no" if the statement is irrelevant, and 'idk' if it is ambiguous (eg., not directly relevant but could be used as a supporting point to address the input).
The "reason" key should provide an explanation for your choice, regardless of which verdict you select.

NOTE: the list of statements was generated from an output corresponding to the provided `input`. Account for this relationship during your evaluation of the content relevancy.

==== OUTPUT FORMATTING ====
IMPORTANT: Please make sure to only return in JSON format, with the "verdicts" key mapping to a list of JSON objects. Each JSON object should contain keys "verdict" (one of ["yes", "no", "idk"]) and "reason" (str). 

==== START OF EXAMPLES ====
Example input 1: How do I make chocolate chip cookies?
Example statements 1: ["Preheat the oven to 375°F.", "I love baking!", "My grandmother had a cat.", "Mix the butter and sugar until creamy.", "Have a great day!"]
Example JSON 1:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "Preheating the oven is a crucial first step in baking cookies"
        }},
        {{
            "verdict": "idk",
            "reason": "While showing enthusiasm for baking, this statement doesn't directly contribute to the recipe instructions"
        }},
        {{
            "verdict": "no",
            "reason": "The statement about the grandmother's cat is completely irrelevant to instructions for making chocolate chip cookies"
        }},
        {{
            "verdict": "yes",
            "reason": "Mixing butter and sugar is an essential step in cookie preparation"
        }},
        {{
            "verdict": "no",
            "reason": "A farewell message is not relevant to the cookie recipe instructions being requested"
        }}
    ]  
}}

Example input 2: What are the main causes of climate change?
Example statements 2: ["Greenhouse gas emissions trap heat in the atmosphere.", "I watched a movie yesterday.", "Industrial processes release large amounts of CO2.", "The weather is nice today."]
Example JSON 2:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "This directly explains a key mechanism of climate change"
        }},
        {{
            "verdict": "no",
            "reason": "Personal entertainment activities are not related to the causes of climate change"
        }},
        {{
            "verdict": "yes",
            "reason": "This identifies a major source of greenhouse gas emissions contributing to climate change"
        }},
        {{
            "verdict": "idk",
            "reason": "While weather is related to climate, a single day's weather observation doesn't directly address the causes of climate change"
        }}
    ]
}}
==== END OF EXAMPLES ====

** LASTLY **
Since you are tasked to choose a verdict for each statement, the number of "verdicts" SHOULD BE EXACTLY EQUAL to the number of "statements".


==== YOUR TURN =====

Input:
{input}

Statements:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(irrelevant_statements: List[Tuple[str, str]], input: str, score: float):
        irrelevant_statements = "\n".join([f"statement: {statement}\nreason: {reason}\n------" for statement, reason in irrelevant_statements])
        return f"""==== TASK INSTRUCTIONS ====\nYou will provided with three inputs: an answer relevancy score, a list of irrelevant statements made in a model's output (with the reason why it's irrelevant), and the corresponding input to the output. Your task is to provide a CLEAR and CONCISE reason for the answer relevancy score.
You should explain why the score is not higher, but also include why its current score is fair.
The irrelevant statements represent parts of the model output that are irrelevant to addressing whatever is asked/talked about in the input. The irrelevant statement will be paired with the reason why it's irrelevant.
If there are no irrelevant statements, instead respond with a positive remark with an upbeat encouraging tone (but don't overblow the kind attitude).


==== FORMATTING YOUR ANSWER ====
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <answer_relevancy_score> because <your_reason>."
}}

==== YOUR TURN ====
---- ANSWER RELEVANCY SCORE ----
{score}

---- IRRELEVANT STATEMENTS ----
{irrelevant_statements}

---- INPUT ----
{input}

---- YOUR RESPONSE ----
JSON:
"""

