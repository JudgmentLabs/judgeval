from typing import List
from pydantic import BaseModel


class ContextualRelevancyVerdict(BaseModel):
    statement: str
    verdict: str
    reason: str


class ContextualRelevancyVerdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str


class ContextualRelevancyTemplate:

    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with an input (str) and a context (str). The input is a question/task proposed to a language model and the context is a list of documents retrieved in a RAG pipeline.
Your task is to determine whether each statement found in the context is relevant to the input. To do so, break down the context into statements (high level pieces of information), then determine whether each statement is relevant to the input.

==== FORMATTING YOUR ANSWER ====

You should format your answer as a list of JSON objects, with each JSON object containing the following fields:
- 'verdict': a string that is EXACTLY EITHER 'yes' or 'no', indicating whether the statement is relevant to the input
- 'statement': a string that is the statement found in the context
- 'reason': an string that is the justification for why the statement is relevant to the input. IF your verdict is 'no', you MUST quote the irrelevant parts of the statement to back up your reason.

IMPORTANT: Please make sure to only return in JSON format.

==== EXAMPLE ====
Example Context: "Einstein won the Nobel Prize for his discovery of the photoelectric effect. He won the Nobel Prize in 1968. There was a cat."
Example Input: "What were some of Einstein's achievements?"

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "statement": "Einstein won the Nobel Prize for his discovery of the photoelectric effect in 1968",
        }},
        {{
            "verdict": "no",
            "statement": "There was a cat.",
            "reason": "The retrieval context contained the information 'There was a cat' when it has nothing to do with Einstein's achievements."
        }}
    ]
}}

==== YOUR TURN ====

Input:
{input}

Context:
{context}

JSON:
"""
    
    @staticmethod
    def generate_reason(
        input: str,
        irrelevancies: List[str],
        relevant_statements: List[str],
        score: float,
    ):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with the following information:
- An input to a RAG pipeline which is a question/task. There is an associated retrieval context to this input in the RAG pipeline (the context is not provided but is relevant to your task).
- A list of irrelevant statements from the retrieval context. These statements are not relevant to the input query.
- A list of relevant statements from the retrieval context. These statements are relevant to the input query.
- A contextual relevancy score (the closer to 1 the better). Contextual relevancy is a measurement of how relevant the retrieval context is to the input query.

Your task is to generate a CLEAR and CONCISE reason for the score. You should quote data provided in the reasons for the irrelevant and relevant statements to support your reason.

==== FORMATTING YOUR ANSWER ====
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_relevancy_score> because <your_reason>."
}}

If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

==== EXAMPLE ====
Input: "What is the capital of France?"

Contextual Relevancy Score: 0.67

Irrelevant Statements from the retrieval context:
[{{"statement": "Flights to Paris are available from San Francisco starting at $1000", "reason": "Flight prices and routes are not relevant to identifying the capital of France"}}]

Relevant Statements from the retrieval context:
[{{"statement": "Paris is the capital of France"}}, {{"statement": "Paris is a major European city"}}]

Example Response:
{{
    "reason": "The score is 0.67 because while the context contains directly relevant information stating that 'Paris is the capital of France', it also includes irrelevant travel information about flight prices from San Francisco."
}}

==== YOUR TURN ====
Contextual Relevancy Score:
{score}

Input:
{input}

Irrelevant Statements from the retrieval context:
{irrelevancies}

Relevant Statements from the retrieval context:
{relevant_statements}

JSON:
"""
