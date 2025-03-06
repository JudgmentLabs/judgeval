from typing import List
from pydantic import BaseModel


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class Reason(BaseModel):
    reason: str


class ContextualPrecisionTemplate:
    @staticmethod
    def generate_verdicts(input, expected_output, retrieval_context):
        return f"""==== TASK INSTRUCTIONS ====\nGiven the input, expected output, and retrieval context, your task is to determine whether each document in the retrieval context was relevant to arrive at the expected output.
You should reason through the documents in the retrieval context thoroughly, and then generate a list of JSON objects representing your decision.

==== FORMAT INSTRUCTIONS ====\nIMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON. These JSON only contain the `verdict` key that outputs only 'yes' or 'no', and a `reason` key to justify the verdict. In your reason, aim to quote parts of the context to support your verdict.

==== EXAMPLE ====
Example Input: "What are the main symptoms of COVID-19?"
Example Expected Output: "The main symptoms of COVID-19 include fever, cough, fatigue, and loss of taste or smell."
Example Retrieval Context: ["Common COVID-19 symptoms include fever and dry cough", "Loss of taste and smell are distinctive COVID-19 symptoms", "The first COVID-19 case was reported in Wuhan", "My friend's birthday party was fun last weekend"]

Example output JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes", 
            "reason": "The text directly lists key COVID-19 symptoms including 'fever and dry cough' which are part of the main symptoms."
        }},
        {{
            "verdict": "yes",
            "reason": "The text mentions 'loss of taste and smell' which are distinctive symptoms of COVID-19 that should be included."
        }},
        {{
            "verdict": "no",
            "reason": "While related to COVID-19, the origin of the first case is not relevant to listing the main symptoms."
        }},
        {{
            "verdict": "no",
            "reason": "A personal anecdote about a birthday party has no relevance to COVID-19 symptoms."
        }}
    ]
}}

Your task is to generate a verdict for each document in the retrieval context, so the number of 'verdicts' SHOULD BE EXACTLY EQUAL to that of the retrievalcontexts.

==== YOUR TURN ====
Input:
{input}

Expected output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""

    @staticmethod
    def generate_reason(input, verdicts, score):
        return f"""==== TASK INSTRUCTIONS ====\nYou will be provided with an input, retrieval contexts, and a contextual precision score. Your task is to provide a CLEAR and CONCISE reason for the score. 
You should explain why the score is not higher, but also the current score is reasonable. Here's a further breakdown of the task:

1. input (str) is a task or question that the model attempted to solve
2. retrieval contexts (list[dict]) is a list of JSON with the following keys:
- `verdict` (str): either 'yes' or 'no', which represents whether the corresponding document in the retrieval context is relevant to the input.
- `reason` (str): a reason for the verdict.
3. The contextual precision score is a float between 0 and 1 and represents if the relevant documents are ranked higher than irrelevant ones in the retrieval context. 
The ranking can be inferred by the order of the retrieval documents: retrieval contexts is given IN THE ORDER OF THE DOCUMENT RANKINGS.
This implies that the score will be higher if the relevant documents are ranked higher (appears earlier in the list) than irrelevant ones.

==== FORMAT INSTRUCTIONS ====\nIMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason for the contextual precision score.
Example JSON:
{{
    "reason": "The score is <contextual_precision_score> because <your_reason>."
}}


DO NOT mention 'verdict' in your reason, but instead phrase it as irrelevant nodes. The term 'verdict' are just here for you to understand the broader scope of things.
Also DO NOT mention there are `reason` fields in the retrieval contexts you are presented with, instead just use the information in the `reason` field.
In your reason, you MUST USE the `reason`, QUOTES in the 'reason', and the node RANK (starting from 1, eg. first node) to explain why the 'no' verdicts should be ranked lower than the 'yes' verdicts.
When addressing nodes, make it explicit that it is nodes in retrieval context.
If the score is 1, keep it short and say something positive with an upbeat tone (but don't overdo it otherwise it gets annoying).

==== YOUR TURN ====
Contextual Precision Score:
{score}

Input:
{input}

Retrieval Contexts:
{verdicts}

JSON:
"""

