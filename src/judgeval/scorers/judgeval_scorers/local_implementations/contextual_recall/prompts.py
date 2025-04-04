from typing import List
from pydantic import BaseModel


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class Reason(BaseModel):
    reason: str


class ContextualRecallTemplate:

    @staticmethod
    def generate_verdicts(expected_output, retrieval_context):
        return f"""
==== TASK INSTRUCTIONS ====
You will be provided with an expected output and a retrieval context (list of retrieved documents). Your task is to take each sentence in the expected output and determine whether the sentence is ATTRIBUTABLE or RELEVANT to ANY PART of the retrieval context.

==== FORMATTING YOUR ANSWER ====
Please format your answer as a list of JSON objects, each with two keys: `verdict` and `reason`.
The `verdict` key should STRICTLY be 'yes' or 'no'. You should answer 'yes' if the sentence can be attributed/is relevant to ANY PART(S) of the retrieval context. If not, you should answer 'no'.
The `reason` key should provide a justification of your verdict. In the justification, you should aim to include references to the document(s) in the retrieval context (eg., 1st document, and 2nd document in the retrieval context) that is attributed/relevant to the expected output sentence. 
Please also AIM TO CITE the specific part of the retrieval context to justify your verdict, but **be extremely concise! Cut short the quote with an ellipsis if possible**.

Here's an example of formatting your answer:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
        ...
    ]  
}}

==== EXAMPLE ====
Expected Output:
The Earth's climate has warmed significantly over the past century. This warming is primarily caused by human activities like burning fossil fuels. Today's weather was sunny and warm.

Retrieval Context:
["Global temperatures have risen by approximately 1.1°C since pre-industrial times, with most of this increase occurring in the past 100 years.", 
"Scientific consensus shows that greenhouse gas emissions from human activities, particularly the burning of coal, oil and gas, are the main driver of observed climate change."]

Example Response:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The 1st document directly confirms this, stating 'temperatures have risen by approximately 1.1°C...in the past 100 years'"
        }},
        {{
            "verdict": "yes", 
            "reason": "The 2nd document explicitly states that 'greenhouse gas emissions from human activities, particularly the burning of...fossil fuels' drive climate change"
        }},
        {{
            "verdict": "no",
            "reason": "Neither document contains information about today's specific weather conditions"
        }}
    ]
}}

Since your task is to generate a verdict for each sentence, the number of 'verdicts' SHOULD BE EXACTLY EQUAL to the number of sentences in of `expected output`.
**

==== YOUR TURN ====
Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""
    @staticmethod
    def generate_reason(
        expected_output, supportive_reasons, unsupportive_reasons, score
    ):
        return f"""
==== PROBLEM SETUP ====
You will be provided with an expected output, a list of supportive reasons, a list of unsupportive reasons, and a contextual recall score. Let's break down each input component:
- expected output: A text generated by a language model to answer a question/solve a task.
- supportive reasons: A list of reasons why a specific sentence in the expected output can be attributed/is relevant to any part of the retrieval context (a list of documents retrieved in a RAG pipeline)
- unsupportive reasons: A list of reasons why a specific sentence in the expected output cannot be attributed/is not relevant to any part of the retrieval context
**NOTE**: The reasons are provided in the form of "Sentence <number>: <reason>", where <number> is the sentence number in the expected output.
- contextual recall score: A score between 0 and 1 (closer to 1 the better) representing how much of the expected output can be attributed/is relevant to any part of the retrieval context. 
The point of this score is to measure how well the retriever in a RAG pipeline operates, retrieving relevant documents that should back the expected output of the RAG generator.

==== TASK INSTRUCTIONS ====
Given these inputs, summarize a CONCISE and CLEAR reason for the value of the contextual recall score. Remember, the score is a measure of how well the retriever in a RAG pipeline operates, retrieving relevant documents that should back the expected output of the RAG generator.
In your reason, you should reference the supportive/unsupportive reasons by their sentence number to justify the score. Make specific references to the retrieval context in your reason if applicable.

==== FORMATTING YOUR ANSWER ====
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <contextual_recall_score> because <your_reason>."
}}

DO NOT mention 'supportive reasons' and 'unsupportive reasons' in your reason, these terms are just here for you to understand the broader scope of things.
If the score is 1, keep it short and say something positive with an upbeat encouraging tone (but don't overdo it otherwise it gets annoying).

==== EXAMPLE ====
Expected Output:
The Earth's climate has warmed significantly over the past century. This warming is primarily caused by human activities like burning fossil fuels. Today's weather was sunny and warm.

Supportive Reasons:
Sentence 1: The first document confirms this by stating global temperatures have risen by 1.1°C in the past 100 years
Sentence 2: The second document directly states that human activities and fossil fuel burning drive climate change

Unsupportive Reasons:
Sentence 3: Neither document contains information about today's specific weather conditions

Contextual Recall Score:
0.67

Example Response:
{{
    "reason": "The score is 0.67 because while sentences 1 and 2 are well-supported by the retrieval context with specific temperature data and human activity impacts, sentence 3 about today's weather has no backing in the provided documents."
}}

==== YOUR TURN ====
Contextual Recall Score:
{score}

Expected Output:
{expected_output}

Supportive Reasons:
{supportive_reasons}

Unsupportive Reasons:
{unsupportive_reasons}

JSON:
"""