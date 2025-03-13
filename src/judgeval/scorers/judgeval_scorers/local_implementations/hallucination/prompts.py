from typing import List
from pydantic import BaseModel


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str


class HallucinationTemplate:
    @staticmethod
    def generate_verdicts(actual_output, contexts):
        return f"""==== TASK INSTRUCTIONS ====
You will be provided with an `actual output` (the response of an LLM to a particular query) and `contexts` (ground truth contextual information from a knowledge base).
Your task is to take each context in contexts and determine whether the `actual output` factually agrees with the context.

Additional notes:
You should NOT use any prior knowledge you have in your decision making process; take each context at face value. 
Since you will determine a verdict for EACH context, the number of 'verdicts' is EXACTLY EQUAL TO the number of contexts. 
You should be lenient in your judgment when the actual output lacks detail with respect to the context segment; you should ONLY provide a 'no' answer if the context contradicts the actual output.

==== FORMATTING INSTRUCTIONS ====
You should return a JSON object with a key 'verdicts', which is a list of JSON objects. Each JSON object corresponds to a context in `contexts`, and should have 2 fields: 'verdict' and 'reason'. 
The 'verdict' key should be EXACTLY one of 'yes' or 'no', representing whether the `actual output` factually agrees with the context segment. 
The 'reason' is the justification for the verdict. If your verdict is 'no', try to provide a correction in the reason. 

==== EXAMPLE ====
Example contexts: ["Einstein won the Nobel Prize for his discovery of the photoelectric effect.", "Einstein won the Nobel Prize in 1968."]
Example actual output: "Einstein won the Nobel Prize in 1969 for his discovery of the photoelectric effect."

Example:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "The actual output agrees with the provided context which states that Einstein won the Nobel Prize for his discovery of the photoelectric effect."
        }},
        {{
            "verdict": "no",
            "reason": "The actual output contradicts the provided context which states that Einstein won the Nobel Prize in 1968, not 1969."
        }}
    ]  
}}

==== YOUR TURN ====
Contexts:
{contexts}

Actual Output:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_reason(contradictions, score):
        return f"""==== TASK INSTRUCTIONS ====
An LLM has been provided with a list of `contexts` (ground truth contextual information from a knowledge base) and `actual output` (the response of an LLM to a particular query). 
You will be provided with a list of `contradictions`, which are factual discrepancies between the context segments and the actual output. 
Additionally, you will be provided with a hallucination score, which is a float (0 - 1, where 0 is the best score) indicating the fraction of context segments that contradict the actual output.

Your task is to provide a CLEAR and CONCISE reason for the hallucination score. 
If the hallucination score is 0 (no contradictions), you should instead respond with a positive remark with an upbeat encouraging tone (but don't overblow the kind attitude).
        
==== FORMATTING INSTRUCTIONS ====
Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <hallucination_score> because <your_reason>."
}}

==== EXAMPLE ====
Example Contradictions:
[
    "The actual output claims Einstein won the Nobel Prize in 1969, which contradicts the context stating he won it in 1968.",
    "The actual output states Einstein was a chemist, but the context indicates he was a physicist.",
    "The actual output claims Einstein was born in Switzerland, while the context states he was born in Germany."
]

Example Hallucination Score: 
0.75

Example Response:
{{
    "reason": "The score is 0.75 because the actual output made multiple factual errors: incorrectly stating Einstein's Nobel Prize year (1969 vs 1968), his profession (chemist vs physicist), and birthplace (Switzerland vs Germany)."
}}

==== YOUR TURN ====
Contradictions:
{contradictions}

Hallucination Score:
{score}

JSON:
"""
