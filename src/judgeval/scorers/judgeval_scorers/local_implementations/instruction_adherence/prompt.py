"""
Util prompts for InstructionAdherenceScorer
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class InstructionAdherenceTemplate:
    @staticmethod
    def get_instructions(input):
        return f"""You will be presented with a piece of text. Your task is to break down the text and generate a list of the instructions contained within the text.

===== START OF EXAMPLES =====
Example 1:
Example text: Hello my name is John Doe. I like cars. Write two poems about the weather and create a joke. Also what is 5 + 5?

Output:
{{
    "instructions": ["Write two poem about the weather", "Create a joke", "What is 5 + 5?"]
}}
===== END OF EXAMPLES =====

        
**
IMPORTANT: Please return your answer in valid JSON format, with the "instructions" key mapping to a list of strings. No words or explanation is needed.
**

==== START OF INPUT ====
Text:
{input}
==== END OF INPUT ====

==== YOUR ANSWER ====
JSON:
"""

    @staticmethod
    def generate_verdicts(instructions, actual_output):
        return f"""
        You will be presented with a list of instructions and a piece of text. For each instruction, determine if the instruction was completed in the text. There are 3 categories: either completed, partially completed, or not completed. The scores for these will be 1, 0.5, and 0 respectively.
        Go through each instruction and provide score for each instruction as well as the reasoning for that score.

        ==== FORMATTING YOUR ANSWER ====
        Please return your answer in JSON format, with a list of JSON objects with keys "instruction", "score", and "reason". No words or explanation beyond the output JSON is needed.


        ===== START OF EXAMPLES =====
        Example 1:
        instructions: ["Write two poems about the weather", "Create a joke", "What is 5 + 5?"]
        output: Poem 1: The Sun's Embrace
        The sun climbs high, a golden flame,
        It whispers warmth, it calls my name.
        The sky, a canvas, blue and clear,
        A perfect day for cars, my dear.

        The asphalt hums beneath the wheels,
        A symphony of speed it feels.
        The weather smiles, no clouds in sight,
        A driver's joy, pure delight.

        Poem 2: The Storm's Dance
        A sunlit meadow, alive with whispers of wind, where daisies dance and hope begins again. Each petal holds a promise—bright, unbruised— a symphony of light that cannot be refused.

        Joke
        Why dont cars ever get cold in the winter?
        Because they have radiators!

        Math Answer
        5 + 5 = 10
        
        YOUR JSON OUTPUT:
        {{
            [
                {{
                    "instruction": "Write two poem about the weather",
                    "score": 0.5,
                    "reason": "The output contained one poem about the weather, but the other poem was not about the weather."
                }},
                {{
                    "instruction": "Create a joke",
                    "score": 1,
                    "reason": "There was a joke created in the output."
                }},
                {{
                    "instruction": "What is 5 + 5?",
                    "score": 1,
                    "reason": "The answer to the math question was provided in the output."
                }}
            ]
        }}
        ===== END OF EXAMPLES =====
        
        ==== START OF INPUT ====
        instructions: {instructions}
        output: {actual_output}
        ==== END OF INPUT ====

        ==== YOUR ANSWER ====
        JSON:
        """

