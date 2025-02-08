"""
Test to implement a PromptScorer

Toy example in this case to determine the sentiment
"""

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.judges import TogetherJudge
from judgeval.scorers import PromptScorer, ClassifierScorer
import random
import string


qwen = TogetherJudge()


class SentimentScorer(PromptScorer):
    """
    Detects negative sentiment (angry, sad, upset, etc.) in a response
    """
    def __init__(
        self, 
        name="Sentiment Scorer", 
        threshold=0.5, 
        include_reason=True, 
        async_mode=True, 
        strict_mode=False, 
        verbose_mode=False
        ):
        super().__init__(
            name=name,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
        )
        self.score = 0.0

    def _build_measure_prompt(self, example: Example):
        SYSTEM_ROLE = (
            'You are a great judge of emotional intelligence. You understand the feelings ' 
            'and intentions of others. You will be tasked with judging whether the following '
            'response is negative (sad, angry, upset) or not. After deciding whether the '
            'response is negative or not, you will be asked to provide a brief, 1 sentence-long reason for your decision.'
            'You should score the response based on a 1 to 5 scale, where 1 is not negative and '
            '5 is very negative.'
                )
        conversation = [
            {"role": "system", "content": SYSTEM_ROLE},
            {"role": "user", "content": f"Response: {example.actual_output}\n\nYour judgment: "}
        ] 
        return conversation
    
    def _build_schema(self):
        return {
            "score": int,
            "reason": str
        }
    
    def _process_response(self, response):
        return response["score"], response["reason"]
    
    def _success_check(self):
        POSITIVITY_THRESHOLD = 3  # we want all model responses to be somewhat positive in tone
        return self.score <= POSITIVITY_THRESHOLD


def generate_random_slug(length=6):
    """Generate a random string of fixed length"""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def main():
    pos_example = Example(
        input="What's the store return policy?",
        actual_output="Our return policy is wonderful! You may return any item within 30 days of purchase for a full refund.",
    )

    neg_example = Example(
        input="I'm having trouble with my order",
        actual_output="That's not my problem. You should have read the instructions more carefully.",
    )

    # scorer = SentimentScorer()
    scorer = ClassifierScorer(
        slug=generate_random_slug(),  # Generate random 6-letter slug
        name="Sentiment Classifier",
        conversation=[{"role": "system", "content": "Is the response positive (Y/N)? The response is: {{actual_output}}."}],
        options={"Y": 1, "N": 0},
        threshold=0.5,
        include_reason=True,
    )

    import requests 
    from dotenv import load_dotenv
    load_dotenv()
    import os
    response = requests.post(
        "http://127.0.0.1:8000/classifier_eval/",
        json={
            "conversation": [{"role": "system", "content": "Is the response positive (Y/N)? The response is: {{actual_output}}."}],
            "options": {"Y": 1, "N": 0},
            "example": pos_example.model_dump(),
            "model": "QWEN",
            "judgment_api_key": os.getenv("JUDGMENT_API_KEY")
        }
    )
    print(response.json())

    # client = JudgmentClient()
    # results = client.run_evaluation(
    #     [pos_example, neg_example],
    #     [scorer],
    #     model="QWEN"
    # ) 
    # print(results)

if __name__ == "__main__":
    main()
