"""
This script is used to score the results of the agentic RAG.

Here, we implement a simple scorer that checks if an agent's research used Twitter as a source. 
For financial analysis, Twitter is not a reliable source of information, so we want to penalize our agent
if it uses Twitter.
"""

from judgeval.scorers import ClassifierScorer


twitter_scorer = ClassifierScorer(
    "No Twitter",
    slug="no_twitter-487126418",
    threshold=1.0,
    conversation=[{
        "role": "system",
        "content": "Does the following research use Twitter as a source? (Y/N).\n\nResearch: {{actual_output}}"
    }],
    options={"Y": 0.0, "N": 1.0}
)

