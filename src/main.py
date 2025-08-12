import os


JUDGMENT_ORG_ID = "27c7d380-6cb0-495f-b27c-8c1ff82b72af"
JUDGMENT_API_KEY = "736046e3-b3a2-4a1a-bd3e-af6a6f8a9a5d"
# JUDGMENT_API_URL = "http://localhost:8000"
JUDGMENT_API_URL = "https://staging.api.judgmentlabs.ai"

os.environ["JUDGMENT_API_KEY"] = JUDGMENT_API_KEY
os.environ["JUDGMENT_ORG_ID"] = JUDGMENT_ORG_ID
os.environ["JUDGMENT_API_URL"] = JUDGMENT_API_URL

from judgeval.scorers.judgeval_scorers.api_scorers.faithfulness import (
    FaithfulnessScorer,
)
from judgeval.tracer import Tracer
from judgeval import JudgmentClient
from judgeval.data import Example

tracer = Tracer(
    project_name="ahh",
)


client = JudgmentClient()

from judgeval.dataset import Dataset


from judgeval.dataset import Dataset
from judgeval.data import Example

# dataset = Dataset.create(
#     name="qa_dataset",
#     project_name="default_project",
#     examples=[Example(input="...", actual_output="...")],
# )

dataset = Dataset.get(name="qa_dataset", project_name="default_project")

print(dataset)

