from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

dataset = client.pull_dataset(alias="test", project_name="simple_trace_demo")

client.evaluate_sequence_dataset(
    dataset=dataset,
    model="gpt-4o",
    project_name="simple_trace_demo",
    scorers=[DerailmentScorer(threshold=0.5)],
    log_results=True,
    override=True,
)