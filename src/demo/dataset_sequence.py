from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

dataset = client.pull_dataset(alias="test2", project_name="agent")
client.run_sequence_evaluation(
    sequences=dataset.sequences,
    model="gpt-4.1",
    project_name="agent",
    scorers=[DerailmentScorer(threshold=0.5)],
    log_results=True,
    override=True
)