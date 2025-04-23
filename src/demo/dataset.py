from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

dataset = client.pull_dataset(alias="test", project_name="travel_agent_demo_test")

client.run_sequence_evaluation(
    sequences=dataset.sequences,
    model="gpt-4o",
    project_name="travel_agent_demo_test",
    scorers=[DerailmentScorer(threshold=0.5)],
    log_results=True,
    override=True,
)