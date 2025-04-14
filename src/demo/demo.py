from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

airlines_example = Example(
    input="Which airlines fly to Paris?",
    actual_output="Air France, Delta, and American Airlines offer direct flights."
)
weather_example = Example(
    input="What is the weather like in Texas?",
    actual_output="It's sunny with a high of 75°F in Texas."
)
airline_sequence = Sequence(
    name="Flight Details",
    items=[airlines_example, weather_example],
    scorers=[DerailmentScorer(threshold=0.5)]
)

# Level 1: Top-level sequence
top_example1 = Example(
    input="I want to plan a trip to Paris.",
    actual_output="Great! When are you planning to go?"
)
top_example2 = Example(
    input="Can you book a flight for me?",
    actual_output="Sure, I'll help you with flights and hotels."
)
top_level_sequence = Sequence(
    name="Travel Planning",
    items=[top_example1, top_example2, airline_sequence],
    scorers=[DerailmentScorer(threshold=1)]
)

results = client.run_sequence_evaluation(
    eval_run_name="sequence-run2",
    project_name="jnpr-demo-sequence",
    sequences=[top_level_sequence],
    model="gpt-4o",
    log_results=True,
    override=True,
)