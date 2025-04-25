from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

airlines_example = Example(
    input="Which airlines fly to Tokyo?",
    actual_output="Japan Airlines, All Nippon Airways, and Chinese Airlines offer direct flights."
)
weather_example = Example(
    input="What is the weather like in Japan?",
    actual_output="It's cloudy with a high of 75°F and a low of 60°F in Japan."
)
airline_sequence = Sequence(
    name="Flight Details",
    items=[airlines_example, weather_example],
)

# Level 1: Top-level sequence
top_example1 = Example(
    input="I want to plan a trip to Tokyok.",
    actual_output="That sounds great! When are you planning to go?"
)
top_example2 = Example(
    input="Can you book a flight for me and anything else I need to know?",
    actual_output="Sure, I'll help you with flights. hotels. and transportation."
)
top_level_sequence = Sequence(
    name="Travel Planning",
    items=[top_example1, top_example2, airline_sequence],
)

other_sequence = Sequence(
    name="Other",
    items=[Example(
        input="What is the weather like in Tokyo?",
        actual_output="It's cloudy with a high of 75°F and a low of 60°F in Tokyo."
    )]
)

results = client.run_sequence_evaluation(
    eval_run_name="sequence-run1",
    project_name="jnpr-demo-sequence",
    scorers=[DerailmentScorer(threshold=1)],
    sequences=[top_level_sequence, other_sequence],
    model="gpt-4o",
    log_results=True,
    override=True,
)