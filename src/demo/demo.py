from judgeval import JudgmentClient
from judgeval.data import Example, Sequence
from judgeval.scorers import DerailmentScorer

client = JudgmentClient()

# example = Example(
#     input="How do I prepare this recipe?",
#     actual_output="Here are the steps: Preheat the oven, mix the ingredients, bake for 30 minutes, etc.",
# )
# example2 = Example(
#     input="What is the weather like?",
#     actual_output="It's sunny with a high of 75°F."
# )
# example3 = Example(
#     input="What is recipe step 5 again?",
#     actual_output="Recipe step 5: Let the dough rest for 10 minutes"
# )

# nested_example = Example(
#     input="What is recipe step 5 again?",
#     actual_output="Recipe step 5: Let the dough rest for 10 minutes",
# )   

# nested_example2 = Example(
#     input="What is weather like?",
#     actual_output="It's sunny with a high of 75°F.",
# )

# nested_sequence = Sequence(
#     name="Recipe Steps",
#     items=[nested_example, nested_example2],
#     scorers=[DerailmentScorer(threshold=0.5)]
# )

# sequence = Sequence(
#     name="Refund Policy",
#     items=[example, example2, nested_sequence, example3],
#     scorers=[DerailmentScorer(threshold=0.5)]
# )

# example4 = Example(
#     input="What is the weather like?",
#     actual_output="It's sunny with a high of 75°F.",
# )

# example5 = Example(
#     input="What is the weather like?",
#     actual_output="It's sunny with a high of 75°F.",
# )

# sequence2 = Sequence(
#     name="Weather",
#     items=[example4, example5],
#     scorers=[DerailmentScorer(threshold=0.5)]
# )

# Level 3: Deepest sequence - Airline baggage policy specifics
airline_policy_example1 = Example(
    input="What is Delta's carry-on limit?",
    actual_output="Delta allows one carry-on and one personal item per passenger."
)
airline_policy_example2 = Example(
    input="Are there weight restrictions?",
    actual_output="Yes, the carry-on must be under 22 lbs and fit in the overhead bin."
)
airline_policy_sequence = Sequence(
    name="Delta Airline Policy",
    items=[airline_policy_example1, airline_policy_example2],
    scorers=[DerailmentScorer(threshold=0.5)]
)

# Level 2: Nested sequence - Airline comparison
airlines_example = Example(
    input="Which airlines fly to Paris?",
    actual_output="Air France, Delta, and American Airlines offer direct flights."
)
airline_sequence = Sequence(
    name="Flight Details",
    items=[airlines_example, airline_policy_sequence],
    scorers=[DerailmentScorer(threshold=0.5)]
)

# Level 1: Top-level sequence
top_example1 = Example(
    input="I want to plan a trip to Paris.",
    actual_output="Great! When are you planning to go?"
)
top_example2 = Example(
    input="Can you book a flight for me?",
    actual_output="Sure, I’ll help you with flights and hotels."
)
top_level_sequence = Sequence(
    name="Travel Planning",
    items=[top_example1, top_example2, airline_sequence],
    scorers=[DerailmentScorer(threshold=0.5)]
)

scorer = DerailmentScorer(threshold=0.5)
results = client.run_sequence_evaluation(
    eval_run_name="test-sequence-run4",
    project_name="test-sequence-project",
    sequences=[top_level_sequence],
    model="gpt-4o",
    log_results=True,
    override=True,
)
print(results)