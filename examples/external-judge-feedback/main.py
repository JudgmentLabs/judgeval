from judgeval import Judgeval

client = Judgeval(project_name="my-project")

# External judges don't run on the Judgment platform -- you register the
# judge's name and score shape once, then submit as many results as you
# like from your own evaluation code (a human review queue, an offline
# eval job, another LLM judge you already run elsewhere, etc.).
judge = client.external_judges.create(
    name="human-thumbs-up",
    score_type="binary",
    judge_description="Whether a human reviewer approved the response.",
)
print(f"Created external judge: {judge.judge_id}")

# Replace with the id of a trace you've already ingested (e.g. from a
# `Tracer.observe()`-instrumented run, or copied from the platform UI).
trace_id = "<trace_id>"

result_id = client.external_judges.submit_result(
    judge_id=judge.judge_id,
    trace_id=trace_id,
    value=True,
    reason="Reviewer approved the response.",
)
print(f"Submitted result: {result_id}")

# A categorical judge scores against a fixed set of named outputs instead
# of a boolean/number.
topic_judge = client.external_judges.create(
    name="topic-classifier",
    score_type="categorical",
    outputs=[
        {"name": "billing", "description": "Billing questions"},
        {"name": "support", "description": "Support requests"},
    ],
)

client.external_judges.submit_result(
    judge_name=topic_judge.name,
    trace_id=trace_id,
    value="billing",
)
