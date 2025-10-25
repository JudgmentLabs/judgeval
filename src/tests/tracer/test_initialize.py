import os
from judgeval.tracer import Tracer
from judgeval.scorers import AnswerRelevancyScorer
from judgeval.data import Example


def test_initialize(monkeypatch):
    monkeypatch.setenv("JUDGMENT_API_KEY", "test_api_key")
    monkeypatch.setenv("JUDGMENT_ORG_ID", "test_org_id")

    # Initialize tracer
    tracer = Tracer(project_name="research-agent")

    # Verify tracer was created successfully
    assert tracer is not None
    assert tracer.project_name == "research-agent"


def test_initialize_empty_env(monkeypatch):
    # Remove environment variables to simulate them not being set
    if "JUDGMENT_API_KEY" in os.environ:
        monkeypatch.delenv("JUDGMENT_API_KEY")
    if "JUDGMENT_ORG_ID" in os.environ:
        monkeypatch.delenv("JUDGMENT_ORG_ID")
    # Initialize tracer
    tracer = Tracer(project_name="research-agent")

    # Verify tracer was created successfully even with missing env vars
    assert tracer is not None
    assert tracer.project_name == "research-agent"

    @tracer.observe(span_type="function")
    def print_hello():
        tracer.set_customer_id("customer-123")
        print("hello")

        tracer.async_evaluate(
            scorer=AnswerRelevancyScorer(),
            example=Example(input="test-input", output="test-output"),
            model="gpt-4o-mini",
        )

    print_hello()
