# Standard library imports
import os
import time
import asyncio
from typing import List
import pytest

# Third-party imports
from openai import OpenAI
from together import Together
from anthropic import Anthropic
import pytest

# Local imports
from judgeval.tracer import Tracer, wrap, TraceClient, TraceManagerClient
from judgeval.constants import APIScorer
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

# Initialize the tracer and clients
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())

@judgment.observe(span_type="tool")
@pytest.mark.asyncio
async def make_upper(input: str) -> str:
    """Convert input to uppercase and evaluate using judgment API.
    
    Args:
        input: The input string to convert
    Returns:
        The uppercase version of the input string
    """
    output = input.upper()
    
    judgment.get_current_trace().async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5)],
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        expected_output="We offer a 30-day full refund at no extra cost.",
        expected_tools=["refund"],
        model="gpt-4o-mini",
        log_results=True
    )

    return output

@judgment.observe(span_type="tool")
@pytest.mark.asyncio
async def make_lower(input):
    output = input.lower()
    
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input="How do I reset my password?",
        actual_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        expected_output="You can reset your password by clicking on 'Forgot Password' at the login screen.",
        context=["User Account"],
        retrieval_context=["Password reset instructions"],
        tools_called=["authentication"],
        expected_tools=["authentication"],
        additional_metadata={"difficulty": "medium"},
        model="gpt-4o-mini",
        log_results=True
    )
    return output

@judgment.observe(span_type="llm")
def llm_call(input):
    time.sleep(1.3)
    return "We have a 30 day full refund policy on shoes."

@judgment.observe(span_type="tool")
@pytest.mark.asyncio
async def answer_user_question(input):
    output = llm_call(input)
    judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=input,
        actual_output=output,
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
        expected_output="We offer a 30-day full refund at no extra cost.",
        model="gpt-4o-mini",
        log_results=True
    )
    return output

@judgment.observe(span_type="tool")
@pytest.mark.asyncio
async def make_poem(input: str) -> str:
    """Generate a poem using both Anthropic and OpenAI APIs.
    
    Args:
        input: The prompt for poem generation
    Returns:
        Combined and lowercase version of both API responses
    """
    try:
        # Using Anthropic API
        anthropic_response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": input}],
            max_tokens=30
        )
        anthropic_result = anthropic_response.content[0].text
        
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerRelevancyScorer(threshold=0.5)],
            input=input,
            actual_output=anthropic_result,
            model="gpt-4o-mini",
            log_results=True
        )
        
        # Using OpenAI API
        openai_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Make a short sentence with the input."},
                {"role": "user", "content": input}
            ]
        )
        openai_result = openai_response.choices[0].message.content
        return await make_lower(f"{openai_result} {anthropic_result}")
    
    except Exception as e:
        print(f"Error generating poem: {e}")
        return ""

@pytest.fixture
def trace_manager_client():
    """Fixture to initialize TraceManagerClient."""
    return TraceManagerClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"))

@pytest.mark.asyncio
async def test_token_counting(trace_manager_client):
    input = "Write a poem about Nissan R32 GTR"

@pytest.fixture
def test_input():
    """Fixture providing default test input"""
    return "What if these shoes don't fit?"

@pytest.mark.asyncio
async def test_evaluation_mixed(test_input):
    PROJECT_NAME = "TestingPoemBot"
    print(f"Using test input: {test_input}")
    with judgment.trace("Use-claude-hehexd123", project_name=PROJECT_NAME, overwrite=True) as trace:
        upper = await make_upper(test_input)
        result = await make_poem(upper)
        await answer_user_question("What if these shoes don't fit?")
        
        # Save trace data and test token counting
        trace_id, trace_data = trace.save()
        
        """Test that token counts are properly aggregated from different LLM API calls."""
        # Verify token counts exist and are properly aggregated
        token_counts = trace_data["token_counts"]
        assert token_counts["prompt_tokens"] > 0, "Prompt tokens should be counted"
        assert token_counts["completion_tokens"] > 0, "Completion tokens should be counted"
        assert token_counts["total_tokens"] > 0, "Total tokens should be counted"
        assert token_counts["total_tokens"] == (
            token_counts["prompt_tokens"] + token_counts["completion_tokens"]
        ), "Total tokens should be equal to the sum of prompt and completion tokens"
        
        # Print token counts for verification
        print("\nToken Count Results:")
        print(f"Prompt Tokens: {token_counts['prompt_tokens']}")
        print(f"Completion Tokens: {token_counts['completion_tokens']}")
        print(f"Total Tokens: {token_counts['total_tokens']}")
        
        trace.print()
        return result

@pytest.mark.asyncio
async def test_trace_delete(trace_manager_client):
    with judgment.trace("TEST_RUN", project_name="TEST", overwrite=True) as trace:
        pass

    response = trace_manager_client.fetch_trace(trace.trace_id)
    assert response, "No traces found"

    trace_manager_client.delete_trace(trace.trace_id)
    response = trace_manager_client.fetch_trace(trace.trace_id)
    assert not response, "Trace should be deleted"

@pytest.mark.asyncio
async def test_trace_delete_batch(trace_manager_client):
    with judgment.trace("TEST_RUN2", project_name="TEST", overwrite=True) as trace:
        pass

    with judgment.trace("TEST_RUN3", project_name="TEST2", overwrite=True) as trace2:
        pass

    response = trace_manager_client.fetch_trace(trace.trace_id)
    assert response, "No traces found"

    response = trace_manager_client.fetch_trace(trace2.trace_id)
    assert response, "No traces found"

    trace_ids = [trace.trace_id, trace2.trace_id]
    response = trace_manager_client.delete_traces(trace_ids)
    assert response, "Delete batch should be successful"

    response = trace_manager_client.fetch_trace(trace.trace_id)
    assert not response, "Trace should be deleted"

    response = trace_manager_client.fetch_trace(trace2.trace_id)
    assert not response, "Trace should be deleted"

    trace_manager_client.delete_project(project_name="TEST")
    trace_manager_client.delete_project(project_name="TEST2")
    
@pytest.mark.asyncio
async def run_selected_tests(test_names: list[str]):
    """
    Run only the specified tests by name.
    
    Args:
        test_names (list[str]): List of test function names to run (without 'test_' prefix)
    """

    trace_manager_client = TraceManagerClient(judgment_api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"))
    print("Client initialized successfully")
    print("*" * 40)
    
    test_map = {
        'token_counting': test_token_counting,
        'trace_delete': test_trace_delete,
        'trace_delete_batch': test_trace_delete_batch,
    }

    for test_name in test_names:
        if test_name not in test_map:
            print(f"Warning: Test '{test_name}' not found")
            continue
            
        print(f"Running test: {test_name}")
        await test_map[test_name](trace_manager_client)
        print(f"{test_name} test successful")
        print("*" * 40)

if __name__ == "__main__":
    # Use a more meaningful test input
    asyncio.run(run_selected_tests([
        "token_counting", 
        "trace_delete", 
        "trace_delete_batch"
        ]))
