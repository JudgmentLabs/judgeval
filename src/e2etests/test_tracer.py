# Standard library imports
import os
import time
import asyncio

# Third-party imports
from openai import OpenAI
from together import Together
from anthropic import Anthropic

# Local imports
from judgeval.common.tracer import Tracer, wrap
from judgeval.constants import APIScorer
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

# Initialize the tracer and clients
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())

@judgment.observe(span_type="tool")
async def make_upper(input: str) -> str:
    """Convert input to uppercase and evaluate using judgment API.
    
    Args:
        input: The input string to convert
    Returns:
        The uppercase version of the input string
    """
    output = input.upper()
    
    await judgment.get_current_trace().async_evaluate(
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
async def make_lower(input):
    output = input.lower()
    
    await judgment.get_current_trace().async_evaluate(
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
async def answer_user_question(input):
    output = llm_call(input)
    await judgment.get_current_trace().async_evaluate(
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
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": input}],
            max_tokens=30
        )
        anthropic_result = anthropic_response.content[0].text
        
        await judgment.get_current_trace().async_evaluate(
            input=input,
            actual_output=anthropic_result,
            score_type=APIScorer.ANSWER_RELEVANCY,
            threshold=0.5,
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
        
        return await make_lower(f"{anthropic_result} {openai_result}")
    
    except Exception as e:
        print(f"Error generating poem: {e}")
        return ""

async def test_evaluation_mixed(input):
    PROJECT_NAME = "NewPoemBot"
    with judgment.trace("Use-claude", project_name=PROJECT_NAME, overwrite=True) as trace:
        upper = await make_upper(input)
        result = await make_poem(upper)
        await answer_user_question("What if these shoes don't fit?")

    trace.save()
        
    trace.print()
    
    return result

if __name__ == "__main__":
    # Use a more meaningful test input
    test_input = "Write a poem about Nissan R32 GTR"
    asyncio.run(test_evaluation_mixed(test_input))

