from judgeval.tracer import Tracer
from judgeval.data import Example
from judgeval.scorers import AnswerRelevancyScorer
import time
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic
from groq import Groq, AsyncGroq
from together import Together, AsyncTogether
from google import genai
from e2etests.utils import (
    retrieve_trace,
    retrieve_score,
    create_project,
    delete_project,
)
from judgeval.tracer import wrap
import os
import random
import pytest
import string
import orjson

project_name = "e2e-tests-" + "".join(
    random.choices(string.ascii_letters + string.digits, k=12)
)

delete_project(project_name=project_name)
create_project(project_name=project_name)


def teardown_module(module):
    delete_project(project_name=project_name)


judgment = Tracer(
    project_name=project_name,
)

# Wrap clients
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())
groq_client = wrap(Groq(api_key=os.getenv("GROQ_API_KEY")))
together_client = wrap(Together(api_key=os.getenv("TOGETHER_API_KEY")))
google_client = wrap(genai.Client(api_key=os.getenv("GOOGLE_API_KEY")))

# Async clients
openai_client_async = wrap(AsyncOpenAI())
anthropic_client_async = wrap(AsyncAnthropic())
groq_client_async = wrap(AsyncGroq(api_key=os.getenv("GROQ_API_KEY")))
together_client_async = wrap(AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY")))

QUERY_RETRY = 15
PROMPT = "I need you to solve this math problem: 1 + 1 = ?"


@judgment.observe(span_type="function")
def scorer_span():
    """Generate a travel itinerary using the researched data."""
    judgment.async_evaluate(
        example=Example(
            input="Tell me the weather in Paris.",
            actual_output="The weather in France is sunny and 72°F.",
        ),
        scorer=AnswerRelevancyScorer(),
        model="gpt-4o-mini",
        sampling_rate=1,
    )

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe(span_type="function")
def recursive_function(number: int):
    if number <= 1:
        trace_id = format(
            judgment.get_current_span().get_span_context().trace_id, "032x"
        )
        return trace_id
    else:
        return recursive_function(number - 1)


@judgment.observe()
def openai_llm_call():
    openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def anthropic_llm_call():
    anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=30,
    )

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def groq_llm_call():
    groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def together_llm_call():
    together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def google_llm_call():
    google_client.models.generate_content(model="gemini-2.0-flash", contents=PROMPT)
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def openai_streaming_llm_call():
    stream = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated_content = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def anthropic_streaming_llm_call():
    stream = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=30,
        stream=True,
    )

    accumulated_content = ""
    for chunk in stream:
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            accumulated_content += chunk.delta.text or ""

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def groq_streaming_llm_call():
    stream = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
    )

    accumulated_content = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
def together_streaming_llm_call():
    stream = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated_content = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def openai_async_llm_call():
    await openai_client_async.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": PROMPT}]
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def anthropic_async_llm_call():
    await anthropic_client_async.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=30,
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def groq_async_llm_call():
    await groq_client_async.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": PROMPT}]
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def together_async_llm_call():
    await together_client_async.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": PROMPT}],
    )
    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def openai_async_streaming_llm_call():
    stream = await openai_client_async.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated_content = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def anthropic_async_streaming_llm_call():
    stream = await anthropic_client_async.messages.create(
        model="claude-3-haiku-20240307",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=30,
        stream=True,
    )

    accumulated_content = ""
    async for chunk in stream:
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
            accumulated_content += chunk.delta.text or ""

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def groq_async_streaming_llm_call():
    stream = await groq_client_async.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
    )

    accumulated_content = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


@judgment.observe()
async def together_async_streaming_llm_call():
    stream = await together_client_async.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROMPT},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    accumulated_content = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    return format(judgment.get_current_span().get_span_context().trace_id, "032x")


def retrieve_trace_helper(trace_id, expected_span_amount):
    query_count = 0
    while query_count < QUERY_RETRY:
        val = retrieve_trace(trace_id)
        if len(val) == expected_span_amount:
            break
        query_count += 1
        time.sleep(1)

    if query_count == QUERY_RETRY:
        assert False, f"Got {len(val)} spans, expected {expected_span_amount}"

    return val


def retrieve_llm_cost_helper(trace_id):
    trace_spans = retrieve_trace_helper(trace_id, 2)

    total_llm_cost = 0
    for span in trace_spans:
        span_attrs = span.get("span_attributes", {})
        if isinstance(span_attrs, str):
            span_attrs = orjson.loads(span_attrs)
        llm_cost = span_attrs.get("gen_ai.usage.total_cost_usd", 0)
        total_llm_cost += llm_cost

    if total_llm_cost == 0:
        assert False, "No LLM cost found"

    return total_llm_cost


def retrieve_streaming_trace_helper(trace_id):
    """Helper to validate streaming traces have proper attributes."""
    trace_spans = retrieve_trace_helper(trace_id, 2)

    # Find the LLM span
    llm_span = None
    for span in trace_spans:
        # Parse span_attributes if it's a JSON string
        span_attrs = span.get("span_attributes", {})
        if isinstance(span_attrs, str):
            span_attrs = orjson.loads(span_attrs)

        if span_attrs.get("judgment.span_kind") == "llm":
            llm_span = span
            break

    if not llm_span:
        assert False, "No LLM span found in streaming trace"

    # Verify streaming-specific attributes
    span_attributes = llm_span.get("span_attributes", {})
    if isinstance(span_attributes, str):
        span_attributes = orjson.loads(span_attributes)

    # Should have completion content
    completion = span_attributes.get("gen_ai.completion")
    if not completion:
        assert False, "No completion content found in streaming span"

    # Should have usage information
    input_tokens = span_attributes.get("gen_ai.usage.input_tokens")
    output_tokens = span_attributes.get("gen_ai.usage.output_tokens")

    if input_tokens is None or output_tokens is None:
        assert False, "Missing usage tokens in streaming span"

    # Should have cost information
    total_cost = span_attributes.get("gen_ai.usage.total_cost_usd")
    if total_cost is None:
        assert False, "Missing cost information in streaming span"

    return trace_spans


def test_trace_spans():
    random_number = random.randint(10, 50)
    trace_id = recursive_function(random_number)
    retrieve_trace_helper(trace_id, random_number)


def test_openai_llm_cost():
    trace_id = openai_llm_call()
    retrieve_llm_cost_helper(trace_id)


def test_anthropic_llm_cost():
    trace_id = anthropic_llm_call()
    retrieve_llm_cost_helper(trace_id)


def test_groq_llm_cost():
    trace_id = groq_llm_call()
    retrieve_llm_cost_helper(trace_id)


def test_together_llm_cost():
    trace_id = together_llm_call()
    retrieve_llm_cost_helper(trace_id)


def test_google_llm_cost():
    trace_id = google_llm_call()
    retrieve_llm_cost_helper(trace_id)


@pytest.mark.asyncio
async def test_openai_async_llm_cost():
    trace_id = await openai_async_llm_call()
    retrieve_llm_cost_helper(trace_id)


@pytest.mark.asyncio
async def test_anthropic_async_llm_cost():
    trace_id = await anthropic_async_llm_call()
    retrieve_llm_cost_helper(trace_id)


@pytest.mark.asyncio
async def test_groq_async_llm_cost():
    trace_id = await groq_async_llm_call()
    retrieve_llm_cost_helper(trace_id)


@pytest.mark.asyncio
async def test_together_async_llm_cost():
    trace_id = await together_async_llm_call()
    retrieve_llm_cost_helper(trace_id)


# Sync streaming tests
def test_openai_streaming_llm_cost():
    trace_id = openai_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


def test_anthropic_streaming_llm_cost():
    trace_id = anthropic_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


def test_together_streaming_llm_cost():
    trace_id = together_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


# Async streaming tests
@pytest.mark.asyncio
async def test_openai_async_streaming_llm_cost():
    trace_id = await openai_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


@pytest.mark.asyncio
async def test_anthropic_async_streaming_llm_cost():
    trace_id = await anthropic_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


@pytest.mark.asyncio
async def test_together_async_streaming_llm_cost():
    trace_id = await together_async_streaming_llm_call()
    retrieve_streaming_trace_helper(trace_id)


def test_online_span_scoring():
    trace_id = scorer_span()
    trace_spans = retrieve_trace_helper(trace_id, 1)
    span_id = trace_spans[0].get("span_id")

    query_count = 0
    while query_count < QUERY_RETRY:
        try:
            scorer_data = retrieve_score(span_id, trace_id)
        except Exception:
            pass

        if scorer_data:
            break
        query_count += 1
        time.sleep(1)

    print(scorer_data)
    if query_count == QUERY_RETRY:
        assert False, "No score found"

    score = scorer_data[0]
    assert score.get("scorer_name") == "Answer Relevancy"
    assert score.get("scorer_success")
    assert score.get("scorer_score") == 1.0
