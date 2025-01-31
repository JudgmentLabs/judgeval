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
judgment = Tracer(api_key=os.getenv("UI_JUDGMENT_API_KEY"))
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

@judgment.observe(span_type="tool")
async def retrieve_keywords(query: str) -> list[str]:
    """Simulate keyword retrieval from query."""

    # Mock relevance filtering - in real system this would use embeddings or semantic search
    # Mock relevance filtering by checking if query terms appear in documents
    from pydantic import BaseModel
    class Document(BaseModel):
        relevant: list[str]

    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "Extract the key words."},
            {"role": "user", "content": query},
        ],
        response_format=Document,
    )

    words = completion.choices[0].message.parsed

    # Mock relevance filtering by checking if query terms appear in documents
    query_terms = [word for term in words.relevant for word in term.split()]
    return query_terms

@judgment.observe(span_type="tool")
async def retrieve_documents(query_terms: str, documents: list[str]) -> list[str]:
    """Simulate document retrieval from enterprise knowledge base."""
    
    # Filter documents that contain any query terms
    relevant_docs = [
        doc for doc in documents 
        if any(term in doc.lower() for term in query_terms)
    ]

    return relevant_docs

@judgment.observe(span_type="tool")
async def summarize_documents(relevant_documents: list[str]) -> str:
    """Simulate document summarization."""


    anthropic_response = anthropic_client.messages.create(
        model="claude-3-sonnet-20240229",
        system="Make a summary with the relevant documents.",
        messages=[
            {"role": "user", "content": "\n".join(relevant_documents)}
        ],
        max_tokens=500
    )

    summary = anthropic_response.content[0].text

    await judgment.get_current_trace().async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5)],
        input="Make a summary with the relevant documents.",
        actual_output=summary,
        retrieval_context=relevant_documents,
        model="gpt-4o-mini",
        log_results=True
    )

    return summary


@judgment.observe(span_type="tool")
async def generate_answer(query: str, context: str) -> str:
    """Generate answer using LLM."""
    # Simulate LLM call
    response = f"Based on our company policy: {context}"
    
    await judgment.get_current_trace().async_evaluate(
        scorers=[AnswerRelevancyScorer(threshold=0.5)],
        input=query,
        actual_output=response,
        retrieval_context=[context],
        expected_output=response,
        model="gpt-4o-mini",
        log_results=True
    )
    return response


async def rag_pipeline(query: str, documents: list[str]) -> str:
    """Enterprise RAG pipeline with multiple steps."""
    PROJECT_NAME = "EnterpriseRAG"
    
    with judgment.trace("RAG-Pipeline", project_name=PROJECT_NAME, overwrite=True) as trace:

        keywords = await retrieve_keywords(query)
        
        # Step 1: Retrieve relevant documents
        relevant_docs = await retrieve_documents(keywords, documents)
        
        # Step 2: Summarize retrieved documents
        context = await summarize_documents(relevant_docs)
        
        # Step 3: Generate answer
        final_response = await generate_answer(query, context)
        
    trace.save()
    trace.print()
    
    return final_response

if __name__ == "__main__":
    # Use a more meaningful test input
    test_input = "Write a poem about Nissan R32 GTR"
    #asyncio.run(test_evaluation_mixed(test_input))

    documents = [
        "The company holiday party is scheduled for December 15th at the Grand Hotel.",
        "Employees can be reimbursed for business travel expenses with proper receipts.",
        "All staff must complete annual cybersecurity training by March 31st.",
        "Travel expenses include flights, hotels, and meals during business trips.",
        "The office will be closed for maintenance this weekend.",
        "Submit expense reports within 30 days of travel completion.",
        "New parking permits will be issued next month for all employees.",
        "International travel requires manager pre-approval.",
        "The cafeteria now offers vegetarian options every Wednesday.",
        "Employee health insurance enrollment period begins next week.",
        "Office supplies can be requested through the internal portal.",
        "Remote work policy requires minimum 2 days in office per week."
    ]

    rag_input = "What is the company policy on reembursing travel expenses?"
    asyncio.run(rag_pipeline(rag_input, documents))

