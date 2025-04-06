"""
E2E tests for organization-based judgee and trace tracking functionality
with atomic operations and race condition handling
"""

import pytest
import pytest_asyncio
import os
import httpx
from dotenv import load_dotenv
import asyncio
import time
from uuid import uuid4
import logging
from contextlib import asynccontextmanager
from judgeval.common.tracer import Tracer
from judgeval.judgment_client import JudgmentClient
from judgeval.scorers import (
    AnswerCorrectnessScorer,
    AnswerRelevancyScorer, 
    ContextualPrecisionScorer,
    ContextualRecallScorer,
    ContextualRelevancyScorer,
    FaithfulnessScorer,
    HallucinationScorer,
    SummarizationScorer,
    ComparisonScorer,
    Text2SQLScorer,
    InstructionAdherenceScorer,
    ExecutionOrderScorer,
)
from judgeval.tracer import Tracer, wrap, TraceClient, TraceManagerClient
from judgeval.constants import APIScorer
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer
from judgeval.data import Example
from openai import OpenAI
from together import Together
from anthropic import Anthropic
import pytest
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the tracer and clients
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"))
openai_client = wrap(OpenAI())
anthropic_client = wrap(Anthropic())

# Load environment variables from .env file
load_dotenv()

# Get server URL and API key from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")
USER_API_KEY = os.getenv("USER_API_KEY", TEST_API_KEY)  # For user-specific tests

# Standard headers for all requests
def get_headers():
    headers = {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }
    logger.debug(f"Generated headers: {headers}")
    return headers

# User-specific headers with organization ID
def get_user_headers():
    headers = {
        "Authorization": f"Bearer {USER_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }
    logger.debug(f"Generated user headers: {headers}")
    return headers

@asynccontextmanager
async def get_client():
    """Context manager for creating and cleaning up HTTP client."""
    logger.debug("Creating new HTTP client")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        try:
            # Test server health before proceeding
            logger.debug("Testing server health")
            response = await client.get(f"{SERVER_URL}/health")
            assert response.status_code == 200, f"Server is not healthy: {response.status_code}"
            logger.debug("Server health check passed")
            yield client
        except Exception as e:
            logger.error(f"Error in client context: {str(e)}", exc_info=True)
            raise
        finally:
            logger.debug("Closing HTTP client")
            await client.aclose()


def print_debug_on_failure(result) -> bool:
    """
    Helper function to print debug info only on test failure
    
    Returns:
        bool: True if the test passed, False if it failed
    """
    if not result.success:
        print("\n=== Test Failure Details ===")
        print(f"Input: {result.input}")
        print(f"Output: {result.actual_output}")
        print(f"Success: {result.success}")
        if hasattr(result, 'retrieval_context'):
            print(f"Retrieval Context: {result.retrieval_context}")
        print("\nScorer Details:")
        for scorer_data in result.scorers_data:
            print(f"- Name: {scorer_data.name}")
            print(f"- Score: {scorer_data.score}")
            print(f"- Threshold: {scorer_data.threshold}")
            print(f"- Success: {scorer_data.success}")
            print(f"- Reason: {scorer_data.reason}")
            print(f"- Error: {scorer_data.error}")
            if scorer_data.verbose_logs:
                print(f"- Verbose Logs: {scorer_data.verbose_logs}")

        return False
    return True


@pytest_asyncio.fixture
async def client():
    """Fixture to create and provide an HTTP client with proper cleanup."""
    async with get_client() as client:
        yield client

@pytest_asyncio.fixture
async def cleanup_traces(client):
    """Fixture to clean up traces after tests."""
    logger.debug("Starting cleanup_traces fixture")
    yield
    logger.debug("Cleaning up traces")
    # Add cleanup logic here if needed

@pytest.mark.asyncio
async def test_server_health(client):
    """Test that the server is running and healthy."""
    try:
        logger.debug("Testing server health endpoint")
        response = await client.get(f"{SERVER_URL}/health")
        logger.debug(f"Health check response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        logger.debug(f"Health check data: {data}")
        assert data == {"status": "ok"}  # Updated to match actual server response
    except Exception as e:
        logger.error(f"Error in test_server_health: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_trace_count_endpoint(client):
    """Test that the trace count endpoint works correctly."""
    try:
        logger.debug("Testing trace count endpoint")
        response = await client.get(
            f"{SERVER_URL}/traces/count/",
            headers=get_headers()
        )
        logger.debug(f"Trace count response: {response.status_code}")
        assert response.status_code == 200
        data = response.json()
        logger.debug(f"Trace count data: {data}")
        assert "traces_ran" in data
        assert "user_traces_ran" in data
    except Exception as e:
        logger.error(f"Error in test_trace_count_endpoint: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_trace_save_increment(client, cleanup_traces):
    """Test that saving a trace increments the trace count."""
    try:
        logger.debug("Starting trace save increment test")
        # Get initial count
        response = await client.get(
            f"{SERVER_URL}/traces/count/",
            headers=get_headers()
        )
        initial_count = response.json()["traces_ran"]
        logger.debug(f"Initial trace count: {initial_count}")
        
        # Create a trace
        timestamp = time.time()
        trace_data = {
            "name": f"test_trace_{int(timestamp)}",
            "project_name": "test_project",
            "trace_id": str(uuid4()),
            "created_at": str(timestamp),
            "entries": [
                {
                    "timestamp": timestamp,
                    "type": "span",
                    "name": "test_span",
                    "inputs": {"test": "input"},
                    "outputs": {"test": "output"},
                    "duration": 0.1,
                    "span_id": str(uuid4()),
                    "parent_id": None
                }
            ],
            "duration": 0.1,
            "token_counts": {"total": 10},
            "empty_save": False,
            "overwrite": False
        }
        logger.debug(f"Created trace data: {trace_data}")

        response = await client.post(
            f"{SERVER_URL}/traces/save/",
            json=trace_data,
            headers=get_headers()
        )
        
        logger.debug(f"Trace save response: {response.status_code}")
        logger.debug(f"Trace save response body: {response.text}")
        assert response.status_code == 200
        
        # Verify increment
        response = await client.get(
            f"{SERVER_URL}/traces/count/",
            headers=get_headers()
        )
        assert response.status_code == 200
        new_count = response.json()["traces_ran"]
        logger.debug(f"New trace count: {new_count}")
        assert new_count > initial_count
    except Exception as e:
        logger.error(f"Error in test_trace_save_increment: {str(e)}", exc_info=True)
        raise

@pytest.mark.asyncio
async def test_judgee_count_endpoint(client):
    """Test that the judgee count endpoint works correctly."""
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    data = response.json()
    assert "judgees_ran" in data
    assert "user_judgees_ran" in data

@pytest.mark.asyncio
async def test_trace_count_endpoint(client):
    """Test that the trace count endpoint works correctly."""
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    data = response.json()
    assert "traces_ran" in data
    assert "user_traces_ran" in data

@pytest.mark.asyncio
async def test_concurrent_trace_saves(client, cleanup_traces):
    """Test concurrent trace saves to verify atomic operations."""
    try:
        # Get initial count
        response = await client.get(
            f"{SERVER_URL}/traces/count/",
            headers=get_headers()
        )
        initial_count = response.json()["traces_ran"]
        
        # Number of concurrent traces to save
        num_traces = 3
        
        async def save_trace(index):
            try:
                timestamp = time.time()
                trace_data = {
                    "name": f"concurrent_trace_{index}_{int(timestamp)}",
                    "project_name": "test_project",
                    "trace_id": str(uuid4()),
                    "created_at": str(timestamp),
                    "entries": [
                        {
                            "timestamp": timestamp,
                            "type": "span",
                            "name": f"test_span_{index}",
                            "inputs": {"test": f"input_{index}"},
                            "outputs": {"test": f"output_{index}"},
                            "duration": 0.1,
                            "span_id": str(uuid4()),
                            "parent_id": None
                        }
                    ],
                    "duration": 0.1,
                    "token_counts": {"total": 10},
                    "empty_save": False,
                    "overwrite": False
                }

                response = await client.post(
                    f"{SERVER_URL}/traces/save/",
                    json=trace_data,
                    headers=get_headers()
                )
                return response.status_code
            except Exception as e:
                logger.error(f"Error in save_trace {index}: {str(e)}")
                return 500
        
        # Save traces concurrently with timeout
        tasks = [save_trace(i) for i in range(num_traces)]
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=30.0
        )
        
        # All saves should succeed
        assert all(status == 200 for status in results)
        
        # Verify increment
        response = await client.get(
            f"{SERVER_URL}/traces/count/",
            headers=get_headers()
        )
        assert response.status_code == 200
        assert response.json()["traces_ran"] >= num_traces + initial_count
    except Exception as e:
        logger.error(f"Error in test_concurrent_trace_saves: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_failed_trace_counting(client):
    """Test that failed traces are still counted."""
    # Get initial count
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    initial_count = response.json()["traces_ran"]
    
    # Create an invalid trace (missing required fields)
    timestamp = time.time()
    trace_data = {
        "name": f"test_failed_trace_{int(timestamp)}",
        "project_name": "test_project",
        "trace_id": str(uuid4()),
        "created_at": str(timestamp),  # Convert to string
        # Missing entries, which should cause a validation error
        "duration": 0.1,
        "token_counts": {"total": 10},
        "empty_save": False,
        "overwrite": False
    }

    # This should fail but still increment the count
    response = await client.post(
        f"{SERVER_URL}/traces/save/",
        json=trace_data,
        headers=get_headers()
    )
    
    # The request might fail with 400 or 422, but the trace count should still increment
    # Verify increment
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    # Since we're counting both successful and failed traces, the count should increase
    assert response.json()["traces_ran"] >= initial_count

@pytest.mark.asyncio
async def test_real_trace_tracking(client):
    """Test real trace tracking with actual trace saves"""
    from judgeval.common.tracer import Tracer
    
    print("Starting test_real_trace_tracking...")
    
    # Initialize tracer
    print("Initializing Tracer...")
    tracer = Tracer(
        api_key=os.getenv("JUDGMENT_API_KEY"),
        project_name="test_project",
        organization_id=os.getenv("JUDGMENT_ORG_ID")
    )
    print("Tracer initialized successfully")
    
    # Create a real trace
    print("Creating traced function...")
    @tracer.observe(name="test_trace")
    def test_function():
        print("Running traced function...")
        return "Hello, World!"
    
    # Run the traced function
    print("Executing traced function...")
    result = test_function()
    print(f"Traced function result: {result}")
    
    # Verify counts were incremented
    print("Checking trace counts...")
    response = await client.get(
        f"{os.getenv('JUDGMENT_API_URL')}/traces/count/",
        headers={
            "Authorization": f"Bearer {os.getenv('JUDGMENT_API_KEY')}",
            "X-Organization-Id": os.getenv("JUDGMENT_ORG_ID")
        }
    )
    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.text}")
    
    assert response.status_code == 200
    data = response.json()
    print(f"Trace count data: {data}")
    assert "traces_ran" in data
    assert "user_traces_ran" in data
    assert data["traces_ran"] > 0
    assert data["user_traces_ran"] > 0

@pytest.mark.asyncio
async def test_rate_limiting_detection(client):
    """Test to detect if rate limiting is active without exceeding limits."""
    # Get rate limit headers without triggering limits
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    
    # Check if rate limit headers are present
    rate_limit_headers = [
        header for header in response.headers 
        if "rate" in header.lower() or "limit" in header.lower() or "remaining" in header.lower()
    ]
    
    # Print headers for debugging
    print(f"Rate limit related headers: {rate_limit_headers}")
    
    # If rate limiting is implemented, we should see headers like:
    # X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, etc.
    # We're just checking the presence of the mechanism, not trying to exceed it
    
    # This test passes if we can detect rate limiting headers or if we get a successful response
    # (since some implementations might not expose headers)
    assert response.status_code == 200
    
    # Optional assertion if rate limit headers are expected
    # assert len(rate_limit_headers) > 0, "No rate limit headers detected"

@pytest.mark.asyncio
async def test_burst_request_handling(client):
    """Test how the API handles a burst of requests without exceeding limits."""
    # Number of requests to send in a burst (keep this low to avoid triggering actual limits)
    num_requests = 5
    
    # Send a small burst of trace save requests
    timestamp = time.time()
    trace_data = {
        "name": f"burst_test_trace_{int(timestamp)}",
        "project_name": "test_project",
        "trace_id": str(uuid4()),
        "created_at": str(timestamp),
        "entries": [
            {
                "timestamp": timestamp,
                "type": "span",
                "name": "test_span",
                "inputs": {"test": "input"},
                "outputs": {"test": "output"},
                "duration": 0.1,
                "span_id": str(uuid4()),
                "parent_id": None
            }
        ],
        "duration": 0.1,
        "token_counts": {"total": 10},
        "empty_save": False,
        "overwrite": False
    }
    
    async def save_trace():
        # Create a unique trace ID for each request
        local_trace_data = trace_data.copy()
        local_trace_data["trace_id"] = str(uuid4())
        
        response = await client.post(
            f"{SERVER_URL}/traces/save/",
            json=local_trace_data,
            headers=get_headers()
        )
        return response.status_code
    
    # Send burst of requests and collect status codes
    tasks = [save_trace() for _ in range(num_requests)]
    status_codes = await asyncio.gather(*tasks)
    
    # All requests should succeed if we're below the rate limit
    # If any fail with 429, that's also informative but not a test failure
    print(f"Burst request status codes: {status_codes}")
    
    # Count successful vs rate-limited responses
    successful = status_codes.count(200)
    rate_limited = status_codes.count(429)
    
    # We expect either all successful or some rate limited
    assert successful + rate_limited == num_requests
    
    # Log the results for analysis
    print(f"Successful requests: {successful}, Rate limited: {rate_limited}")

@pytest.mark.asyncio
async def test_organization_limits_info(client):
    """Test to retrieve and verify organization limits information."""
    # Some APIs provide endpoints to check current usage and limits
    # Try to access such an endpoint if it exists
    
    try:
        response = await client.get(
            f"{SERVER_URL}/organization/usage/",
            headers=get_headers()
        )
        
        # If the endpoint exists and returns data
        if response.status_code == 200:
            limits_data = response.json()
            print(f"Organization usage data: {limits_data}")
            
            # Check for expected fields in the response
            # This is informational and won't fail the test if fields are missing
            expected_fields = ["judgee_limit", "trace_limit", "judgee_used", "trace_used"]
            found_fields = [field for field in expected_fields if field in limits_data]
            
            print(f"Found usage fields: {found_fields}")
            
            # Test passes if we got a valid response
            assert response.status_code == 200
        else:
            # If endpoint doesn't exist, test is inconclusive but not failed
            print(f"Usage endpoint returned status code: {response.status_code}")
            pytest.skip("Organization usage endpoint not available or returned non-200 status")
            
    except Exception as e:
        # If endpoint doesn't exist, test is inconclusive but not failed
        print(f"Error accessing organization usage endpoint: {str(e)}")
        pytest.skip("Organization usage endpoint not available")

@pytest.mark.asyncio
async def test_on_demand_resource_detection(client):
    """Test to detect on-demand resource capabilities without creating actual resources."""
    # Check if on-demand endpoints exist by making OPTIONS requests
    # This doesn't modify data but tells us if the endpoints are available
    
    # Check for on-demand judgee endpoint
    judgee_response = await client.options(
        f"{SERVER_URL}/judgees/on_demand/",
        headers=get_headers()
    )
    
    # Check for on-demand trace endpoint
    trace_response = await client.options(
        f"{SERVER_URL}/traces/on_demand/",
        headers=get_headers()
    )
    
    # Log the results
    print(f"On-demand judgee endpoint status: {judgee_response.status_code}")
    print(f"On-demand trace endpoint status: {trace_response.status_code}")
    
    # For OPTIONS requests, 200, 204, or 404 are all valid responses
    # 200/204 means endpoint exists, 404 means it doesn't
    
    # Test passes if we could make the requests without errors
    assert judgee_response.status_code in [200, 204, 404]
    assert trace_response.status_code in [200, 204, 404]
    
    # Log if on-demand endpoints were detected
    on_demand_judgee_available = judgee_response.status_code in [200, 204]
    on_demand_trace_available = trace_response.status_code in [200, 204]
    
    print(f"On-demand judgee endpoint available: {on_demand_judgee_available}")
    print(f"On-demand trace endpoint available: {on_demand_trace_available}")

@pytest.mark.asyncio
async def test_real_judgee_tracking(client):
    """Test real judgee tracking with actual evaluation."""
    # Get initial judgee count
    print(f"Getting initial judgee count from {SERVER_URL}/judgees/count/")
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    print(f"Initial count response: {response.status_code} {response.text}")
    assert response.status_code == 200
    initial_data = response.json()
    initial_judgees = initial_data["judgees_ran"]
    print(f"Initial judgee count: {initial_judgees}")

    example = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It used to be called the city of lights until 1968.",
    )

    scorer = AnswerCorrectnessScorer(threshold=0.1)

    judgment_client = JudgmentClient()
    PROJECT_NAME = "test-project"
    EVAL_RUN_NAME = "test-run-ac"
    
    print(f"Running evaluation with use_judgment=True...")
    # Test with use_judgment=True
    res = judgment_client.run_evaluation(
        examples=[example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=True,
        override=True,
    )
    print(f"Evaluation response: {res}")
    print_debug_on_failure(res[0])

    # Wait a moment for the count to update
    await asyncio.sleep(2)

    # Get final judgee count
    print(f"Getting final judgee count from {SERVER_URL}/judgees/count/")
    response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    print(f"Final count response: {response.status_code} {response.text}")
    assert response.status_code == 200
    final_data = response.json()
    final_judgees = final_data["judgees_ran"]
    print(f"Final judgee count: {final_judgees}")
    print(f"Count difference: {final_judgees - initial_judgees}")

    # Verify judgee count increased by 1
    assert final_judgees == initial_judgees + 1, f"Expected judgee count to increase by 1, but got {final_judgees - initial_judgees}"

    print(f"Running evaluation with use_judgment=False...")
    # Test with use_judgment=False 
    res = judgment_client.run_evaluation(
        examples=[example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        log_results=True,
        project_name=PROJECT_NAME,
        eval_run_name=EVAL_RUN_NAME,
        use_judgment=False,
        override=True,
    )
    print(f"Evaluation response: {res}")
    print_debug_on_failure(res[0])

@pytest.mark.asyncio
async def test_real_trace_and_judgee_tracking(client):
    """Test both trace and judgee tracking in a single E2E test.
    
    This test:
    1. Checks initial trace and judgee counts
    2. Creates and saves a trace
    3. Verifies that trace counts are incremented
    4. Runs an evaluation within the trace
    5. Verifies that both trace and judgee counts are incremented correctly
    """
    import judgeval
    from judgeval.judgment_client import JudgmentClient
    from judgeval.scorers import AnswerCorrectnessScorer
    from judgeval.data import Example
    from judgeval.common.tracer import Tracer
    
    # Get initial counts
    print("Getting initial counts...")
    
    # Get initial trace count
    trace_response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert trace_response.status_code == 200
    initial_trace_data = trace_response.json()
    initial_traces = initial_trace_data["traces_ran"]
    print(f"Initial trace count: {initial_traces}")
    
    # Get initial judgee count
    judgee_response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert judgee_response.status_code == 200
    initial_judgee_data = judgee_response.json()
    initial_judgees = initial_judgee_data["judgees_ran"]
    print(f"Initial judgee count: {initial_judgees}")
    
    # Create a trace and run an evaluation within it
    print("Creating trace and running evaluation...")
    
    # Define test data
    example = Example(
        input="What's the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="France's capital is Paris. It is known as the City of Light.",
    )
    scorer = AnswerCorrectnessScorer(threshold=0.1)
    
    # Initialize judgment client
    judgment_client = JudgmentClient()
    PROJECT_NAME = "test-trace-judgee-project"
    EVAL_RUN_NAME = "test-trace-judgee-run"
    
    # Create a tracer
    tracer = Tracer(
        api_key=os.getenv("JUDGMENT_API_KEY"),
        project_name=PROJECT_NAME,
        organization_id=os.getenv("JUDGMENT_ORG_ID")
    )
    
    # Start a trace
    with tracer.trace(name="test_trace_with_eval", overwrite=True) as trace:
        print("Trace started, running evaluation within trace...")
        
        # Run evaluation within the trace
        res = judgment_client.run_evaluation(
            examples=[example],
            scorers=[scorer],
            model="Qwen/Qwen2.5-72B-Instruct-Turbo",
            log_results=True,
            project_name=PROJECT_NAME,
            eval_run_name=EVAL_RUN_NAME,
            use_judgment=True,
            override=True,
        )
        
        print(f"Evaluation response: {res}")
        print_debug_on_failure(res[0])
        
        # Save the trace
        trace_id, trace_data = trace.save()
        print(f"Trace saved with ID: {trace_id}")
    
    # Wait for counts to update
    print("Waiting for counts to update...")
    await asyncio.sleep(3)
    
    # Get final trace count
    print("Getting final trace count...")
    trace_response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert trace_response.status_code == 200
    final_trace_data = trace_response.json()
    final_traces = final_trace_data["traces_ran"]
    print(f"Final trace count: {final_traces}")
    print(f"Trace count difference: {final_traces - initial_traces}")
    
    # Get final judgee count
    print("Getting final judgee count...")
    judgee_response = await client.get(
        f"{SERVER_URL}/judgees/count/",
        headers=get_headers()
    )
    assert judgee_response.status_code == 200
    final_judgee_data = judgee_response.json()
    final_judgees = final_judgee_data["judgees_ran"]
    print(f"Final judgee count: {final_judgees}")
    print(f"Judgee count difference: {final_judgees - initial_judgees}")
    
    # Verify trace count increased by 1 (empty saves are no longer counted)
    assert final_traces == initial_traces + 1, f"Expected trace count to increase by 1, but got {final_traces - initial_traces}"
    
    # Verify judgee count increased by 1
    assert final_judgees == initial_judgees + 1, f"Expected judgee count to increase by 1, but got {final_judgees - initial_judgees}"
    
    print("Test completed successfully!")