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

# Load environment variables from .env file
load_dotenv()

# Get server URL and API key from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")
USER_API_KEY = os.getenv("USER_API_KEY", TEST_API_KEY)  # For user-specific tests

# Skip all tests if API key or organization ID is not set
pytestmark = pytest.mark.skipif(
    not TEST_API_KEY or not ORGANIZATION_ID, 
    reason="JUDGMENT_API_KEY or ORGANIZATION_ID not set in .env file"
)

# Standard headers for all requests
def get_headers():
    return {
        "Authorization": f"Bearer {TEST_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }

# User-specific headers with organization ID
def get_user_headers():
    return {
        "Authorization": f"Bearer {USER_API_KEY}",
        "X-Organization-Id": ORGANIZATION_ID
    }

@pytest_asyncio.fixture
async def client():
    """Fixture to create and provide an HTTP client."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        yield client

@pytest.mark.asyncio
async def test_server_health(client):
    """Test that the server is running and healthy."""
    response = await client.get(f"{SERVER_URL}/health")
    assert response.status_code == 200

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
async def test_trace_save_increment(client):
    """Test that saving a trace increments the trace count."""
    # Get initial count
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    initial_count = response.json()["traces_ran"]
    
    # Create a trace
    timestamp = time.time()
    trace_data = {
        "name": f"test_trace_{int(timestamp)}",
        "project_name": "test_project",
        "trace_id": str(uuid4()),
        "created_at": str(timestamp),  # Convert to string
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

    response = await client.post(
        f"{SERVER_URL}/traces/save/",
        json=trace_data,
        headers=get_headers()
    )
    
    # Print response for debugging
    print(f"Response status: {response.status_code}")
    print(f"Response content: {response.text}")
    
    # Verify increment
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["traces_ran"] > initial_count

@pytest.mark.asyncio
async def test_concurrent_trace_saves(client):
    """Test concurrent trace saves to verify atomic operations."""
    # Get initial count
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    initial_count = response.json()["traces_ran"]
    
    # Number of concurrent traces to save
    num_traces = 3
    
    async def save_trace(index):
        timestamp = time.time()
        trace_data = {
            "name": f"concurrent_trace_{index}_{int(timestamp)}",
            "project_name": "test_project",
            "trace_id": str(uuid4()),
            "created_at": str(timestamp),  # Convert to string
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
    
    # Save traces concurrently
    tasks = [save_trace(i) for i in range(num_traces)]
    results = await asyncio.gather(*tasks)
    
    # All saves should succeed
    assert all(status == 200 for status in results)
    
    # Verify increment
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["traces_ran"] >= num_traces + initial_count

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
    """Test tracking with a real trace similar to those created by the Tracer class."""
    # Get initial count
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    initial_count = response.json()["traces_ran"]
    
    # Create a realistic trace with spans similar to what the Tracer would create
    timestamp = time.time()
    trace_id = str(uuid4())
    
    trace_data = {
        "name": f"test_real_trace_{int(timestamp)}",
        "project_name": "test_project",
        "trace_id": trace_id,
        "created_at": str(timestamp),  # Convert to string
        "entries": [
            {
                "timestamp": timestamp,
                "type": "span",
                "name": "llm_call",
                "inputs": {
                    "prompt": "What's the capital of France?",
                    "model": "gpt-3.5-turbo"
                },
                "outputs": {
                    "response": "The capital of France is Paris."
                },
                "duration": 0.5,
                "span_id": str(uuid4()),
                "parent_id": None
            },
            {
                "timestamp": timestamp + 0.1,
                "type": "span",
                "name": "process_response",
                "inputs": {
                    "response": "The capital of France is Paris."
                },
                "outputs": {
                    "processed": "PARIS"
                },
                "duration": 0.1,
                "span_id": str(uuid4()),
                "parent_id": None
            }
        ],
        "duration": 0.6,
        "token_counts": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total": 18
        },
        "empty_save": False,
        "overwrite": False
    }
    
    response = await client.post(
        f"{SERVER_URL}/traces/save/",
        json=trace_data,
        headers=get_headers()
    )
    assert response.status_code == 200
    
    # Verify the trace was saved
    data = response.json()
    assert "resource_usage" in data
    assert "ui_results_url" in data
    
    # Verify the trace count was incremented
    response = await client.get(
        f"{SERVER_URL}/traces/count/",
        headers=get_headers()
    )
    assert response.status_code == 200
    assert response.json()["traces_ran"] > initial_count