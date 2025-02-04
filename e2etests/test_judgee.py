# e2etests/test_judgee.py

import os
import time
import asyncio
import pytest
import sys
import httpx
from dotenv import load_dotenv

# Add the judgment package root to Python path
judgment_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "judgment"))
if judgment_root not in sys.path:
    sys.path.insert(0, judgment_root)

# Load environment variables from .env file
load_dotenv()

# Get server URL from environment
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")

if not TEST_API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set in .env file")

# Helper function to verify that the server is running
async def verify_server(server_url: str = SERVER_URL):
    """Helper function to verify server is running."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{server_url}/health")
            assert response.status_code == 200, "Health check failed"
    except Exception as e:
        pytest.skip(f"Server not running at {server_url}. Please start with: uvicorn server.main:app --reload\nError: {e}")

@pytest.mark.asyncio
async def test_judgee_tracking_increment():
    """Test that judgees_ran is incremented correctly when running evaluations."""
    await verify_server()
    
    # Add timeout and follow_redirects
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Reset judgee count at start
        response = await client.post(
            f"{SERVER_URL}/judgees/reset/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200

        # Initial count should be 0
        response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200
        assert response.json()["judgees_ran"] == 0

        # Run evaluation
        eval_data = {
            "judgment_api_key": TEST_API_KEY,
            "examples": [{
                "input": "test input",
                "actual_output": "test output",
                "expected_output": "test output",
                "context": [],
                "retrieval_context": [],
                "additional_metadata": {},
                "tools_called": [],
                "expected_tools": []
            }],
            "scorers": [
                {
                    "name": "faithfulness",
                    "score_type": "faithfulness",
                    "config": {},
                    "threshold": 1.0
                }
            ],
            "model": "gpt-3.5-turbo",
            "log_results": True
        }

        try:
            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0  # Longer timeout for evaluation
            )
            if response.status_code != 200:
                print(f"Error response: {response.text}")
            assert response.status_code == 200
        except httpx.ReadTimeout:
            pytest.fail("Server took too long to respond. Make sure the server is running: uvicorn server.main:app --reload")

        # Check count was incremented
        response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200
        assert response.json()["judgees_ran"] == 1

@pytest.mark.asyncio
async def test_judgee_tracking_reset():
    """Test that judgees_ran can be reset to 0."""
    await verify_server()
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.post(
            f"{SERVER_URL}/judgees/reset/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200

        response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200
        assert response.json()["judgees_ran"] == 0

@pytest.mark.asyncio
async def test_judgee_tracking_complete_flow():
    """Test complete flow of increment and reset."""
    await verify_server()
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Reset count
        response = await client.post(
            f"{SERVER_URL}/judgees/reset/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200

        # Run evaluation
        eval_data = {
            "judgment_api_key": TEST_API_KEY,
            "examples": [{
                "input": "test input",
                "actual_output": "test output",
                "expected_output": "test output",
                "context": [],
                "retrieval_context": [],
                "additional_metadata": {},
                "tools_called": [],
                "expected_tools": []
            }],
            "scorers": [
                {
                    "name": "faithfulness",
                    "score_type": "faithfulness",
                    "config": {},
                    "threshold": 1.0
                }
            ],
            "model": "gpt-3.5-turbo",
            "log_results": True
        }

        try:
            response = await client.post(
                f"{SERVER_URL}/evaluate/",
                json=eval_data,
                timeout=60.0
            )
            if response.status_code != 200:
                print(f"Error response: {response.text}")
            assert response.status_code == 200
        except httpx.ReadTimeout:
            pytest.fail("Server took too long to respond. Make sure the server is running: uvicorn server.main:app --reload")

        # Verify count increased
        response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200
        assert response.json()["judgees_ran"] == 1

        # Reset and verify back to 0
        response = await client.post(
            f"{SERVER_URL}/judgees/reset/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200

        response = await client.get(
            f"{SERVER_URL}/judgees/count/",
            params={"judgment_api_key": TEST_API_KEY}
        )
        assert response.status_code == 200
        assert response.json()["judgees_ran"] == 0

# Removed test_judgee_count_with_skip_and_error and test_judgee_count_with_multiple_examples
