import os
import pytest
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv("JUDGMENT_API_KEY")
BACKEND_URL = os.getenv("JUDGMENT_BACKEND_URL", "https://api.judgment.dev")

@pytest.mark.skip(reason="Waiting for judgee tracking endpoints to be deployed")
@pytest.mark.asyncio
async def test_judgee_tracking_basic():
    """Test basic judgee counting with successful scorers"""
    if not API_KEY:
        pytest.skip("JUDGMENT_API_KEY not set")

    async with httpx.AsyncClient() as client:
        # Reset count to start fresh
        await client.post(f"{BACKEND_URL}/judgees/reset/", params={"judgment_api_key": API_KEY})
        
        # Run evaluation with two scorers
        response = await client.post(
            f"{BACKEND_URL}/evaluate/",
            json={
                "examples": [{
                    "input": "What is 2+2?",
                    "expected_output": "4",
                }],
                "scorers": [
                    {"name": "exact_match", "threshold": 0.5},
                    {"name": "rouge", "threshold": 0.5}
                ],
                "model": "gpt-4",
                "judgment_api_key": API_KEY,
                "log_results": False
            }
        )
        assert response.status_code == 200

        # Verify count = 2 (one for each scorer)
        count_response = await client.get(
            f"{BACKEND_URL}/judgees/count/",
            params={"judgment_api_key": API_KEY}
        )
        count = count_response.json()["judgees_ran"]
        assert count == 2

@pytest.mark.skip(reason="Waiting for judgee tracking endpoints to be deployed")
@pytest.mark.asyncio
async def test_judgee_tracking_with_skipped_scorers():
    """Test that skipped scorers are not counted"""
    async with httpx.AsyncClient() as client:
        await client.post(f"{BACKEND_URL}/judgees/reset/", params={"judgment_api_key": API_KEY})
        
        response = await client.post(
            f"{BACKEND_URL}/evaluate/",
            json={
                "examples": [{
                    "input": "Test input",
                    "expected_output": "Test output",
                    "skip_scorers": ["rouge"]  # Skip rouge scorer
                }],
                "scorers": [
                    {"name": "exact_match", "threshold": 0.5},
                    {"name": "rouge", "threshold": 0.5}
                ],
                "model": "gpt-4",
                "judgment_api_key": API_KEY,
                "log_results": False
            }
        )
        assert response.status_code == 200

        # Verify only non-skipped scorer was counted
        count_response = await client.get(
            f"{BACKEND_URL}/judgees/count/",
            params={"judgment_api_key": API_KEY}
        )
        count = count_response.json()["judgees_ran"]
        assert count == 1  # Only exact_match ran

@pytest.mark.skip(reason="Waiting for judgee tracking endpoints to be deployed")
@pytest.mark.asyncio
async def test_judgee_tracking_multiple_examples():
    """Test counting with multiple examples"""
    async with httpx.AsyncClient() as client:
        await client.post(f"{BACKEND_URL}/judgees/reset/", params={"judgment_api_key": API_KEY})
        
        response = await client.post(
            f"{BACKEND_URL}/evaluate/",
            json={
                "examples": [
                    {
                        "input": "Example 1",
                        "expected_output": "Output 1"
                    },
                    {
                        "input": "Example 2",
                        "expected_output": "Output 2"
                    }
                ],
                "scorers": [
                    {"name": "exact_match", "threshold": 0.5}
                ],
                "model": "gpt-4",
                "judgment_api_key": API_KEY,
                "log_results": False
            }
        )
        assert response.status_code == 200

        # Verify count = 2 (1 scorer × 2 examples)
        count_response = await client.get(
            f"{BACKEND_URL}/judgees/count/",
            params={"judgment_api_key": API_KEY}
        )
        count = count_response.json()["judgees_ran"]
        assert count == 2

@pytest.mark.skip(reason="Waiting for judgee tracking endpoints to be deployed")
@pytest.mark.asyncio
async def test_judgee_reset():
    """Test resetting judgee count"""
    async with httpx.AsyncClient() as client:
        # Run an evaluation to get non-zero count
        await client.post(
            f"{BACKEND_URL}/evaluate/",
            json={
                "examples": [{"input": "Test", "expected_output": "Test"}],
                "scorers": [{"name": "exact_match", "threshold": 0.5}],
                "model": "gpt-4",
                "judgment_api_key": API_KEY,
                "log_results": False
            }
        )

        # Reset count
        reset_response = await client.post(
            f"{BACKEND_URL}/judgees/reset/",
            params={"judgment_api_key": API_KEY}
        )
        assert reset_response.status_code == 200

        # Verify count is 0
        count_response = await client.get(
            f"{BACKEND_URL}/judgees/count/",
            params={"judgment_api_key": API_KEY}
        )
        count = count_response.json()["judgees_ran"]
        assert count == 0 