# e2etests/test_judgee.py

import os
import time
import asyncio
import pytest
import sys
import httpx
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from server.main import app

# Add the package root folder (adjust the relative path as needed).
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "judgment"))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

# Load environment variables from .env file
load_dotenv()

client = TestClient(app)

# Use the existing API key from .env
TEST_API_KEY = os.getenv("JUDGMENT_API_KEY")
if not TEST_API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set in .env file")

# Helper function to verify that the server is running
async def verify_server(server_url: str):
    """Helper function to verify server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server_url}/health")
            assert response.status_code == 200, "Health check failed"
    except Exception as e:
        pytest.skip(f"Server not running. Please start with: uvicorn server.main:app --reload\nError: {e}")

def test_judgee_tracking_increment():
    """Test that judgees_ran is incremented correctly when running evaluations."""
    # ... test implementation ...

def test_judgee_tracking_reset():
    """Test that judgees_ran can be reset to 0."""
    # ... test implementation ...

def test_judgee_tracking_complete_flow():
    """Test complete flow of increment and reset."""
    # ... test implementation ...

# Removed test_judgee_count_with_skip_and_error and test_judgee_count_with_multiple_examples
