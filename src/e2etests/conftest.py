"""
Shared fixtures and configuration for E2E tests.
"""

import os
import pytest
import random
import string
import logging
from dotenv import load_dotenv

from judgeval.judgment_client import JudgmentClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
SERVER_URL = os.getenv("JUDGMENT_API_URL", "http://localhost:8000")
API_KEY = os.getenv("JUDGMENT_API_KEY")
ORGANIZATION_ID = os.getenv("JUDGMENT_ORG_ID")

if not API_KEY:
    pytest.skip("JUDGMENT_API_KEY not set", allow_module_level=True)


@pytest.fixture(scope="session")
def client() -> JudgmentClient:
    """Create a single JudgmentClient instance for all tests."""
    # Setup
    client = JudgmentClient(judgment_api_key=API_KEY, organization_id=ORGANIZATION_ID)
    yield client
    # Teardown
    # Add more projects to delete as needed
    client.delete_project(project_name="test-project")
    client.delete_project(project_name="custom_judge_test")
    client.delete_project(project_name="test_project")
    client.delete_project(project_name="test_eval_run_naming_collisions")
    client.delete_project(project_name="ToneScorerTest")
    client.delete_project(project_name="sentiment_test")
    client.delete_project(project_name="rules_test")
    client.delete_project(project_name="rules-test-project")
    client.delete_project(project_name="TestingPoemBot")
    client.delete_project(project_name="TEST")
    client.delete_project(project_name="TEST2")
    client.delete_project(project_name="text2sql")
    client.delete_project(project_name="OutreachWorkflow")
    client.delete_project(project_name="test-langgraph-project")
    client.delete_project(project_name="test-trace-judgee-project")
    client.delete_project(project_name="TestTogetherStreamUsage")
    client.delete_project(project_name="TestAnthropicStreamUsage")
    client.delete_project(project_name="TestTokenAggregation")
    client.delete_project(project_name="TestAsyncStreamUsage")
    client.delete_project(project_name="TestSyncStreamUsage")
    client.delete_project(project_name="DeepTracingTest")
    client.delete_project(project_name="ResponseAPITest")
    client.delete_project(project_name="TestingPoemBotAsync")
    client.delete_project(project_name="TestGoogleResponseAPI")
    client.delete_project(project_name="TraceEvalProjectFromYAMLTest")
    client.delete_project(project_name="test_s3_trace_saving")
    client.delete_project(project_name="test-langgraph-project-sync")
    client.delete_project(project_name="test-langgraph-project-async")


@pytest.fixture
def random_name() -> str:
    """Generate a random name for test resources."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


def pytest_configure(config):
    """Add markers for test categories."""
    config.addinivalue_line(
        "markers", "basic: mark test as testing basic functionality"
    )
    config.addinivalue_line(
        "markers", "advanced: mark test as testing advanced features"
    )
    config.addinivalue_line("markers", "custom: mark test as testing custom components")
    config.addinivalue_line("markers", "traces: mark test as testing trace operations")
