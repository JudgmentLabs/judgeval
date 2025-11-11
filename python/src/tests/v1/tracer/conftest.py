import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_tracer_client():
    return MagicMock()
