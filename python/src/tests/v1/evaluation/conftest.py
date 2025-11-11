import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_evaluation_client():
    return MagicMock()
