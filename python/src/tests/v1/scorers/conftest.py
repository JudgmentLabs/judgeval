import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_scorer_client():
    return MagicMock()
