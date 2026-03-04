import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_judge_client():
    return MagicMock()
