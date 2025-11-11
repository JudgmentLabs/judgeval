import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_dataset_client():
    return MagicMock()
