import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_instrumentation_tracer():
    tracer = MagicMock()
    tracer.get_tracer = MagicMock(return_value=MagicMock())
    return tracer
