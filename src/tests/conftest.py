from __future__ import annotations

import random
import string

import pytest
from unittest.mock import MagicMock

from judgeval.data.example import Example


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def sample_examples():
    return [
        Example.create(input=f"question {i}", output=f"answer {i}") for i in range(3)
    ]


@pytest.fixture
def project_name():
    return "test-project"


@pytest.fixture
def random_name():
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    return f"test-{suffix}"


@pytest.fixture
def eval_run_name(random_name):
    return f"eval-run-{random_name}"
