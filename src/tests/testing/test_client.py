from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval import testing
from judgeval.exceptions import JudgmentProjectNotFoundError


def _client(response=None):
    client = MagicMock()
    client.base_url = "https://api.example.com"
    client._request.return_value = response or {}
    return client


def test_start_posts_to_the_test_and_drops_unset_fields():
    client = _client({"run_id": "r1", "test_suite_version": 2, "cases": []})
    runs = testing.TestRunsFactory(client, "p1", "proj")

    run = runs.start("t1", candidate="abc123", stored_traces=True)

    assert run["run_id"] == "r1"
    client._request.assert_called_once_with(
        "POST",
        "https://api.example.com/v1/projects/p1/test-suites/t1/runs",
        {"candidate": "abc123", "stored_traces": True},
        None,
    )


def test_an_id_that_leaves_the_test_suites_route_is_rejected():
    client = _client()
    suites = testing.TestSuitesFactory(client, "p1", "proj")

    with pytest.raises(ValueError):
        suites.delete("../../projects/p2")
    client._request.assert_not_called()


def test_a_missing_project_raises_before_calling_the_api():
    client = _client()
    tests = testing.TestSuitesFactory(client, None, "proj")

    with pytest.raises(JudgmentProjectNotFoundError):
        tests.list()
    client._request.assert_not_called()
