from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from judgeval.agent_tests import AgentTestRunsFactory, AgentTestsFactory
from judgeval.exceptions import JudgmentAPIError, JudgmentConflictError

BASE = "https://api.example.com/v1/projects/p1/tests"


def _client() -> MagicMock:
    client = MagicMock()
    client.base_url = "https://api.example.com"
    client._request.return_value = {"ok": True}
    return client


def test_routes_each_call_to_its_v1_test_endpoint():
    client = _client()
    tests = AgentTestsFactory(client, "p1", "proj")
    runs = AgentTestRunsFactory(client, "p1", "proj")

    tests.create("t", labels=["a"])
    tests.list_cases("t1", version="draft")
    tests.update_case("t1", "c1", {"input": {}})
    tests.create_draft("t1")
    runs.start("t1", candidate="sha")
    runs.results("r1")

    assert [call.args for call in client._request.call_args_list] == [
        ("POST", BASE, {"name": "t", "labels": ["a"]}, None),
        (
            "GET",
            f"{BASE}/t1/cases",
            {},
            {"offset": 0, "limit": 100, "version": "draft"},
        ),
        ("PUT", f"{BASE}/t1/cases/c1", {"input": {}}, None),
        ("POST", f"{BASE}/t1/draft", {}, None),
        ("POST", f"{BASE}/t1/runs", {"candidate": "sha"}, None),
        ("GET", f"{BASE}/runs/r1/results", {}, None),
    ]


def test_maps_api_errors_to_sdk_errors():
    client = _client()
    client._request.side_effect = JudgmentAPIError(409, "exists", None)
    with pytest.raises(JudgmentConflictError):
        AgentTestsFactory(client, "p1", "proj").create("t")
