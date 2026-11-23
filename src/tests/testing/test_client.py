from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from judgeval import testing
from judgeval.exceptions import JudgmentProjectNotFoundError
from judgeval.offline_tests import OFFLINE_TESTS_DEPRECATION


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


def test_offline_tests_warn_that_they_are_deprecated(monkeypatch):
    from judgeval.judgeval import Judgeval

    judgeval = Judgeval.__new__(Judgeval)
    monkeypatch.setattr(judgeval, "_internal_client", _client(), raising=False)
    monkeypatch.setattr(judgeval, "_project_id", "p1", raising=False)
    monkeypatch.setattr(judgeval, "_project_name", "proj", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        judgeval.offline_tests

    assert [str(w.message) for w in caught] == [OFFLINE_TESTS_DEPRECATION]


def test_canned_tool_result_returns_the_first_matching_rule():
    definition = {
        "tools": {
            "lookup_order": [
                {"match": {"order_id": "A1", "fields": ["status"]}, "return": None},
                {"match": {"order_id": "A1"}, "return": {"status": "shipped"}},
                {"return": {"error": "not found"}},
            ]
        }
    }

    def result(tool_name, arguments):
        return testing.canned_tool_result(definition, tool_name, arguments)

    assert [
        result("lookup_order", {"order_id": "A1", "fields": ["status"]}),
        result("lookup_order", {"order_id": "A1", "fields": []}),
        result("lookup_order", {"order_id": "B2"}),
        result("create_ticket", {"order_id": "A1"}),
    ] == [None, {"status": "shipped"}, {"error": "not found"}, testing.NO_MATCH]


def test_canned_tool_result_returns_a_copy_that_does_not_share_the_rules_objects():
    canned = {"orders": [{"id": "A1"}]}
    definition = {"tools": {"list_orders": [{"return": canned}]}}
    result = testing.canned_tool_result(definition, "list_orders", {})
    canned["orders"][0]["id"] = "changed"
    assert result == {"orders": [{"id": "A1"}]}


def test_link_and_unlink_pull_request_call_the_issue_routes():
    client = _client()
    issues = testing.IssuesFactory(client, "p1", "proj")
    issue = "0b6f6a1e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"

    issues.link_pull_request(
        issue, repo="acme/agent", number=7, url="https://github.com/acme/agent/pull/7"
    )
    issues.unlink_pull_request(issue, repo="acme/agent", number=7)

    url = f"https://api.example.com/v1/projects/p1/issues/{issue}/pull-requests"
    assert client._request.call_args_list == [
        (
            (
                "PUT",
                url,
                {
                    "repo": "acme/agent",
                    "number": 7,
                    "url": "https://github.com/acme/agent/pull/7",
                    "state": "open",
                },
                None,
            ),
        ),
        (("DELETE", url, {}, {"repo": "acme/agent", "number": 7}),),
    ]


def test_issue_id_from_description_reads_the_judgment_issue_line():
    issue = "0b6f6a1e-1c2d-4e5f-8a9b-0c1d2e3f4a5b"

    assert [
        testing.issue_id_from_description(f"Fix.\n\nJudgment-Issue: {issue}"),
        testing.issue_id_from_description(
            f"judgment-issue: https://app.judgmentlabs.ai/org/o/project/p/issues/{issue.upper()}\n"
        ),
        testing.issue_id_from_description(f"See Judgment-Issue: {issue}"),
        testing.issue_id_from_description(None),
    ] == [issue, issue, None, None]
