import logging

from judgeval.utils.guards import (
    expect_exists,
    expect_api_key,
    expect_organization_id,
    expect_project_id,
)


def test_expect_exists_returns_value_when_present():
    assert expect_exists("hello", "msg", "default") == "hello"


def test_expect_exists_returns_default_when_none():
    assert expect_exists(None, "missing", "fallback") == "fallback"


def test_expect_exists_returns_default_for_empty_string():
    assert expect_exists("", "empty", "fallback") == "fallback"


def test_expect_exists_logs_on_missing(caplog):
    with caplog.at_level(logging.ERROR, logger="judgeval"):
        expect_exists(None, "custom error message", "x")
    assert any("custom error message" in r.message for r in caplog.records)


def test_expect_api_key_returns_key():
    assert expect_api_key("my-key") == "my-key"


def test_expect_api_key_returns_none_when_missing():
    assert expect_api_key(None) is None


def test_expect_api_key_logs_on_missing(caplog):
    with caplog.at_level(logging.ERROR, logger="judgeval"):
        expect_api_key(None)
    assert any("JUDGMENT_API_KEY" in r.message for r in caplog.records)


def test_expect_organization_id_returns_id():
    assert expect_organization_id("org-123") == "org-123"


def test_expect_organization_id_returns_none_when_missing():
    assert expect_organization_id(None) is None


def test_expect_organization_id_logs_on_missing(caplog):
    with caplog.at_level(logging.ERROR, logger="judgeval"):
        expect_organization_id(None)
    assert any("JUDGMENT_ORG_ID" in r.message for r in caplog.records)


def test_expect_project_id_returns_id():
    assert expect_project_id("proj-abc") == "proj-abc"


def test_expect_project_id_returns_none_when_missing():
    assert expect_project_id(None) is None


def test_expect_project_id_logs_caller_name(caplog):
    with caplog.at_level(logging.ERROR, logger="judgeval"):
        expect_project_id(None)
    assert any(
        "test_expect_project_id_logs_caller_name" in r.message for r in caplog.records
    )
