from judgeval.utils.guards import (
    expect_exists,
    expect_api_key,
    expect_organization_id,
    expect_project_id,
)


class TestExpectExists:
    def test_returns_value_when_present(self):
        result = expect_exists("test_value", "error message", default=None)
        assert result == "test_value"

    def test_returns_default_when_none(self):
        result = expect_exists(None, "error message", default="default_value")
        assert result == "default_value"

    def test_returns_default_when_empty_string(self):
        result = expect_exists("", "error message", default="default_value")
        assert result == "default_value"

    def test_returns_default_when_zero(self):
        result = expect_exists(0, "error message", default=42)
        assert result == 42


class TestExpectApiKey:
    def test_returns_api_key_when_present(self):
        result = expect_api_key("test_api_key")
        assert result == "test_api_key"

    def test_returns_none_when_missing(self):
        result = expect_api_key(None)
        assert result is None


class TestExpectOrganizationId:
    def test_returns_org_id_when_present(self):
        result = expect_organization_id("test_org_id")
        assert result == "test_org_id"

    def test_returns_none_when_missing(self):
        result = expect_organization_id(None)
        assert result is None


class TestExpectProjectId:
    def test_returns_project_id_when_present(self):
        result = expect_project_id("test_project_id")
        assert result == "test_project_id"

    def test_returns_none_when_missing(self):
        result = expect_project_id(None)
        assert result is None

    def test_returns_none_when_empty_string(self):
        result = expect_project_id("")
        assert result is None

    def test_logs_caller_name_when_missing(self, caplog):
        import logging

        def my_function():
            return expect_project_id(None)

        with caplog.at_level(logging.ERROR):
            result = my_function()

        assert result is None
        assert "project_id is not set" in caplog.text
        assert "my_function()" in caplog.text
