from judgeval.utils.url import url_for


def test_basic_path():
    assert (
        url_for("/health", base="https://api.example.com")
        == "https://api.example.com/health"
    )


def test_base_with_trailing_slash():
    assert (
        url_for("health", base="https://api.example.com/")
        == "https://api.example.com/health"
    )


def test_deep_path():
    assert (
        url_for("/v1/spans", base="https://api.example.com")
        == "https://api.example.com/v1/spans"
    )


def test_absolute_path_replaces_base_path():
    result = url_for("/otel/v1/traces", base="https://api.example.com/prefix/")
    assert result == "https://api.example.com/otel/v1/traces"


def test_relative_path_joins_with_base():
    result = url_for("otel/v1/traces", base="https://api.example.com/prefix/")
    assert result == "https://api.example.com/prefix/otel/v1/traces"
