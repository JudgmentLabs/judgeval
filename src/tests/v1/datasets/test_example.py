from judgeval.v1.data.example import Example


def test_example_initialization():
    example = Example()
    assert example.example_id is not None
    assert example.created_at is not None
    assert example.name is None
    assert example.properties == {}


def test_example_with_name():
    example = Example(name="test_example")
    assert example.name == "test_example"


def test_set_property():
    example = Example()
    example.set_property("input", "test input")
    assert example.get_property("input") == "test input"


def test_set_multiple_properties():
    example = Example()
    example.set_property("input", "test input")
    example.set_property("output", "test output")
    example.set_property("expected", "expected output")

    assert example.get_property("input") == "test input"
    assert example.get_property("output") == "test output"
    assert example.get_property("expected") == "expected output"


def test_get_property_nonexistent():
    example = Example()
    assert example.get_property("nonexistent") is None


def test_set_property_returns_self():
    example = Example()
    result = example.set_property("key", "value")
    assert result is example


def test_create_method():
    example = Example.create(
        input="test input", output="test output", context="test context"
    )

    assert isinstance(example, Example)
    assert example.get_property("input") == "test input"
    assert example.get_property("output") == "test output"
    assert example.get_property("context") == "test context"


def test_to_dict():
    example = Example(name="test_example")
    example.set_property("input", "test input")
    example.set_property("output", "test output")

    result = example.to_dict()

    assert "example_id" in result
    assert "created_at" in result
    assert result["name"] == "test_example"
    assert result["input"] == "test input"
    assert result["output"] == "test output"


def test_properties_returns_copy():
    example = Example()
    example.set_property("key", "value")

    props = example.properties
    props["key"] = "modified"

    assert example.get_property("key") == "value"


def test_properties_dict_contains_all_properties():
    example = Example()
    example.set_property("a", 1)
    example.set_property("b", 2)
    example.set_property("c", 3)

    props = example.properties
    assert len(props) == 3
    assert props["a"] == 1
    assert props["b"] == 2
    assert props["c"] == 3


def test_example_with_complex_data_types():
    example = Example()
    example.set_property("list_data", [1, 2, 3])
    example.set_property("dict_data", {"key": "value"})
    example.set_property("nested", {"list": [1, 2], "dict": {"a": "b"}})

    assert example.get_property("list_data") == [1, 2, 3]
    assert example.get_property("dict_data") == {"key": "value"}
    assert example.get_property("nested") == {"list": [1, 2], "dict": {"a": "b"}}


def test_to_dict_with_complex_data():
    example = Example(name="complex_example")
    example.set_property("list_data", [1, 2, 3])
    example.set_property("dict_data", {"key": "value"})

    result = example.to_dict()
    assert result["list_data"] == [1, 2, 3]
    assert result["dict_data"] == {"key": "value"}
