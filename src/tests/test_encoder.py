from collections import defaultdict
from pydantic import BaseModel

from judgeval.utils.serialize import json_encoder, safe_serialize


class SimpleModel(BaseModel):
    id: int
    name: str


def test_basic_serialization():
    data = {"a": 1, "b": "string", "c": [1, 2, 3]}
    result = json_encoder(data)
    assert result == data


def test_pydantic_model_serialization():
    model = SimpleModel(id=1, name="Test")
    result = json_encoder(model)
    assert result == {"id": 1, "name": "Test"}


def test_unserializable_builtin_function():
    result = json_encoder(print)
    assert isinstance(result, str)
    assert "built-in function print" in result


def test_unserializable_builtin_class():
    result = json_encoder(defaultdict)
    assert isinstance(result, str)
    assert "class" in result and "collections.defaultdict" in result


def test_function_wrapped_in_dict():
    obj = {"key": print}
    result = json_encoder(obj)
    assert isinstance(result["key"], str)
    assert "built-in function print" in result["key"]


def test_integer_keys_serialization():
    """Test that dictionaries with integer keys can be serialized with orjson.OPT_NON_STR_KEYS"""
    data = {1: "one", 2: "two", 3: "three"}
    result = safe_serialize(data)
    # Should serialize successfully and parse back
    import orjson
    parsed = orjson.loads(result)
    # orjson converts integer keys back to strings when parsing
    assert parsed == {"1": "one", "2": "two", "3": "three"}


def test_mixed_keys_serialization():
    """Test that dictionaries with mixed key types can be serialized"""
    data = {"str_key": "value1", 1: "value2", 2.5: "value3"}
    result = safe_serialize(data)
    # Should serialize successfully without errors
    assert isinstance(result, str)
    assert "str_key" in result
    assert "value1" in result


def test_nested_integer_keys():
    """Test nested dictionaries with integer keys"""
    data = {"outer": {1: "inner1", 2: "inner2"}, "list": [1, 2, 3]}
    result = safe_serialize(data)
    assert isinstance(result, str)
    import orjson
    parsed = orjson.loads(result)
    # Verify structure is preserved
    assert "outer" in parsed
    assert "list" in parsed
    assert parsed["list"] == [1, 2, 3]
