from collections import defaultdict
from pydantic import BaseModel

from judgeval.utils.serialize import json_encoder


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
