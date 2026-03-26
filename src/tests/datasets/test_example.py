from __future__ import annotations

import pytest

from judgeval.data.example import Example


class TestExample:
    def test_default_fields_set(self):
        e = Example()
        assert e.example_id
        assert e.created_at
        assert e.name is None
        assert e._properties == {}

    def test_create_factory(self):
        e = Example.create(input="q", output="a")
        assert e["input"] == "q"
        assert e["output"] == "a"

    def test_getitem(self):
        e = Example.create(key="val")
        assert e["key"] == "val"

    def test_getitem_missing_raises(self):
        e = Example()
        with pytest.raises(KeyError):
            _ = e["missing"]

    def test_contains(self):
        e = Example.create(x=1)
        assert "x" in e
        assert "y" not in e

    def test_to_dict_includes_properties(self):
        e = Example.create(input="hi", output="hello")
        d = e.to_dict()
        assert d["input"] == "hi"
        assert d["output"] == "hello"
        assert "example_id" in d
        assert "created_at" in d

    def test_to_dict_includes_name(self):
        e = Example(name="my-example")
        d = e.to_dict()
        assert d["name"] == "my-example"

    def test_properties_returns_copy(self):
        e = Example.create(x=1)
        props = e.properties
        props["x"] = 999
        assert e["x"] == 1

    def test_unique_example_ids(self):
        ids = {Example().example_id for _ in range(100)}
        assert len(ids) == 100

    def test_overwrite_property(self):
        e = Example.create(val=1)
        e._properties["val"] = 2
        assert e["val"] == 2
