from __future__ import annotations

import pytest

from judgeval.prompts.prompt import Prompt


def _make_prompt(template: str) -> Prompt:
    return Prompt(
        name="test-prompt",
        prompt=template,
        created_at="2024-01-01",
        tags=[],
        commit_id="c1",
    )


class TestPromptCompile:
    def test_compiles_placeholder(self):
        assert _make_prompt("Hello {{name}}!").compile(name="Ada") == "Hello Ada!"

    def test_preserves_literal_dollar_sign(self):
        p = _make_prompt("The price is $5 for {{item}}.")
        assert p.compile(item="apples") == "The price is $5 for apples."

    def test_preserves_doubled_dollar_and_brace_syntax(self):
        assert _make_prompt("Total: $$100").compile() == "Total: $$100"
        assert _make_prompt("JS template: ${x}").compile() == "JS template: ${x}"

    def test_compiles_placeholder_with_surrounding_whitespace(self):
        assert _make_prompt("Hello {{ name }}!").compile(name="Ada") == "Hello Ada!"

    def test_dollar_directly_before_placeholder(self):
        assert _make_prompt("Cost: ${{amount}}").compile(amount="5") == "Cost: $5"

    def test_missing_variable_raises_value_error(self):
        with pytest.raises(ValueError, match="Missing required variable: name"):
            _make_prompt("Hi {{name}}").compile()
