from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import ANY

import pytest
from typer.testing import CliRunner

from judgeval.cli import app
from judgeval.testing import EvaluatorParseError, parse_checks

HEADER = "from judgeval.testing import evaluator, select, classify, judge\nev = evaluator()\n"

CANONICAL = """\
from judgeval.testing import evaluator, select, classify, judge, passed, failed

ev = evaluator()


@ev.check("no_leak", gates=True)
def no_leak(case, trace):
    spans = select(trace, "llm")
    return classify(spans, "leaks a secret")


@ev.check("tool_sequence", applies="when_declared")
def tool_sequence(case, trace, settings):
    return passed() if settings["tools"] == trace.tools else failed()


@ev.check("tone", labels=["calm", "curt"], gates=False)
def tone(case, trace):
    return judge(trace, "Which tone?")


ev.metric("execution_time_ms", lambda case, trace: trace.duration_ms, rollup="p95", better="lower")
"""


def parsed(source: str) -> list[dict[str, Any]]:
    return [asdict(check) for check in parse_checks(dedent(source))]


def row(name: str, line_start: int, line_end: int, **overrides: Any) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "pass_fail",
        "gating": True,
        "applies": "always",
        "label_set": [],
        "metric": None,
        "primitives": [],
        "line_start": line_start,
        "line_end": line_end,
        "body_hash": ANY,
        **overrides,
    }


def hashes(source: str) -> dict[str, str]:
    return {check.name: check.body_hash for check in parse_checks(source)}


def test_canonical_evaluator():
    assert parsed(CANONICAL) == [
        row(
            "no_leak",
            6,
            9,
            primitives=["classify", "select"],
            body_hash="a3dadb4eb6e581cca43a331f394b7b00cc5781f1c24771b3b7ab18388ccee139",
        ),
        row(
            "tool_sequence",
            12,
            14,
            applies="when_declared",
            body_hash="82ea5014da6901338ce6b85970c380e6701a33942a3f39907a8a1bedd7c7d6a0",
        ),
        row(
            "tone",
            17,
            19,
            kind="label",
            gating=False,
            label_set=["calm", "curt"],
            primitives=["judge"],
            body_hash="6a24c34249518ea23e287800d9aaae90c8877dea6fce5fbcfb2d113e2cc757b0",
        ),
        row(
            "execution_time_ms",
            22,
            22,
            kind="number",
            gating=False,
            metric={"rollup": "p95", "better_when": "lower"},
            body_hash="256bacdf6dbe46904d7a991070b0afc45fd68650a741325cc3705322f6ee1ef0",
        ),
    ]


def test_label_check_keeps_declared_label_order():
    source = HEADER + dedent("""\
        @ev.check("tone", labels=("curt", "calm", "warm"))
        def tone(case, trace):
            return "calm"
    """)
    assert parsed(source) == [
        row("tone", 3, 5, kind="label", label_set=["curt", "calm", "warm"])
    ]


def test_extra_decorators_and_async_checks():
    source = HEADER + dedent("""\
        @retry
        @ev.check("slow", gates=False)
        async def slow(case, trace):
            return await classify(trace, "slow?")
    """)
    assert parsed(source) == [row("slow", 3, 6, gating=False, primitives=["classify"])]


@pytest.mark.parametrize(
    "options, gating, metric",
    [
        (
            'rollup="mean", better="higher"',
            False,
            {"rollup": "mean", "better_when": "higher"},
        ),
        (
            'rollup="p95", better="lower", budget=2000',
            True,
            {"rollup": "p95", "better_when": "lower", "budget": 2000},
        ),
        (
            'rollup="max", better="lower", budget=0.5, gates=False',
            False,
            {"rollup": "max", "better_when": "lower", "budget": 0.5},
        ),
        (
            'rollup="sum", better="higher", budget=-1, applies="when_declared"',
            True,
            {"rollup": "sum", "better_when": "higher", "budget": -1},
        ),
    ],
)
def test_metric_options(options: str, gating: bool, metric: dict[str, Any]):
    source = HEADER + f'ev.metric("cost", lambda case, trace: trace.cost, {options})\n'
    assert parsed(source) == [
        row(
            "cost",
            3,
            3,
            kind="number",
            gating=gating,
            applies="when_declared" if "when_declared" in options else "always",
            metric=metric,
        )
    ]


def test_metric_with_module_level_function_takes_its_primitives():
    source = HEADER + dedent("""\
        def judged_turns(case, trace):
            return len(select(trace, "turn"))

        ev.metric(
            "turns",
            judged_turns,
            rollup="p50",
            better="lower",
        )
    """)
    assert parsed(source) == [
        row(
            "turns",
            6,
            11,
            kind="number",
            gating=False,
            metric={"rollup": "p50", "better_when": "lower"},
            primitives=["select"],
        )
    ]


def test_primitives_resolve_through_imports_only():
    source = dedent("""\
        import judgeval.testing as jt
        import judgeval.testing
        from judgeval.testing import evaluator as make, judge as llm_judge

        ev = make()

        @ev.check("mixed")
        def mixed(case, trace):
            trace.select("not ours")
            classify("not imported")
            jt.select(trace)
            judgeval.testing.classify(trace)
            return llm_judge(trace)
    """)
    assert parsed(source) == [
        row("mixed", 7, 13, primitives=["classify", "judge", "select"])
    ]


def test_star_import():
    source = dedent("""\
        from judgeval.testing import *
        ev = evaluator()

        @ev.check("a")
        def a(case, trace):
            return judge(trace)
    """)
    assert parsed(source) == [row("a", 4, 6, primitives=["judge"])]


def test_source_without_an_evaluator_has_no_checks():
    assert parsed("def helper(case, trace):\n    return True\n") == []


def test_source_is_never_executed():
    source = (
        'raise SystemExit("executed")\n'
        + HEADER
        + dedent("""\
        @ev.check("a")
        def a(case, trace):
            raise SystemExit("executed")
    """)
    )
    assert parsed(source) == [row("a", 4, 6)]


@pytest.mark.parametrize(
    "body, line, message",
    [
        (
            'NAME = "a"\n@ev.check(NAME)\ndef a(case, trace): ...\n',
            4,
            "the check name must be a literal",
        ),
        (
            '@ev.check("Bad-Name")\ndef a(case, trace): ...\n',
            3,
            "check name must match [a-z][a-z0-9_]{0,99}, got 'Bad-Name'",
        ),
        (
            "@ev.check(42)\ndef a(case, trace): ...\n",
            3,
            "check name must match [a-z][a-z0-9_]{0,99}, got 42",
        ),
        (
            "@ev.check()\ndef a(case, trace): ...\n",
            3,
            "expected a name as positional arguments",
        ),
        (
            '@ev.check("a", "b")\ndef a(case, trace): ...\n',
            3,
            "expected a name as positional arguments",
        ),
        (
            'GATES = True\n@ev.check("a", gates=GATES)\ndef a(case, trace): ...\n',
            4,
            "gates= must be a literal",
        ),
        (
            '@ev.check("a", gates="yes")\ndef a(case, trace): ...\n',
            3,
            "gates must be True or False",
        ),
        (
            '@ev.check("a", applies="sometimes")\ndef a(case, trace): ...\n',
            3,
            "applies must be one of ['always', 'when_declared']",
        ),
        (
            '@ev.check("a", budget=3)\ndef a(case, trace): ...\n',
            3,
            "unknown option budget=; expected one of ['applies', 'gates', 'labels']",
        ),
        (
            '@ev.check("a", **opts)\ndef a(case, trace): ...\n',
            3,
            "**options are not allowed",
        ),
        (
            '@ev.check("a", labels=[])\ndef a(case, trace): ...\n',
            3,
            "labels must be a non-empty list of strings",
        ),
        (
            '@ev.check("a", labels=["x", 1])\ndef a(case, trace): ...\n',
            3,
            "labels must be a non-empty list of strings",
        ),
        (
            '@ev.check("a", labels=["x", "x"])\ndef a(case, trace): ...\n',
            3,
            "labels must be unique",
        ),
        (
            '@ev.check("a")\ndef a(case, trace): ...\n@ev.check("a")\ndef b(case, trace): ...\n',
            5,
            "duplicate check name 'a' (first declared on line 3)",
        ),
        (
            '@ev.check("a")\n@ev.check("b")\ndef a(case, trace): ...\n',
            4,
            "one @check decorator per function",
        ),
        (
            "@ev.check\ndef a(case, trace): ...\n",
            3,
            '@ev.check needs a name: @ev.check("name")',
        ),
        (
            'if True:\n    @ev.check("a")\n    def a(case, trace): ...\n',
            4,
            "declare checks at module level as @ev.check(...) or ev.metric(...)",
        ),
        (
            'def a(case, trace): ...\nev.check("a")(a)\n',
            4,
            "declare checks at module level as @ev.check(...) or ev.metric(...)",
        ),
        (
            'ev.metric("m", lambda c, t: 1, better="lower")\n',
            3,
            "metric 'm' needs rollup= one of ['p50', 'p95', 'mean', 'sum', 'max']",
        ),
        (
            'ev.metric("m", lambda c, t: 1, rollup="p99", better="lower")\n',
            3,
            "rollup must be one of ['p50', 'p95', 'mean', 'sum', 'max']",
        ),
        (
            'ev.metric("m", lambda c, t: 1, rollup="p50", better="faster")\n',
            3,
            "better must be one of ['lower', 'higher']",
        ),
        (
            'ev.metric("m", lambda c, t: 1, rollup="p50", better="lower", gates=True)\n',
            3,
            "metric 'm' gates only with a budget=",
        ),
        (
            'ev.metric("m", lambda c, t: 1, rollup="p50", better="lower", budget=True)\n',
            3,
            "budget must be a number",
        ),
        (
            'ev.metric("m", lambda c, t: 1, rollup="p50", better="lower", labels=["a"])\n',
            3,
            "unknown option labels=; expected one of ['applies', 'better', 'budget', 'gates', 'rollup']",
        ),
        (
            'from x import f\nev.metric("m", f, rollup="p50", better="lower")\n',
            4,
            "metric 'm' takes a lambda or a function defined at module level",
        ),
        (
            'ev.metric("m", rollup="p50", better="lower")\n',
            3,
            "expected a name and a function as positional arguments",
        ),
        (
            "other = evaluator()\n",
            3,
            "an evaluator file declares one evaluator()",
        ),
        (
            "def build():\n    return evaluator()\n",
            4,
            "assign evaluator() to a name at module level: ev = evaluator()",
        ),
        (
            "def broken(:\n",
            3,
            "invalid Python: invalid syntax",
        ),
    ],
)
def test_parse_errors(body: str, line: int, message: str):
    with pytest.raises(EvaluatorParseError) as error:
        parse_checks(HEADER + body)
    assert (error.value.line, error.value.message) == (line, message)


def test_hash_ignores_changes_outside_the_check():
    original = HEADER + dedent("""\
        @ev.check("a")
        def a(case, trace):
            return select(trace, "x")

        @ev.check("b")
        def b(case, trace):
            return True
    """)
    edited = (
        "# a new leading comment\n\n"
        + original.replace("ev = evaluator()\n", "ev = evaluator()   \n\n\n")
        .replace('@ev.check("b")', '# about b\n@ev.check("b")')
        .replace("return True", "return True  # inside b, so b changes")
        + "\n# trailing comment\n"
    )
    before, after = hashes(original), hashes(edited)
    assert {name: after[name] == before[name] for name in before} == {
        "a": True,
        "b": False,
    }


def test_hash_ignores_trailing_whitespace_inside_the_check():
    original = HEADER + '@ev.check("a")\ndef a(case, trace):\n    return True\n'
    edited = HEADER + '@ev.check("a")  \ndef a(case, trace):\t\n    return True \n'
    assert hashes(edited) == hashes(original)


def test_hash_tracks_options_and_body():
    base = HEADER + '@ev.check("a")\ndef a(case, trace):\n    return True\n'
    variants = [
        base,
        base.replace('@ev.check("a")', '@ev.check("a", gates=False)'),
        base.replace("return True", "return False"),
    ]
    assert len({hashes(v)["a"] for v in variants}) == 3


def test_metric_hash_includes_its_function():
    source = HEADER + dedent("""\
        def turns(case, trace):
            return 1

        ev.metric("turns", turns, rollup="p50", better="lower")
    """)
    edited = source.replace("return 1", "return 2")
    assert hashes(edited)["turns"] != hashes(source)["turns"]


def test_output_is_stable_across_processes(tmp_path: Path):
    path = tmp_path / "evaluator.py"
    path.write_text(CANONICAL)

    def run(seed: str) -> str:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from judgeval.cli import app; app(sys.argv[1:])",
                "test",
                "checks",
                str(path),
            ],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    assert {run("1"), run("2")} == {json.dumps(parsed(CANONICAL), indent=2) + "\n"}


def test_cli_lists_checks(tmp_path: Path):
    path = tmp_path / "evaluator.py"
    path.write_text(CANONICAL)
    result = CliRunner().invoke(app, ["test", "checks", str(path)])
    assert (result.exit_code, json.loads(result.stdout)) == (0, parsed(CANONICAL))


def test_cli_reports_parse_errors_with_location(tmp_path: Path):
    path = tmp_path / "evaluator.py"
    path.write_text(HEADER + '@ev.check("a", gates=maybe)\ndef a(case, trace): ...\n')
    result = CliRunner().invoke(app, ["test", "checks", str(path)])
    assert (result.exit_code, result.stderr) == (
        1,
        f"{path}:3: gates= must be a literal\n",
    )


# A stand-in for the UI composer's code generator: generated code must parse back to the
# configuration it was generated from.
def render(config: dict[str, Any]) -> str:
    if config["kind"] == "number":
        metric = config["metric"]
        options = [f'rollup="{metric["rollup"]}"', f'better="{metric["better_when"]}"']
        if "budget" in metric:
            options.append(f"budget={metric['budget']!r}")
            if not config["gating"]:
                options.append("gates=False")
        if config["applies"] != "always":
            options.append(f'applies="{config["applies"]}"')
        return f'ev.metric("{config["name"]}", lambda case, trace: 0, {", ".join(options)})\n'
    options = [f'"{config["name"]}"']
    if not config["gating"]:
        options.append("gates=False")
    if config["applies"] != "always":
        options.append(f'applies="{config["applies"]}"')
    if config["label_set"]:
        options.append(f"labels={config['label_set']!r}")
    settings = ", settings" if config["applies"] == "when_declared" else ""
    return (
        f"@ev.check({', '.join(options)})\n"
        f"def {config['name']}(case, trace{settings}):\n"
        "    return True\n"
    )


ROUND_TRIP_CONFIGS = [
    {"name": "no_leak", "kind": "pass_fail", "gating": True, "applies": "always"},
    {"name": "style", "kind": "pass_fail", "gating": False, "applies": "when_declared"},
    {
        "name": "tone",
        "kind": "label",
        "gating": True,
        "applies": "always",
        "label_set": ["calm", "it's curt"],
    },
    {
        "name": "latency",
        "kind": "number",
        "gating": True,
        "applies": "always",
        "metric": {"rollup": "p95", "better_when": "lower", "budget": 1500.5},
    },
    {
        "name": "cost",
        "kind": "number",
        "gating": False,
        "applies": "when_declared",
        "metric": {"rollup": "sum", "better_when": "lower", "budget": 3},
    },
    {
        "name": "turns",
        "kind": "number",
        "gating": False,
        "applies": "always",
        "metric": {"rollup": "mean", "better_when": "higher"},
    },
]

CONFIG_FIELDS = ("name", "kind", "gating", "applies", "label_set", "metric")


@pytest.mark.parametrize("config", ROUND_TRIP_CONFIGS, ids=lambda c: c["name"])
def test_generated_check_round_trips(config: dict[str, Any]):
    expected = {"label_set": [], "metric": None, **config}
    [check] = parsed(HEADER + render(expected))
    assert {k: check[k] for k in CONFIG_FIELDS} == expected
