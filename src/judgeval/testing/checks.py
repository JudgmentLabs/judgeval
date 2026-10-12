"""Derive the checks an evaluator declares from its source, without importing or running it.

An evaluator declares checks with literal options only::

    from judgeval.testing import evaluator, classify

    ev = evaluator()

    @ev.check("no_leak", gates=True)
    def no_leak(case, trace): ...

    @ev.check("tone", labels=["calm", "curt"], applies="when_declared")
    def tone(case, trace, settings): ...

    ev.metric("latency_ms", lambda case, trace: trace.duration_ms,
              rollup="p95", better="lower", budget=2000)

Each parsed check matches one row of ``agent_test_checks``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeGuard, Union

SDK_MODULE = "judgeval.testing"
PRIMITIVES = frozenset({"select", "classify", "judge"})
APPLIES = ("always", "when_declared")
ROLLUPS = ("p50", "p95", "mean", "sum", "max")
BETTER = ("lower", "higher")
CHECK_NAME = re.compile(r"[a-z][a-z0-9_]{0,99}")

FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


class EvaluatorParseError(ValueError):
    def __init__(self, line: int, message: str):
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


@dataclass(frozen=True)
class Check:
    name: str
    kind: Literal["pass_fail", "label", "number"]
    gating: bool
    applies: Literal["always", "when_declared"]
    label_set: list[str]
    # Number checks only: {rollup, better_when, budget?}.
    metric: Optional[dict[str, Any]]
    primitives: list[str]
    line_start: int
    line_end: int
    body_hash: str


def parse_checks(source: str) -> list[Check]:
    """Return the evaluator's checks in source order; raise EvaluatorParseError on any non-literal option."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise EvaluatorParseError(e.lineno or 1, f"invalid Python: {e.msg}") from e
    return _Parser(source, tree).parse()


class _Parser:
    def __init__(self, source: str, tree: ast.Module):
        self.lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.tree = tree
        self.sdk_names: dict[str, str] = {}
        self.sdk_modules: set[str] = set()
        self.functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.declarations: set[int] = set()

    def parse(self) -> list[Check]:
        self._collect_imports()
        ev = self._find_evaluator()
        if ev is None:
            return []
        checks: list[Check] = []
        first_line: dict[str, int] = {}
        for stmt in self.tree.body:
            check = self._declaration(stmt, ev)
            if check is None:
                continue
            if check.name in first_line:
                raise EvaluatorParseError(
                    check.line_start,
                    f"duplicate check name '{check.name}' (first declared on line {first_line[check.name]})",
                )
            first_line[check.name] = check.line_start
            checks.append(check)
        for node in ast.walk(self.tree):
            if (
                _is_ev_attr(node, ev, ("check", "metric"))
                and id(node) not in self.declarations
            ):
                raise EvaluatorParseError(
                    node.lineno,
                    f"declare checks at module level as @{ev}.check(...) or {ev}.metric(...)",
                )
        return checks

    def _collect_imports(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == SDK_MODULE:
                for alias in node.names:
                    if alias.name == "*":
                        self.sdk_names.update(
                            (n, n) for n in PRIMITIVES | {"evaluator"}
                        )
                    else:
                        self.sdk_names[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == SDK_MODULE:
                        self.sdk_modules.add(alias.asname or alias.name)

    def _sdk_name(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Name):
            return self.sdk_names.get(node.id)
        if isinstance(node, ast.Attribute) and _dotted(node.value) in self.sdk_modules:
            return node.attr
        return None

    def _find_evaluator(self) -> Optional[str]:
        found: Optional[str] = None
        top_level: set[int] = set()
        for stmt in self.tree.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)
                and self._sdk_name(stmt.value.func) == "evaluator"
            ):
                if found is not None:
                    raise EvaluatorParseError(
                        stmt.lineno, "an evaluator file declares one evaluator()"
                    )
                found = stmt.targets[0].id
                top_level.add(id(stmt.value))
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Call)
                and self._sdk_name(node.func) == "evaluator"
                and id(node) not in top_level
            ):
                raise EvaluatorParseError(
                    node.lineno,
                    "assign evaluator() to a name at module level: ev = evaluator()",
                )
        return found

    def _declaration(self, stmt: ast.stmt, ev: str) -> Optional[Check]:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = [
                d
                for d in stmt.decorator_list
                if _is_ev_attr(d.func if isinstance(d, ast.Call) else d, ev, ("check",))
            ]
            if not decorators:
                return None
            if len(decorators) > 1:
                raise EvaluatorParseError(
                    decorators[1].lineno, "one @check decorator per function"
                )
            decorator = decorators[0]
            if not isinstance(decorator, ast.Call):
                raise EvaluatorParseError(
                    decorator.lineno, f'@{ev}.check needs a name: @{ev}.check("name")'
                )
            self.declarations.add(id(decorator.func))
            return self._check(decorator, stmt)
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Call)
            and _is_ev_attr(stmt.value.func, ev, ("metric",))
        ):
            self.declarations.add(id(stmt.value.func))
            return self._metric(stmt.value, stmt)
        return None

    def _check(self, call: ast.Call, fn: FunctionNode) -> Check:
        name = self._name(call, positional=1)
        options = self._options(call, {"gates", "applies", "labels"})
        labels = options.get("labels")
        if labels is not None:
            if (
                not isinstance(labels, (list, tuple))
                or not labels
                or not all(isinstance(label, str) and label for label in labels)
            ):
                raise EvaluatorParseError(
                    call.lineno, "labels must be a non-empty list of strings"
                )
            if len(set(labels)) != len(labels):
                raise EvaluatorParseError(call.lineno, "labels must be unique")
        start = fn.decorator_list[0].lineno
        assert fn.end_lineno is not None
        return Check(
            name=name,
            kind="label" if labels else "pass_fail",
            gating=options.get("gates", True),
            applies=options.get("applies", "always"),
            label_set=list(labels or []),
            metric=None,
            primitives=self._primitives(fn),
            line_start=start,
            line_end=fn.end_lineno,
            body_hash=self._hash((start, fn.end_lineno)),
        )

    def _metric(self, call: ast.Call, stmt: ast.Expr) -> Check:
        name = self._name(call, positional=2)
        options = self._options(
            call, {"gates", "applies", "rollup", "better", "budget"}
        )
        for key, allowed in (("rollup", ROLLUPS), ("better", BETTER)):
            if key not in options:
                raise EvaluatorParseError(
                    call.lineno, f"metric '{name}' needs {key}= one of {list(allowed)}"
                )
            if options[key] not in allowed:
                raise EvaluatorParseError(
                    call.lineno, f"{key} must be one of {list(allowed)}"
                )
        metric: dict[str, Any] = {
            "rollup": options["rollup"],
            "better_when": options["better"],
        }
        if "budget" in options:
            budget = options["budget"]
            if isinstance(budget, bool) or not isinstance(budget, (int, float)):
                raise EvaluatorParseError(call.lineno, "budget must be a number")
            metric["budget"] = budget
        gating = options.get("gates", "budget" in metric)
        if gating and "budget" not in metric:
            raise EvaluatorParseError(
                call.lineno, f"metric '{name}' gates only with a budget="
            )

        assert stmt.end_lineno is not None
        spans = [(stmt.lineno, stmt.end_lineno)]
        body: ast.AST = call.args[1]
        if isinstance(body, ast.Name) and body.id in self.functions:
            fn = self.functions[body.id]
            assert fn.end_lineno is not None
            spans.append(
                (
                    fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno,
                    fn.end_lineno,
                )
            )
            body = fn
        elif not isinstance(body, ast.Lambda):
            raise EvaluatorParseError(
                call.lineno,
                f"metric '{name}' takes a lambda or a function defined at module level",
            )
        return Check(
            name=name,
            kind="number",
            gating=gating,
            applies=options.get("applies", "always"),
            label_set=[],
            metric=metric,
            primitives=self._primitives(body),
            line_start=stmt.lineno,
            line_end=stmt.end_lineno,
            body_hash=self._hash(*spans),
        )

    def _name(self, call: ast.Call, positional: int) -> str:
        if len(call.args) != positional or any(
            isinstance(arg, ast.Starred) for arg in call.args
        ):
            what = "a name" if positional == 1 else "a name and a function"
            raise EvaluatorParseError(
                call.lineno, f"expected {what} as positional arguments"
            )
        name = _literal(call.args[0], "the check name")
        if not isinstance(name, str) or not CHECK_NAME.fullmatch(name):
            raise EvaluatorParseError(
                call.lineno,
                f"check name must match {CHECK_NAME.pattern}, got {name!r}",
            )
        return name

    def _options(self, call: ast.Call, allowed: set[str]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise EvaluatorParseError(kw.value.lineno, "**options are not allowed")
            if kw.arg not in allowed:
                raise EvaluatorParseError(
                    kw.value.lineno,
                    f"unknown option {kw.arg}=; expected one of {sorted(allowed)}",
                )
            options[kw.arg] = _literal(kw.value, f"{kw.arg}=")
        if "gates" in options and not isinstance(options["gates"], bool):
            raise EvaluatorParseError(call.lineno, "gates must be True or False")
        if "applies" in options and options["applies"] not in APPLIES:
            raise EvaluatorParseError(
                call.lineno, f"applies must be one of {list(APPLIES)}"
            )
        return options

    def _primitives(self, node: ast.AST) -> list[str]:
        called = (
            self._sdk_name(call.func)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
        return sorted(PRIMITIVES.intersection(called))

    def _hash(self, *spans: tuple[int, int]) -> str:
        # Trailing whitespace is ignored; everything outside the spans can't affect the hash.
        text = "\n".join(
            line.rstrip()
            for start, end in spans
            for line in self.lines[start - 1 : end]
        )
        return hashlib.sha256(text.encode()).hexdigest()


def _literal(node: ast.expr, what: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        raise EvaluatorParseError(node.lineno, f"{what} must be a literal") from None


def _is_ev_attr(
    node: ast.AST, ev: str, attrs: tuple[str, ...]
) -> TypeGuard[ast.Attribute]:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == ev
        and node.attr in attrs
    )


def _dotted(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None
