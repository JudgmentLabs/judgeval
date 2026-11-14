from __future__ import annotations

import json
from typing import Any, Callable, Optional

import httpx
import pytest

import judgeval.trajectory.observation_exporter as observation_exporter
from judgeval.trajectory.observation_exporter import ObservationExporter
from judgeval.exceptions import JudgmentAPIError


class FakeClient:
    """Records each batch sent; `respond` decides each request's outcome."""

    def __init__(self, respond: Optional[Callable[[list[Any]], None]] = None):
        self.batches: list[list[Any]] = []
        self.attempts = 0
        self._respond = respond

    def post_projects_observations(self, project_id: str, payload: Any) -> Any:
        self.attempts += 1
        batch = payload["observations"]
        if self._respond is not None:
            self._respond(batch)
        self.batches.append(batch)
        return {"status": "success"}


def api_error(status: int, retry_after_seconds: Optional[int] = None):
    return JudgmentAPIError(
        status, "error", None, retry_after_seconds=retry_after_seconds
    )


def sent(conversation_id: str) -> Any:
    return {
        "observation": {"conversation_id": conversation_id, "input": []},
        "observed_at": "2026-11-11T00:00:00+00:00",
    }


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(observation_exporter.time, "sleep", recorded.append)
    return recorded


def export(fake: FakeClient, batch: list[Any]) -> None:
    ObservationExporter(fake, "project-1").export(batch)  # type: ignore[arg-type]


def test_sends_the_batch(waits):
    fake = FakeClient()
    export(fake, [sent("a")])
    assert (fake.batches, waits) == ([[sent("a")]], [])


def test_retries_a_temporary_failure_with_backoff(waits):
    failures = [api_error(503), api_error(503)]

    def fail_twice(batch: list[Any]) -> None:
        if failures:
            raise failures.pop()

    fake = FakeClient(fail_twice)
    export(fake, [sent("a")])
    assert (len(fake.batches), waits) == (1, [1, 2])


def test_waits_at_least_as_long_as_retry_after_but_no_longer_than_the_cap(waits):
    failures = [api_error(429, retry_after_seconds=3600), api_error(429, 5)]

    def rate_limited(batch: list[Any]) -> None:
        if failures:
            raise failures.pop()

    export(FakeClient(rate_limited), [sent("a")])
    assert waits == [5, observation_exporter.MAX_BACKOFF_SECONDS]


def test_a_network_failure_is_dropped_after_the_last_attempt(waits):
    def network_down(batch: list[Any]) -> None:
        raise httpx.ConnectError("network down")

    fake = FakeClient(network_down)
    export(fake, [sent("a")])
    assert (fake.batches, fake.attempts) == ([], observation_exporter.MAX_ATTEMPTS)


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_does_not_retry_a_request_the_server_will_not_accept(waits, status):
    def refuse(batch: list[Any]) -> None:
        raise api_error(status)

    fake = FakeClient(refuse)
    export(fake, [sent("a")])
    assert (fake.attempts, waits) == (1, [])


def rejection(*paths: list[Any]) -> JudgmentAPIError:
    """A 400 as the observations route sends it, with one issue per path."""
    issues = [{"path": path, "message": "Invalid input"} for path in paths]
    return JudgmentAPIError(
        400, json.dumps({"errors": issues, "found": "the request"}), None
    )


def reject_named(*names: str) -> Callable[[list[Any]], None]:
    """Rejects a batch that holds any of `names`, naming each by position."""

    def respond(batch: list[Any]) -> None:
        paths = [
            ["observations", index, "input"]
            for index, item in enumerate(batch)
            if item["observation"]["conversation_id"] in names
        ]
        if paths:
            raise rejection(*paths)

    return respond


def test_drops_only_the_observations_the_server_names(waits):
    # In both halves, which a binary search could not tell from a fault in
    # the request as a whole.
    fake = FakeClient(reject_named("bad-1", "bad-2"))
    batch = [sent(name) for name in ["bad-1", "a", "b", "c", "d", "bad-2"]]
    export(fake, batch)
    assert (fake.attempts, fake.batches) == (
        2,
        [[sent(name) for name in ["a", "b", "c", "d"]]],
    )


def test_logs_each_issue_without_the_echoed_request(waits, caplog):
    export(FakeClient(reject_named("bad")), [sent("a"), sent("bad")])
    assert "observations.1.input: Invalid input" in caplog.text
    assert "the request" not in caplog.text


def test_drops_a_batch_the_server_rejects_without_naming_an_observation(waits):
    def reject(batch: list[Any]) -> None:
        raise rejection(["observations"])

    fake = FakeClient(reject)
    export(fake, [sent(str(index)) for index in range(100)])
    assert (fake.attempts, fake.batches) == (1, [])


def test_drops_a_batch_that_cannot_be_encoded(waits):
    def encode(batch: list[Any]) -> None:
        raise ValueError("Out of range float values are not JSON compliant")

    fake = FakeClient(encode)
    export(fake, [sent("a"), sent("b")])
    assert (fake.attempts, fake.batches, waits) == (1, [], [])
