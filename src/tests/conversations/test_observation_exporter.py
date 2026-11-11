from __future__ import annotations

from typing import Any, Callable, Optional

import pytest

import judgeval.conversations.observation_exporter as observation_exporter
from judgeval.conversations.observation_exporter import ObservationExporter
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
        raise ConnectionError("network down")

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


def test_one_observation_that_does_not_fit_the_format_does_not_drop_the_others(waits):
    def reject_bad(batch: list[Any]) -> None:
        if any(item["observation"]["conversation_id"] == "bad" for item in batch):
            raise api_error(400)

    fake = FakeClient(reject_bad)
    export(fake, [sent(name) for name in ["a", "bad", "b", "c"]])
    stored = sorted(
        item["observation"]["conversation_id"]
        for batch in fake.batches
        for item in batch
    )
    assert stored == ["a", "b", "c"]
