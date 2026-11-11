from __future__ import annotations

import copy
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import pytest

import judgeval.observations.observations_client as observations_client
from judgeval.exceptions import JudgmentAPIError
from judgeval.observations import Observation, ObservationsClient


class FakeClient:
    """Records each batch sent; `respond` decides each request's outcome."""

    def __init__(self, respond: Optional[Callable[[list[Any]], None]] = None):
        self.batches: list[list[Any]] = []
        self.attempts = 0
        self._respond = respond

    def post_projects_observations(self, project_id: str, payload: Any) -> Any:
        self.attempts += 1
        batch = copy.deepcopy(payload["observations"])
        if self._respond is not None:
            self._respond(batch)
        self.batches.append(batch)
        return {"status": "success"}


def api_error(status: int) -> JudgmentAPIError:
    return JudgmentAPIError(status, "error", None)


def call(conversation_id: str = "ticket-1", text: str = "Hi") -> Observation:
    return {
        "conversation_id": conversation_id,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(observations_client, "FLUSH_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(observations_client.time, "sleep", lambda _: None)


@pytest.fixture
def clients():
    created: list[ObservationsClient] = []

    def make(fake: FakeClient, project_id: Optional[str] = "project-1"):
        client = ObservationsClient(fake, project_id)  # type: ignore[arg-type]
        created.append(client)
        return client

    yield make
    for client in created:
        client.shutdown(timeout_seconds=1)


def test_sends_a_recorded_call_with_its_time_and_environment(clients):
    fake = FakeClient()
    observations = clients(fake)
    observed_at = datetime(2026, 9, 25, 12, 0, 1, tzinfo=timezone.utc)
    observations.record(call(), observed_at=observed_at, environment="production")
    assert observations.flush(timeout_seconds=5)
    assert fake.batches == [
        [
            {
                "observation": call(),
                "observed_at": "2026-09-25T12:00:01+00:00",
                "environment": "production",
            }
        ]
    ]


def test_sends_at_most_one_hundred_calls_per_request(clients):
    fake = FakeClient()
    observations = clients(fake)
    for index in range(250):
        observations.record(call(text=str(index)))
    assert observations.flush(timeout_seconds=5)
    assert [len(batch) for batch in fake.batches] == [100, 100, 50]


def test_a_send_failure_never_reaches_the_caller(clients):
    def fail(batch: list[Any]) -> None:
        raise RuntimeError("network down")

    fake = FakeClient(fail)
    observations = clients(fake)
    observations.record(call())
    assert observations.flush(timeout_seconds=5)
    assert (fake.batches, fake.attempts) == ([], observations_client.MAX_ATTEMPTS)


def test_retries_a_temporary_failure(clients):
    failures = [api_error(503)]

    def fail_once(batch: list[Any]) -> None:
        if failures:
            raise failures.pop()

    fake = FakeClient(fail_once)
    observations = clients(fake)
    observations.record(call())
    assert observations.flush(timeout_seconds=5)
    assert (len(fake.batches), fake.attempts) == (1, 2)


def test_a_call_that_does_not_fit_the_format_does_not_drop_the_others(clients):
    def reject_bad(batch: list[Any]) -> None:
        if any(sent["observation"]["conversation_id"] == "bad" for sent in batch):
            raise api_error(400)

    fake = FakeClient(reject_bad)
    observations = clients(fake)
    for conversation_id in ["a", "bad", "b", "c"]:
        observations.record(call(conversation_id))
    assert observations.flush(timeout_seconds=5)
    sent = sorted(
        sent["observation"]["conversation_id"]
        for batch in fake.batches
        for sent in batch
    )
    assert sent == ["a", "b", "c"]


def test_drops_calls_without_retrying_when_conversations_are_off(clients):
    def hidden(batch: list[Any]) -> None:
        raise api_error(404)

    fake = FakeClient(hidden)
    observations = clients(fake)
    observations.record(call())
    assert observations.flush(timeout_seconds=5)
    assert fake.attempts == 1


def test_records_nothing_without_a_project(clients):
    fake = FakeClient()
    observations = clients(fake, project_id=None)
    observations.record(call())
    assert observations.flush(timeout_seconds=5)
    assert fake.attempts == 0


def test_drops_calls_once_the_queue_is_full(clients, monkeypatch):
    monkeypatch.setattr(observations_client, "MAX_QUEUE_SIZE", 3)
    release = threading.Event()
    fake = FakeClient(lambda batch: release.wait(5))
    observations = clients(fake)
    observations.record(call(text="sending"))
    # Wait until the first call is taken off the queue and being sent.
    for _ in range(500):
        if fake.attempts:
            break
        threading.Event().wait(0.01)
    for index in range(5):
        observations.record(call(text=str(index)))
    release.set()
    assert observations.flush(timeout_seconds=5)
    assert sum(len(batch) for batch in fake.batches) == 4


def test_reads_a_naive_time_as_local_time(clients):
    fake = FakeClient()
    observations = clients(fake)
    local = datetime(2026, 9, 25, 12, 0, 1)
    observations.record(call(), observed_at=local)
    assert observations.flush(timeout_seconds=5)
    sent_at = datetime.fromisoformat(fake.batches[0][0]["observed_at"])
    assert (sent_at.utcoffset(), sent_at) == (timedelta(0), local.astimezone())


def test_sends_the_call_as_it_was_when_recorded(clients):
    fake = FakeClient()
    observations = clients(fake)
    recorded = call()
    observations.record(recorded)
    recorded["conversation_id"] = "changed"
    assert observations.flush(timeout_seconds=5)
    assert fake.batches[0][0]["observation"]["conversation_id"] == "ticket-1"


def test_ignores_calls_recorded_after_shutdown(clients):
    fake = FakeClient()
    observations = clients(fake)
    observations.shutdown(timeout_seconds=1)
    observations.record(call())
    assert observations.flush(timeout_seconds=1)
    assert fake.attempts == 0
